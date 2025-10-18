from pathlib import Path
from prefect import flow, task
from topicmodeling import TicketTopicModeling
from topicRouter import TopicRouter

@task(retries=3, retry_delay_seconds=2, name="Run Topic Modeling Pipeline")
def run_topic_pipeline(path: str, file_name: str):
    """
    Prefect task that runs the full TicketTopicModeling pipeline.

    Args:
        path (str): Path to the directory containing the JSON file
        file_name (str): Name of the JSON file to process

    Returns:
        dict: Summary of processing results
    """
    model = TicketTopicModeling(
        model_name="all-MiniLM-L6-v2",
        output_dir="splits"
    )
    train_labeled, test_labeled = model.run(path, file_name)
    return {
        "train_rows": len(train_labeled),
        "test_rows": len(test_labeled),
        "train_sample": train_labeled.head(3).to_dict(orient="records"),
        "test_sample": test_labeled.head(3).to_dict(orient="records"),
    }


@task(retries=2, retry_delay_seconds=60, name="Run Topic Routing Pipeline")
def run_topic_routing(model_path: str, departments: list, sentence_model_dir: str, threshold: float = 0.5):
    """
    Prefect task that runs the full TopicRouter pipeline.

    Args:
        model_path (str): Path to saved BERTopic model directory
        departments (list): List of department names for routing
        sentence_model_dir (str): Path to saved SentenceTransformer model directory
        threshold (float): Similarity threshold for department assignment

    Returns:
        dict: Summary of routing results
    """
    router = TopicRouter(
        model_path=model_path,
        departments=departments,
        threshold=threshold
    )
    df = router.run(sentence_model_dir)
    return {
        "rows": len(df),
        "sample": df.head(5).to_dict(orient="records"),
    }


@flow(name="Topic Routing Flow")
def topic_routing_flow(model_path: str, departments: list, sentence_model_dir: str, threshold: float = 0.5):
    """
    Orchestrates the TopicRouter pipeline task.
    """
    result = run_topic_routing(model_path, departments, sentence_model_dir, threshold)
    print("✅ Topic routing finished")
    return result

@flow(name="Ticket Topic Modeling Flow")
def ticket_pipeline_flow(path: str, file_name: str):
    """
    Orchestrates the ticket topic modeling pipeline.

    Args:
        path (str): Path to the directory containing the JSON file
        file_name (str): Name of the JSON file to process

    Returns:
        dict: Results from the topic modeling pipeline
    """
    result = run_topic_pipeline(path, file_name)
    print("✅ Topic modeling pipeline finished")
    return result


if __name__ == "__main__":
    # Resolve project root dynamically
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Example 1: Run topic modeling pipeline
    print("🚀 Running topic modeling pipeline...")
    result = ticket_pipeline_flow(
        path=str(PROJECT_ROOT / "Data" / "data_raw"),
        file_name="smoke.json"
    )
    print("Topic modeling result:", result)

    # Example 2: Run topic routing pipeline (after models are trained)
    print("\n🚀 Running topic routing pipeline...")
    result = topic_routing_flow(
        model_path=str(PROJECT_ROOT / "Data" / "artifacts" / "BERT" / "latest"),
        departments=[
            "Bank Account Services",
            "Credit Report or Prepaid Card",
            "Mortgage/Loan"
        ],
        sentence_model_dir=str(PROJECT_ROOT / "Data" / "artifacts" / "embedders" / "latest"),
        threshold=0.5
    )
    print("Topic routing result:", result)
