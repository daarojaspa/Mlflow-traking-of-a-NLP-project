import os
import shutil
from datetime import datetime
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from datetime import datetime
from prefect import flow, task
from 1 import TicketTopicModeling
from 2 import TopicRouter

@task(retries=3, retry_delay_seconds=2,
      name="topic modeling",
      tags=["pos_tag"])
def run_topic_pipeline(path: str, file_name: str):
    """
    Prefect task that runs the full TicketTopicModeling pipeline.
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


@task(name="Run Topic Routing Pipeline", retries=2, retry_delay_seconds=60)

def run_topic_routing(model_path: str, departments: list, sentence_model_dir: str, threshold: float = 0.5):
    """
    Prefect task that runs the full TopicRouter pipeline.
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
    Orchestrates the ticket pipeline task.
    """
    result = run_ticket_pipeline(path, file_name)
    print("✅ Pipeline finished")
    return result


if __name__ == "__main__":
    # Example execution (replace with your actual JSON location + filename)
    result = ticket_pipeline_flow(
        path="/path/to/json",
        file_name="tickets.json"
    )
    print(result)
    result = topic_routing_flow(
    model_path="/path/to/bertopic_model",
    departments=["HR", "Finance", "IT", "Support"],
    sentence_model_dir="/path/to/sentence_transformer"
)
    print(result)