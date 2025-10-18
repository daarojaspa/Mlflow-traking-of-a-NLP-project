# BERT Pipeline Documentation

This folder contains the BERT-based topic modeling and routing pipeline for processing customer complaints. The pipeline uses Sentence Transformers for embeddings, BERTopic for topic modeling, and Prefect for orchestration.

## Files Overview

### topicModeling.py - TicketTopicModeling Class
Main pipeline class for processing tickets and performing topic modeling.

**Key Features:**
- Loads JSON data and transforms it (keeps complaint text, product, sub-product)
- Splits data into train/test sets and saves to timestamped folders
- Generates embeddings using SentenceTransformer
- Trains BERTopic model on training data and transforms test data
- Saves models (embedder and BERTopic) to artifacts directories
- Outputs labeled datasets with topic assignments in the folder splited, inside the Data  folder on the root of the proyect, as a csv format.

**Usage:**
```python
model = TicketTopicModeling(model_name="all-MiniLM-L6-v2")
train_labeled, test_labeled = model.run(path, file_name)
```

**Inputs:**

- the path and the filename of the raw data  `Data/data_raw/`.

**Outputs:**

- Train/test CSV with description of the situation and bert labels splits in `Data/splitted/`
- Embedder model in `Data/artifacts/embedders/`
- BERTopic model in `Data/artifacts/BERT/

**Note:** All outputs include timestamped folders and "latest" symbolic copies for easy access.
### topicRouter.py - TopicRouter Class
Routes discovered topics to appropriate departments using semantic similarity (cosine).

**Key Features:**
- Loads saved BERTopic model and SentenceTransformer from local directories
- Extracts topic embeddings from the loaded BERTopic model
- Embeds department names using the loaded SentenceTransformer
- Maps topics to departments via cosine similarity scoring
- Saves topic-department mapping to timestamped CSV files with "latest" folder copy
                
**Usage:**
```python
router = TopicRouter(model_path, departments, threshold=0.5)
df = router.run(sentence_model_dir)
```

**Inputs:**
- `model_path` (str): Path to saved BERTopic model directory
- `departments` (list): List of department names for routing
- `sentence_model_dir` (str): Path to saved SentenceTransformer model directory
- `threshold` (float): Minimum cosine similarity for department assignment (default 0.5)

**Outputs:**
- Topic-department mapping DataFrame with columns: topic_id, topic_label, mapped_department, similarity
- CSV file saved to `Data/routed/YYYY-MM-DD/topic_department_mapping_YYYY-MM-DD.csv`
- Latest copy updated in `Data/routed/latest/`

### pipelineEnsamble.py - Prefect Orchestration
Orchestrates the pipelines using Prefect flows and tasks.

**Flows:**
- `ticket_pipeline_flow`: Orchestrates the topic modeling pipeline
- `topic_routing_flow`: Orchestrates the topic routing pipeline

**Tasks:**
- `run_topic_pipeline`: Prefect task for topic modeling with retries and error handling
- `run_topic_routing`: Prefect task for topic routing with retries and error handling

**Usage:**
```python
# Run topic modeling pipeline
result = ticket_pipeline_flow(path, file_name)

# Run topic routing pipeline
result = topic_routing_flow(model_path, departments, sentence_model_dir, threshold=0.5)
```

### bitacora.md - Research Notes
Detailed notes on topic modeling concepts, data leakage prevention, and topic alignment challenges.

**Key Topics:**
- Sentence Transformers usage
- BERTopic architecture and limitations
- Why topic alignment across separate models is unreliable
- Strategies for consistent topic labeling (train once, transform; shared UMAP/HDBSCAN)

### departments.json - Department Mappings
JSON file defining department categories for routing.

**Structure:**
```json
{
    "0": "Billing",
    "1": "Technical Support",
    "2": "Account Management",
    "-1": "Other"
}
```

Used by TopicRouter to map topics to human-readable departments.

## Pipeline Flow

1. **Data Processing (topicmodeling.py)**: Load JSON → Transform → Split → Embed → Model → Label
2. **Topic Routing (topicRouter.py)**: Load models → Extract embeddings → Map topics to departments
3. **Orchestration (pipelineEnsamble.py)**: Run pipelines with Prefect for reliability and monitoring

## Dependencies
- bertopic
- sentence-transformers
- prefect
- pandas
- numpy
- scikit-learn

## Data Flow
- Raw JSON → Processed DataFrames → Embeddings → Topics → Labeled Data → Department Routing
