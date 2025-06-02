# Standard library imports
import os
import warnings

# Third party imports
import numpy as np
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_recall_fscore_support, 
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from prefect import flow, task

# Local application imports
from config import *
from utils import * 
from featuresExtraction import FeatureExtraction
from textprocessing import TextProcessing

warnings.filterwarnings("ignore")


@task(retries=3, retry_delay_seconds=2,
      name="text_processing_task",
      tags=["pos_tag"])
def text_processing_task(language: str, file_name: str, version: int):
    """This task is used to run the text processing process
    Args:
        language (str): language of the text
        file_name (str): file name of the data
        version (int): version of the data
    Returns:
        None
    """
    text_processing_processor = TextProcessing(language=language)
    text_processing_processor.run(file_name=file_name, version=version)


@task(retries=3, retry_delay_seconds=2,
      name="feature_extraction_task",
      tags=["feature_extraction", "topic_modeling"])
def feature_extraction_task(data_path_processed: str, data_version: int) -> None:
    """This task is used to run the feature extraction process
    Args:
        data_path_processed (str): path where the data is stored
        data_version (int): version of the data
    Returns:
        None
    """
    feature_extraction_processer = FeatureExtraction()
    feature_extraction_processer.run(data_path_processed=data_path_processed,
                                     data_version=data_version)


@task(retries=3, retry_delay_seconds=2,
      name="Data transformation task",
      tags=["data_transform", "split_data", "train_test_split"])
def data_transform_task(data_input_path: str, filename: str, version: int):
    """This function transforms the data into X and y
    Args:
        data_input_path (str): path to the input data
        filename (str): base filename
        version (int): data version
    Returns:
        Tuple of transformed data: (X_train, X_test, y_test, y_train, X_vectorized)
    """
    # Read data
    df = pd.read_csv(os.path.join(data_input_path, f"{filename}{version}.csv"))
    X = df['processed_text']
    y = df['relevant_topics']

    # Feature extraction for text input
    count_vectorizer = CountVectorizer()
    X_vectorized = count_vectorizer.fit_transform(X)

    # Transform labels into index
    y = decode_labels(labels=y, idx2label=label2idx)

    # Transform into tf-idf data
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_vectorized)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=37
    )

    print("Data transformation and split task successfully completed")
    return X_train, X_test, y_test, y_train, count_vectorizer


@task(
    retries=3,
    retry_delay_seconds=2,
    name="Train best model",
    tags=["train", "best_model", "LogisticRegressionClassifier"],
)
def train_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
    model_name: str,
):
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("developer", DEVELOPER_NAME)
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_train_pred_proba = model.predict_proba(X_train)
        y_test_pred_proba = model.predict_proba(X_test)

        roc_auc_score_train = round(
            roc_auc_score(y_train, y_train_pred_proba, average="weighted", multi_class="ovr"), 2
        )
        roc_auc_score_test = round(
            roc_auc_score(y_test, y_test_pred_proba, average="weighted", multi_class="ovr"), 2
        )

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        precision_train, recall_train, fscore_train, _ = precision_recall_fscore_support(
            y_train, y_train_pred, average="weighted"
        )
        precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average="weighted"
        )

        mlflow.log_metrics({
            "roc_auc_train": roc_auc_score_train,
            "roc_auc_test": roc_auc_score_test,
            "precision_train": precision_train,
            "precision_test": precision_test,
        })

        mlflow.sklearn.log_model(model, f"model_{MODEL_NAME}")

        save_pickle(model, "model_lr")

        metric_data = [
            roc_auc_score_train,
            roc_auc_score_test,
            round(precision_train, 2),
            round(precision_test, 2),
            round(recall_train, 2),
            round(recall_test, 2),
            round(fscore_train, 2),
            round(fscore_test, 2),
        ]

        print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred)}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")

        print("Classification Report for Train:\n", classification_report(y_train, y_train_pred))
        print("Classification Report for Test:\n", classification_report(y_test, y_test_pred))

        return metric_data


@flow
def main_flow():
    #ter parameters here are the variables set in the config  file
    text_processing_task(language=LANGUAGE,file_name=FILE_NAME_DATA, version=VERSION)
    feature_extraction_task(data_path_processed=PATH, data_version=VERSION)

    X_train, X_test, y_test, y_train, count_vectorizer = data_transform_task(
        data_input_path=PATH,
        filename="tickets_inputs_eng",
        version=VERSION
    )

    save_pickle((X_train, y_train), "train")
    save_pickle((X_test, y_test), "test")
    save_pickle(count_vectorizer, "count_vectorizer")

    print("Data transformation and split task successfully completed and stored in pickle files")

    metrics_classification = train_best_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=PARAMETERS_MODEL,
        model_name=MODEL_NAME
    )

    print(metrics_classification)


main_flow()


      