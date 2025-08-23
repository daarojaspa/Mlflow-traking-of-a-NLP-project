import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from datetime import datetime


class TicketTopicModeling:
    """
    A pipeline class to process tickets, generate embeddings,
    split train/test sets, and perform topic modeling with BERTopic.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", output_dir: str = "splits"):
        """
        Initialize with a pre-trained sentence transformer and an output directory.
        """
        self.embedder = SentenceTransformer(model_name)
        self.output_dir = output_dir
        self.train_results = None
        self.test_results = None
        self.split_dir = None  # will store where the splits are saved

    def read_json(self, path: str, file_name: str) -> pd.DataFrame:
        """
        Reads a JSON file into a pandas DataFrame.
        """
        file_path = os.path.join(path, file_name)
        with open(file_path, "r") as file:
            datos = json.load(file)
        df_tickets = pd.json_normalize(datos)
        return df_tickets

    def data_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the raw dataframe:
        - Keeps only needed columns
        - Renames them
        - Creates 'ticket_classification' by combining product + sub_product
        - Drops empty rows
        - Applies text preprocessing (lowercase + strip)
        """
        df = df[
            [
                "_source.complaint_what_happened",
                "_source.product",
                "_source.sub_product",
            ]
        ]

        df = df.rename(
            columns={
                "_source.complaint_what_happened": "complaint_what_happened",
                "_source.product": "category",
                "_source.sub_product": "sub_product",
            }
        )

        df["ticket_classification"] = df["category"] + " + " + df["sub_product"]

        df = df.drop(["sub_product", "category"], axis=1)
        df["complaint_what_happened"] = df["complaint_what_happened"].replace("", np.nan)

        df = df.dropna(subset=["complaint_what_happened", "ticket_classification"])
        df = df.reset_index(drop=True)

        df["complaint_what_happened"] = (
            df["complaint_what_happened"].str.lower().str.strip()
        )

        return df

    def split(self, df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
        """
        Splits the DataFrame into train/test sets and saves them into timestamped folder.
        return the df splotted
        """
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.split_dir = os.path.join(self.output_dir,'splitted', timestamp)
        os.makedirs(self.split_dir, exist_ok=True)

        # Save CSVs
        train_path = os.path.join(self.split_dir, "train.csv")
        test_path = os.path.join(self.split_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train and test sets saved to {self.split_dir}")
        return test_df,train_df

    def embedding_creation(self, df: pd.DataFrame, text_column: str = "complaint_what_happened") -> np.ndarray:
        """
        Creates embeddings for a given text column using a pre-trained transformer.
        """
        texts = df[text_column].tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        print("Embeddings shape:", embeddings.shape)
        return embeddings

    def create_train_test_embeddings(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Loads train/test datasets from CSVs and creates embeddings for each.
        """
        train_embeddings = self.embedding_creation(train_df, "complaint_what_happened")
        test_embeddings = self.embedding_creation(test_df, "complaint_what_happened")

        return train_embeddings, train_df["complaint_what_happened"].tolist(), test_embeddings, test_df["complaint_what_happened"].tolist()

    def BERT_modeling(self, train_embed: np.ndarray, train_texts: list, test_embed: np.ndarray, test_texts: list):
        """
        Trains BERTopic separately on train and test sets.
        """
        topic_model_train = BERTopic()
        topics_train, probs_train = topic_model_train.fit_transform(train_embed, train_texts)

        topic_model_test = BERTopic()
        topics_test, probs_test = topic_model_test.fit_transform(test_embed, test_texts)

        self.train_results = (topic_model_train, topics_train, probs_train)
        self.test_results = (topic_model_test, topics_test, probs_test)

        return self.train_results, self.test_results

    def run_pipeline(self, path: str, file_name: str):
        """
        Full pipeline:
        1. Read JSON
        2. Transform
        3. Split into train/test and save to disk
        4. Create embeddings
        5. Train/test BERTopic
        """
        df = self.read_json(path, file_name)
        df = self.data_transform(df)
        train_path, test_path = self.split(df)
        train_embed, train_texts, test_embed, test_texts = self.create_train_test_embeddings(train_path, test_path)
        return self.BERT_modeling(train_embed, train_texts, test_embed, test_texts)
