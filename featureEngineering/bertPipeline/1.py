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
        self.fileDir = os.path.dirname(os.path.abspath(__file__))
# will store where the splits are saved

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
        Splits the DataFrame into train/test sets and saves them into both a timestamped
        folder and a 'latest' folder for easy access.
        Returns (train_df, test_df).
        """
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        self.save_data_csv({'train.csv': train_df, 'test.csv': test_df})
        print(f"✅ Train and test sets saved to {self.output_dir} and updated 'latest'")
        return train_df, test_df


    def save_data_csv(self, filenames: dict, subdir: str = "splitted"):
        """
        Save multiple DataFrames to CSV files inside ../Data/<subdir>.
        
        filenames: dict[str, pd.DataFrame], e.g. {'train.csv': train_df}
        subdir: which subfolder inside ../Data (default = 'splitted')
        """
        # Create timestamped folderfrom datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d")  # Year-Month-Day
        base_dir = os.path.join(self.fileDir, f"../Data/{subdir}")
        self.output_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)


        # Save CSVs in timestamped folder
        for name, df in filenames.items():
            path = os.path.join(self.output_dir, name)
            df.to_csv(path, index=False)

        # Update "latest" folder (clear it first, then copy new files)
        latest_dir = os.path.join(base_dir, "latest")
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)  # remove old "latest"
        shutil.copytree(self.output_dir, latest_dir)


    def embedding_creation(self, df: pd.DataFrame, text_column: str = "complaint_what_happened") -> np.ndarray:
        """
        Creates embeddings for a given text column using a pre-trained transformer.
        """
        texts = df[text_column].tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        print("Embeddings shape:", embeddings.shape)
        self.save_artifact(self.embedder,"./artifacts/embedders")
        return embeddings

    def create_train_test_embeddings(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Loads train/test datasets from CSVs and creates embeddings for each.
        """
        train_embeddings = self.embedding_creation(train_df, "complaint_what_happened")
        test_embeddings = self.embedding_creation(test_df, "complaint_what_happened")

        return train_embeddings, train_df["complaint_what_happened"].tolist(), test_embeddings, test_df["complaint_what_happened"].tolist()

    def save_artifact(self,model, subdir: str ):
        """
        Saves a BERTopic model into a timestamped folder.
        or a sentence transformer
        Parameters
        ----------
        topic_model : BERTopic
            The BERTopic model to save.
        base_output_dir : str
            The base directory where the model folder will be created.
        
        Returns
        -------
        str
            The full path where the model was saved.
        """
        # Create a timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y-%m-%d")  # Year-Month-Day
        base_dir = os.path.join(self.fileDir, f"../../Data/{subdir}")
        self.output_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the BERTopic model
        if subdir=="./artifacts/BERT":
            model.save(self.output_dir,serialization='safetensors')
            print(f"✅ bert Model saved at: {self.output_dir}")
        else:
            model.save(self.output_dir)
            print(f"✅ sentence transformer saved at: {self.output_dir}")
            
        return self.output_dir

    
    
    
    def bert_modeling(self, train_embeddings, train_texts, test_embeddings, test_texts):
            """Run BERTopic modeling using precomputed embeddings and texts"""
            topic_model= BERTopic()
            topics_train, probs_train = topic_model.fit_transform(
                train_texts, train_embeddings
            )
            info_train = topic_model.get_topic_info()
            train_label_map = dict(zip(info_train["Topic"], info_train["Name"]))
            """ if i am not remembering wrongly i do the fit again because my intention at the time was to avoid any data lekage  so the labels for the training and the test set should be inde pendent from each other
                now i realize i could do a iterative  active learnning pipeline instaed, but maybe for other iteration
            """
            topics_test, probs_test = topic_model.transform(
                test_texts, test_embeddings
            )
            
    # save modeling bert artifact
            self.save_artifact(topic_model,"./artifacts/BERT")
    # Create labeled DataFrames with human-readable labels
            train_labeled = pd.DataFrame({
                "text": train_texts,
                "topic": topics_train,
                "probability": probs_train,
            })
            train_labeled["topic_label"] = train_labeled["topic"].map(train_label_map)

            test_labeled = pd.DataFrame({
                "text": test_texts,
                "topic": topics_test,
                "probability": probs_test,
            })
            test_labeled["topic_label"] = test_labeled["topic"].map(train_label_map)

            # Save results (now include human-readable labels)
            self.save_data_csv(
                {"train_labeled.csv": train_labeled, "test_labeled.csv": test_labeled},
                subdir="label",
            )

            return train_labeled, test_labeled
    def run(self, path: str, file_name: str):
        """recives the path to the jason file and the name of the jason file"""
        """
        Full pipeline:
        1. Read JSON
        2. Transform
        3. Split into train/test and save to disk
        4. Create embeddings
        5. Train/test BERTopic
        """
     
        raw_df = self.read_json(path, file_name)
        df = self.data_transform(raw_df)

        train_df, test_df = self.split(df)

        train_embeddings, train_texts, test_embeddings, test_texts = self.create_train_test_embeddings(train_df, test_df)

        train_labeled, test_labeled = self.bert_modeling(
            train_embeddings, train_texts, test_embeddings, test_texts
        )

        return train_labeled, test_labeled

if __name__ == "__main__":
    path= 'Data/data_raw' #if the file is ran from the root of the proyect
    file_name='smoke.json'
    model = TicketTopicModeling(
            model_name="all-MiniLM-L6-v2",
            output_dir="splits"
        )
    train_labeled, test_labeled = model.run(path, file_name)        