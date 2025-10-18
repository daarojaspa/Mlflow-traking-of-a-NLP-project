import shutil
from pathlib import Path
import os
import numpy as np
import pandas as pd
from datetime import datetime
from bertopic import BERTopic
from sentence_transformers import  util,SentenceTransformer
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TopicRouter:
    def __init__(self, model_path: str, departments: list, threshold: float = 0.5):
        """
        Args:
            model_path (str): Path to a saved BERTopic model.
            departments (list): List of department names to map topics to.
            threshold (float): Minimum cosine similarity to accept mapping.
        """
        self.model_path = model_path
        self.departments = departments
        self.threshold = threshold
        self.topic_model = None
        self.topic_embeddings = None
        self.dept_embeddings = None
        self.sentence_transformer=None

        # Base directory for saving results
        self.base_dir = os.path.join(str(PROJECT_ROOT), "Data", "routed")
        self.output_dir = None  # Will be set in save_mapping

    # 1. Load the BERTopic model
    def load_model(self):
        self.topic_model = BERTopic.load(self.model_path)
        return self.topic_model

    # 2. Get topic embeddings
    def get_topic_embeddings(self):
        if self.topic_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        self.topic_embeddings = self.topic_model.topic_embeddings_
        return self.topic_embeddings

    # 3. Load sentence transformer and embed department names
    def load_sentence_transformer(self,model_dir: str) -> SentenceTransformer:
        """
        Load a saved SentenceTransformer model from a local directory.

        Args:
            model_dir (str): Path to the folder where the model was saved
                             using model.save("path/to/folder").

        Returns:
            SentenceTransformer: The loaded model ready for inference.
        """
        model_dir_str = str(model_dir)
        if not os.path.exists(model_dir_str):
            raise FileNotFoundError(f"Model directory not found: {model_dir_str}")

        self.sentence_transformer = SentenceTransformer(model_dir_str)
        print(f"✅ Loaded SentenceTransformer model from {model_dir_str}")
        return self.sentence_transformer

    def embed_departments(self):
        embedder = self.sentence_transformer
        self.dept_embeddings = embedder.encode(self.departments, normalize_embeddings=True)
        return self.dept_embeddings

    # 4. Apply human-readable labels from BERTopic
    def get_topic_label(self, topic_id: int):
        if topic_id == -1:
            return "Noise/Outlier"
        return self.topic_model.get_topic_info().set_index("Topic").loc[topic_id]["Name"]

    # 5. Create topic → department mapping
    def create_mapping(self):
        if self.topic_embeddings is None or self.dept_embeddings is None:
            raise ValueError("Embeddings not ready. Call get_topic_embeddings() and embed_departments().")
        valid_topics = self.topic_model.get_topic_info()["Topic"].tolist()
        valid_topics = [t for t in valid_topics if t != -1]  # exclude noise if needed

        results = []

        for topic_id in valid_topics:
            if topic_id == -1:  # noise topic
                results.append({
                    "topic_id": -1,
                    "topic_label": "Noise/Outlier",
                    "mapped_department": "Unassigned",
                    "similarity": None
                })
                continue

            topic_vector = self.topic_embeddings[topic_id]
            sims = util.cos_sim(topic_vector, self.dept_embeddings)[0].cpu().numpy()

            best_idx = np.argmax(sims)
            best_score = sims[best_idx]

            if best_score >= self.threshold:
                assigned_dept = self.departments[best_idx]
            else:
                assigned_dept = "Unassigned"

            results.append({
                "topic_id": topic_id,
                "topic_label": self.get_topic_label(topic_id),
                "mapped_department": assigned_dept,
                "similarity": round(float(best_score), 3)
            })

        return results

    # 6. Convert to DataFrame and save mapping
    def save_mapping(self, mapping: list):
        df = pd.DataFrame(mapping)

        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y-%m-%d")  # Year-Month-Day
        self.output_dir = os.path.join(self.base_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        out_path = os.path.join(self.output_dir, f"topic_department_mapping_{timestamp}.csv")
        df.to_csv(out_path, index=False)

        # Update "latest" folder
        latest_dir = os.path.join(self.base_dir, "latest")
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)  # remove old "latest"
        shutil.copytree(self.output_dir, latest_dir)

        print(f"Mapping saved to {out_path}")
        print(f"Latest copy updated at: {latest_dir}")
        return df


    def run(self, sentence_model_dir: str):
        """
        Run the full pipeline:
          1. Load BERTopic model
          2. Extract topic embeddings
          3. Load SentenceTransformer
          4. Embed department names
          5. Create mapping
          6. Save mapping to CSV
        """
        # Step 1: load BERTopic model
        self.load_model()

        # Step 2: get topic embeddings
        self.get_topic_embeddings()

        # Step 3: load SentenceTransformer from local dir
        self.load_sentence_transformer(sentence_model_dir)

        # Step 4: embed department names
        self.embed_departments()

        # Step 5: create mapping
        mapping = self.create_mapping()

        # Step 6: save mapping
        df = self.save_mapping(mapping)

        return df

if __name__ == "__main__":
    model_path=str(PROJECT_ROOT/"Data"/"artifacts"/"BERT"/"latest")
    sentence_model_dir=str(PROJECT_ROOT/"Data"/"artifacts"/"embedders"/"latest")
    departments=["Bank Account Services",
            "Credit Report or Prepaid Card",
         "Mortgage/Loan"]

    router = TopicRouter(model_path, departments, threshold=0.5)
    df = router.run(sentence_model_dir)
    