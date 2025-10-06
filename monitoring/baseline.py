

# %%
import numpy as np
import pandas as pd 
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# %% 
data = pd.read_csv("data/tickets_inputs_eng_2.csv")
data.describe()

# %%
data.head(5)
# %% 
data["relevant_topics"].value_counts()