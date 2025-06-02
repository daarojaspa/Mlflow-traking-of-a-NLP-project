""" THIS file contains all the utils functions for the orchestration, functions that are not tasks or flows"""

import os
from config import PATH
import pickle
import pandas as pd

def labels_decode (labels:pd.Series,idx2labels:dict)->pd.Series:
  """This function decode the labels into idx
   Args:
   labels (pd.Series): series with the labels
  idx2label (dict): dictionary with the mapping
  Returns:
  labels (pd.Series): series with the labels decoded
  """
  return labels.map(idx2labels)
def save_oikle(data:object,filename:str)->None:
  filepath=os.path.join(DATA_PATH_PROCESSED,f"{filename}.pkl")
  with open(filepath,'wb')as file:
    pickle.dump(data,file)
  
def load_pikle(data:object,filename:str)->object:
  """
    This function loads data from a pickle file.
    Args:
        filename (str): filename.
    Returns:
        data (object): data loaded from the pickle file.
    """
  filepath = os.path.join(DATA_PATH_PROCESSED, f"{filename}.pkl")
  with open(filepath, 'rb') as file:
        data = pickle.load(file)
  return data