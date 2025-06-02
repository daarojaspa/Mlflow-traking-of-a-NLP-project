"""this file contains the configuration parameters for the workflows using prefext"""
#path with data process
PATH="/home/dan/PLATZI/data/MLops/localMlflow/orchestration/Data/Processed"
#version of the data
VERSION=2
#languaje for the input parameter for  the text processing class
LANGUAJE="english"
#file data name for the input for the text processing class
FILE_NAME_DATA="english_clasification_english"
#parameters took it from mlflow register of the best model
PARAMETERS_MODEL = {
    "C": 1.0,
    "class_weight": None,
    "l1_ratio": None,
    "max_iter": 100,
    "penalty": "l2",
    "random_state": 40,
    "solver": "liblinear",
    "tol": 0.0001,
}

idx2label = {"0": "Bank Account Services", "1": "Credit Report or Prepaid Card", "2": "Mortgage/Loan"}
label2idx = {v: k for k, v in idx2label.items()}
# tags for mlflow tracking
DEVELOPER_NAME = "Maria"
MODEL_NAME = "LogisticRegression"