# Repo overvew

introMlflow.ipynb: is a notebook where i  explain the basic usage of the mlflow traking uris  and loging

utils: In this folder i have 2 python scripts that are involve in the nlp pipeline it is important to run them in ordetr first
 text processing.py (preprocessing of the data) and then featureExtraction.py.

tracking: here is the Data raw and preprocessed that the EDA , and textProcessing, featureExtraction will use
tracking/base_line.ipynb: after having our tickets_input_n.csv file where  our topics have a correct label humanly made, now we palay using diferent  ml models to see which one has better metrics.
Data_raw: has a json file, and a little jupyter notebook  where i do a first  esploration of the raw dataset.
data_processed:the tickets input file that was humanly revew to  do the labeling in a second version  and the vectorizer
pkl  assets from traking_data_baseline  for the  construction of the next model that will be used in production.
