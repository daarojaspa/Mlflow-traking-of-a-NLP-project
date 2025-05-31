from prefect import task, flow
from datetime import timedelta
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pprint
@task (
    name="load iris Dataset",
    tags=["data_ingestion"],
    description="normally very important"
)
def get_data()->dict:
    data =load_iris()
    return {"data":data.data,"target":data.target}
@task(
    name="split_data",
    tags=['splitting'],
    description="is not self explenatory?"
)
def splitting(dataset:dict)->tuple:
    "train and set split"
    X_train,X_test,y_train,y_test=train_test_split(
            dataset["data"],dataset["target"],test_size=0.2,random_state=41
    )
    return X_train,X_test,y_train,y_test
@task(
    name="train_model",
    tags=['rainning'],
    description="tainning  of forest clasifier"
)
def train(X_train:list,X_test:list,y_train:list,y_test:list)->str:
    """it returns acuracy metrics after trainning"""
    model=RandomForestClassifier(random_state=37)
    model.fit(X_train,y_train)
    accuracy=model.score(X_test,y_test)
    pprint.pprint("el accuracy es :"+ str(accuracy))
    return f"model train with accuray:{accuracy}"
    
@flow(
    retries=3,
    retry_delay_seconds=5,
    log_prints=True
)
def iris_clasification()->None:
    """this will orchestrait the hole flow """
    data=get_data()
    X_train, X_test, y_train, y_test = splitting(data)
    train(X_train, X_test, y_train, y_test)


iris_clasification()