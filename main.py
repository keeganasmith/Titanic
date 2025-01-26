import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

def prepare_data(df):

    df.dropna(inplace = True)
    df.drop(columns = ["PassengerId", "Name", "Cabin", "Ticket"], inplace=True)
    print(df.dtypes)
    print(df["Embarked"].unique())
    label_encoder = LabelEncoder()
    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    df["Embarked"] = label_encoder.fit_transform(df["Embarked"])
    
def train_nn(df):
    print(df.dtypes)
    X = df.drop(columns=["Survived"]).values
    Y = df["Survived"].values
    
def main():
    df = pd.read_csv("./data/train.csv")   
    print(df.head())
    prepare_data(df)
    train_nn(df)
if __name__ == "__main__":
    main()