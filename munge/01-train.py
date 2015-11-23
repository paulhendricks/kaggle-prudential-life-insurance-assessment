import pandas as pd


train = pd.read_csv("../data/original/train.csv")

train.to_csv("../data/prepped/train.csv")
__author__ = 'paul'
