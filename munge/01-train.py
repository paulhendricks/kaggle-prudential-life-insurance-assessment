import pandas as pd


train = pd.read_csv("../data/original/train.csv", index_col=0)

train.to_csv("../data/prepped/train.csv")
__author__ = 'paul'
