import pandas as pd


train = pd.read_csv("../data/original/test.csv", index_col=0)

train.to_csv("../data/prepped/test.csv")
