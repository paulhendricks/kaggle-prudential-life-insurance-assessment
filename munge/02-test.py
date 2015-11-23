import pandas as pd


train = pd.read_csv("../data/original/test.csv")

train.to_csv("../data/prepped/test.csv")
