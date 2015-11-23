import pandas as pd


train = pd.read_csv("../data/prepped/train.csv")
test = pd.read_csv("../data/prepped/test.csv")

train.head()
train.describe()