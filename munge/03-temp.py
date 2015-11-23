import pandas as pd


train = pd.read_csv("../data/prepped/train.csv")
test = pd.read_csv("../data/prepped/test.csv")

columns_of_interest = ['Ins_Age', 'Ht', 'Wt', 'BMI']


train = train[columns_of_interest]
test = test[columns_of_interest]

train.head()
train.describe()