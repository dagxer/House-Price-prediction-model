import pandas as pd
import torch
import torch.nn as nn


# the below two varibles load up the datasets train and test csv files
train = pd.read_csv("datasets\train.csv")
test = pd.read_csv("datasets\test.csv")

# the y value stemmed from the idea of y = ln(1+x) which summed up make the difference in mansions and duplex's not freak the model out
y = torch.log1p(torch.tensor(train["SalePrice"].values, dtype = torch.float32)).unsqueeze(1)

# this line is going to take the training data, drop the saleprice column because thats pretty much the answer key and we drop the PID's because it will skew the data
x = pd.concat([train.drop(columns=["SalePrice"]), test]).drop(columns=["PID"])
# converts the data into 1.0 and 0.0 form
x = pd.get_dummies(x).astype("float32")
# fills the empty sections with the median to avoid getting zeros because that can mess up the math
x = x.fillna(x.median())
# this takes the values of x and converts everything into a smaller value for example say you have a house with 2 BR's and 2000 SqFt then [2, 2000] the network then will only look at sqft because its overpowering the BR's
x = (x - x.median()) / (x.std() + 1e-8) #1e-8 is added to prevent division by 0
