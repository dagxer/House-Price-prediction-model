import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# the below two varibles load up the datasets train and test csv files
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

# the y value stemmed from the idea of y = ln(1+x) which summed up make the difference in mansions and duplex's not freak the model out
y = torch.log1p(torch.tensor(train["SalePrice"].values, dtype = torch.float32)).unsqueeze(1)

# this line is going to take the training data, drop the saleprice column because thats pretty much the answer key and we drop the PID's because it will skew the data
x = pd.concat([train.drop(columns=["SalePrice"]), test]).drop(columns=["PID"])
# converts the data into 1.0 and 0.0 form
x = pd.get_dummies(x).astype("float32")
# print(x.sum().sort_values().head(15))
# fills the empty sections with the median to avoid getting zeros because that can mess up the math
x = x.fillna(x.median())

# this takes the values of x and converts everything into a smaller value for example say you have a house with 2 BR's and 2000 SqFt then [2, 2000] the network then will only look at sqft because its overpowering the BR's
x = (x - x.median()) / (x.std() + 1e-8) #1e-8 is added to prevent division by 0
x = x.clip(-5, 5)
# print(x.abs().max().sort_values(ascending=False).head(10))

# So the n varible is the list of pandas training numbers
n = len(train)
# the x_all takes the pandas table into a tensor
x_all = torch.tensor(x.values, dtype=torch.float32)
#this splits the data into two seperate tensors for the training values and the test values
x_train, x_test = x_all[:n], x_all[n:]

# the below is our training and validation split (80/20)

idx = torch.randperm(n)
training, validation = idx[:int(0.8 * n)], idx[int(0.8 * n):]

#the models hidden layer
#the train shape is take the F features of x_train and creates 64 different weighted summaries  then we make it nonlinear and then we blend 64 features into 1 which will be used to find our price
model = nn.Sequential(nn.Linear(x_train.shape[1], 64), nn.ReLU(), nn.Linear(64,1))
opt = torch.optim.AdamW(model.parameters(), lr = 0.003, weight_decay=1e-2)
loss_fn = nn.MSELoss()


# below is for all the actual training aswell as the plotting
# steps: predict, score predictions, work out what weights cause error, adjust them, test on held out houses but dont learn from it, record scores
train_losses, val_losses = [], []
for epoch in range(1000):
    loss = loss_fn(model(x_train[training]), y[training])
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        val_loss = loss_fn(model(x_train[validation]), y[validation])
    train_losses.append(loss.sqrt().item())
    val_losses.append(val_loss.sqrt().item())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# plot 1: loss curves
ax1.plot(train_losses, label="Training")
ax1.plot(val_losses, label="Validation")
ax1.set_xlabel("epoch")
ax1.set_ylabel("RMSE (Log Price)")
ax1.legend()

# plot 2: predicted vs actual
with torch.no_grad():
    pred = torch.expm1(model(x_train[validation])).squeeze()
    actual = torch.expm1(y[validation]).squeeze()

ax2.scatter(actual, pred, alpha=0.5)
ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--")
ax2.set_xlabel("Actual Price")
ax2.set_ylabel("Predicted Price")

plt.tight_layout()
plt.show()
