# House-Price-prediction-model
A simple to understand house price prediction model, using kaggle data sets
Considering this is my first time dealing with this type of project I will be writing down things I didn't know and had to learn

1st problem: when writing my first tensor value (y) I realized that it would create a 1D array because we were converting just the saleprices into an array but then I realized that Pytorch structures the data in 2D (not because its limited to 2D and cant use 1D but because it will rarely be 1D due to it comparing batches and features) so after some googling I figured out that I can use .unsqueeze(1) to make the newly created tensor a 2D array

2nd problem: when I was making my x.fillin(x.median()) I originally had it as pd.fillin(pd.median()) because I did not realize that once you used .concat() make x a pandas dataset so we can use x.fillin because its recognzied as pd (or at least thats how I made sense of it)

3rd problem: when I was doing the Model creation using torch nn I was unaware how it worked so I did have to look into that. From what I understand what Im doing is taking the features of our training set and breaking it down from F-->64 so the model will find things that dont change value in the feature set and then account for it. Then I went ahead and used nn.ReLU which will take all numbers and make clip any that are negative to make sure we get a non linear model. Then we take the 64 convert to 1 value which will be used to find the price in dollars.

4th problem: I forgot initally I needed a way to graph the output so I added the matplotlib.


