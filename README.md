# House-Price-prediction-model
A simple to understand house price prediction model, using kaggle data sets
Considering this is my first time dealing with this type of project I will be writing down things I didn't know and had to learn

1st problem: when writing my first tensor value (y) I realized that it would create a 1D array because we were converting just the saleprices into an array but then I realized that Pytorch structures the data in 2D (not because its limited to 2D and cant use 1D but because it will rarely be 1D due to it comparing batches and features) so after some googling I figured out that I can use .unsqueeze(1) to make the newly created tensor a 2D array

2nd problem: when I was making my x.fillin(x.median()) I originally had it as pd.fillin(pd.median()) because I did not realize that once you used .concat() make x a pandas dataset so we can use x.fillin because its recognzied as pd (or at least thats how I made sense of it)

