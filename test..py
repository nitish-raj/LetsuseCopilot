# Train a basic model using pytorch

#%%
# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import kaggle


# Download data from Kaggle API and load it into a pandas dataframe
# kaggle datasets download -d odedgolden/movielens-1m-dataset
# unzip movielens-1m-dataset.zip
# rm movielens-1m-dataset.zip

#%%

# Read data from open source dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1', names=['movie_id', 'title', 'genre'])
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1', names=['UserID','MovieID','Rating','Timestamp'])
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1', names=['UserID','MovieID','Rating','Timestamp'])
#%%
movies.head()
#%%

# Use sklearn encoding to convert movies to numerical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(movies['genre'])

# Map movie genres to numbers
movies['genre'] = le.transform(movies['genre'])



#%%

# Visualize the data
# print(movies)
# print(users)
# print(ratings)

# Plot scatter plot of ratings
import matplotlib.pyplot as plt
plt.scatter(ratings[:, 1], ratings[:, 2])
plt.xlabel('User ID')
plt.ylabel('Rating')
plt.show()

#%%
# Split the data into training and test set usning sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(movies, ratings, test_size=0.25, random_state=42)

#%%
X_train

#%%
# Convert the data into torch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# Create the architecture of the neural network
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(1682, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, 1682)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

# Create the model
model = SAE()
# Create the optimizer
optimizer = optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.5)
# Create the loss function
criterion = nn.MSELoss()

# Train the model
n_epochs = 1000
for epoch in range(1, n_epochs + 1):
    train_loss = 0
    for data in zip(X_train, y_train):
        input, target = data
        input, target = Variable(input), Variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    print('Epoch: {} | Loss: {}'.format(epoch, train_loss / len(X_train)))

# Test the model
test_loss = 0
for data in zip(X_test, y_test):
    input, target = data
    input, target = Variable(input), Variable(target)
    output = model(input)
    loss = criterion(output, target)
    test_loss += loss.data[0]
print('Test loss: {}'.format(test_loss / len(X_test)))

# Save the model
torch.save(model.state_dict(), 'sae.pkl')










