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
<<<<<<< HEAD
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1', names=['MovieID', 'title', 'genre'])
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1', names=['UserID','MovieID','Rating','Timestamp'])
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1', names=['UserID','MovieID','Rating','Timestamp'])
#%%
ratings.head()
=======
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



>>>>>>> b86971d70a6d8c5ec578d4908a9bf66004daa988
#%%

# Use sklearn encoding to convert movies to numerical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(movies['genre'])

# Map movie genres to numbers
movies['genre'] = le.transform(movies['genre'])

<<<<<<< HEAD
# Map movie titles to numbers
le.fit(movies['title'])
movies['title'] = le.transform(movies['title'])

#Join movies and ratings dataframes
data = pd.merge(movies, ratings)
=======
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
>>>>>>> b86971d70a6d8c5ec578d4908a9bf66004daa988

# Drop the timestamp column
data = data.drop(['Timestamp'], axis=1)
data.head()


# Visualize the data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Create a histogram of the ratings
plt.hist(data['Rating'], bins=20)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Create X and y matrices
X = data.iloc[:, :4]
y = data.iloc[:, 4]

# Split the data into training and test set usning sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

# Convert the data into torch tensors
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
y_train = torch.FloatTensor(y_train.values)
y_test = torch.FloatTensor(y_test.values)

# Create model to predict the rating of a movie
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_size))
        # Forward propagate input
        out, hn = self.rnn(x, h0)
        
        # Propagate hidden state through time
        out = self.sigmoid(self.output_layer(out[:, -1, :]))
        return out


# Initialize the model
model = RNN(input_size=X_train.size(1), hidden_size=10, output_size=1)

# Define the loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(1000):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print ('Epoch [%d/%d], Loss: %.4f' % (epoch+1, 1000, loss.data[0]))

# Test the model
# Initialize hidden state
h0 = Variable(torch.zeros(1, 1, 10))
# Predict the rating of a movie
outputs = model(X_test)
# Calculate the loss
loss = criterion(outputs, y_test)
print ('Test Loss: %.4f' % loss.data[0])

