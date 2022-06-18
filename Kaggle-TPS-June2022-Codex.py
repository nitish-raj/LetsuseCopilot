
import os
import pandas as pd
import numpy as np
import kaggle

# 1. Download using Kaggle api for competition "tabular-playground-series-jun-2022"
kaggle.api.authenticate()
kaggle.api.competition_download_files("tabular-playground-series-jun-2022", path=os.getcwd())

# 2. Unzip data
import zipfile
with zipfile.ZipFile("tabular-playground-series-jun-2022.zip", 'r') as zip_ref:
    zip_ref.extractall()

# 3. Delete zip file
os.remove("tabular-playground-series-jun-2022.zip")

# 4. Read data into dataframe
df = pd.read_csv("data.csv")

# 5. Find location of each missing element
df.isnull().stack()

# 6. Create a new dataframe with row+column location of missing value as index
df_missing = df.isnull().stack().reset_index()
df_missing.columns = ['row', 'column', 'missing']
df_missing = df_missing[df_missing['missing'] == True]
df_missing = df_missing.drop(columns=['missing'])
df_missing.head()

# 7. Concat row and column of df_missing with '-' as seperator and drop all other column. Also sort the dataframe using new column
df_missing['row_column'] = df_missing['row'].astype(str) + '-' + df_missing['column']
df_missing = df_missing.drop(columns=['row', 'column'])
df_missing = df_missing.sort_values(by=['row_column'])
df_missing.head()

# 8. Create seperate model with each column with missing values as target variable 
#    and predict missing values using the model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 9. Create a list of column names
columns = df.columns.tolist()
columns

# 10. Create a list of column names with missing values
columns_with_missing = df.columns[df.isnull().any()].tolist()
columns_with_missing

# 11. Create a list of column names without missing values
columns_without_missing = [c for c in columns if c not in columns_with_missing]
columns_without_missing

# 12. Write a function for feature enginering
def feature_engineering(df):
    # Add mean , mode and median of each column
    df['mean'] = df.mean(axis=0)
    df['mode'] = df.mode(axis=0)
    df['median'] = df.median(axis=0)
    return df
