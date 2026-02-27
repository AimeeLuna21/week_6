# %%
import pandas as pd 
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

# %%
# load data
salary_data = pd.read_csv("2025_salaries.csv", header=1, encoding ='latin-1') # header=1 means that the first row of the data will be treated as the column names
# the issue here is that the column names are unamed because it's counting the columns names as the first ROW
# Meaning the first row of the data is being treated as the column names, which is why the columns are labeled as "Unnamed: 0", "Unnamed: 1"...
salary_data.head()

# %%
stats = pd.read_csv("nba_2025.txt", sep=",", encoding='latin-1') # sep=',' means that the data is separated by commas, which is the default for csv files, but since this is a txt file we need to specify it; encoding='latin-1' is used to specify the encoding of the file, which is necessary because there are characters in the file that fall outside of the default 'utf-8' encoding, which is why we need to use 'latin-1' to properly read the file without errors.
stats.head()


# %%
salary_data.info()
salary_data.head()
# %%
# inner join --our players...like the spine of the books 
# merge the two dataframes on the "Player" columns
help(pd.merge)
# %%
merged_data = pd.merge(salary_data, stats, on="Player") 
#our key is PLAYER
# syntax --makes everything lowercase so the merge is easy
# %%
# duplicates in the Player column

duplicates = merged_data[merged_data.duplicated(subset="Player", keep=False)]
# when we see one pair of brakcets it means we're not necesseraily indexing but rather we are trying to subset the existing dataset and pass it back onto a new variable
# if two brackets we'd be slicing the data frame and getting a specific row or column
print(duplicates)
# checked in data wrangler and there's only 56 unique entries the rest are duplicates
# %%
#Sklearn four steps
#1 create an instance of the model: exmaple: mymodel = KMeans(n_clusters=3, random_state=42)
#2 fit the model to the data: example: mymodel.fit(X)
#3 make predictions using the model: example: predictions = mymodel.predict(X)
#4 evaluate the models performance: example: score = mymodel.score(X)

# for kmeans you don't need to predict, you can just use the labels_ attribute to get the cluster labels for each data point, and you can use the inertia_ attribute to evaluate the performance of the model.
# to get the cluster assignments for each data point after fitting the model

# %%
#lambda functions are anonymous functions that can be defined in a single line of code
# they are often used for simple operations that can be defined in a single line, 
# for example, you can use a lamda function to create a new columns in a dataframe that is the result of a simple operation on existing columns, such as adding two columns together or applying a mathematical function to a column.
#ex. you want to create a new columns "Salary_in_thousands" that is the result of 
# that is the salary divided by 1000, you can use a lambda function like this:
merged_data["Salary_in_thousands"] = merged_data["Salary"].apply(lambda x: x / 1000)