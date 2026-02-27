# %%
# load libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# %%
# load data
help(pd.read_csv) #list help and get the function you want to learn more about it
df = pd.read_csv("house_votes_Dem.csv", encoding='latin-1')
#gives error 'utf-8' there's characters that fall outside the documentation


# %%
# take a look at the data
df.info() # gives you the number of rows and columns + data types of each column


# %%
# separate out the numeric features
#want to separate into numeric features 

c_num = df[["aye", "nay", "other"]]

# don't need to standardize the data because all data are on the same scale --same metrics 

# %%
# documentation for kmeans in sklearn

# default initiation process kmeans++, defualts to 1  and will determine the best distance it is from the data points to the centroids, and then it will select the next centroid based on the distance from the first centroid to optimize convergence ;
#kmeans ++ ensures the dots are not placed in a way that is not optimal for convergence, it is a better way to initialize the centroids than random initialization
help(KMeans)
# %% build a kmeans model
kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans.fit(c_num)
# verbose means it will print out the progress of the algorithm as it runs, which can be helpful for understanding how the algorithm is working and for debugging purposes.

#we need to find the parameters inside class KMeans,
# %% look at the information in the model
print(kmeans.cluster_centers_)  # coordinates of cluster centers
print(kmeans.labels_)  # cluster labels for each data point

# %%
# add the cluster labels to the original data frame
df['cluster'] = kmeans.labels_
# %%
  
# %% simple plot of the clusters
help(plt.scatter)
 
# %%
# use a for lopp to check different clusters
# numbers and see how intertia changes 

intertias = []
k_values = range(1, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(c_num)
    intertias.append(kmeans.inertia_)       

# %%
#plot the inertia values to see if there is an elbow in the plot 

plt.figure(figsize=(10,5))
plt.plot(k_values, intertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
# %%
