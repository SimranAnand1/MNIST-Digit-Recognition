import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

a = pd.read_csv("C:/Users/simmu/Downloads/mnist_train.csv/mnist_train.csv")
#print(np.shape(a))

l  = a["label"]

d = a.drop("label", axis = 1)

labels = l.head(15000)

data = d.head(15000)

from sklearn.preprocessing import StandardScaler

Standardized_data = StandardScaler().fit_transform(data)

#print(Standardized_data)

#print(Standardized_data.shape)

covar_matrix = np.matmul(Standardized_data.T,Standardized_data)

#print(covar_matrix.shape)

from scipy.linalg import eigh

values, vectors = eigh(covar_matrix, eigvals=(782,783))

vectors = vectors.T

#print(vectors.shape)

new_coordinates = np.matmul(vectors,Standardized_data.T)

#print(new_coordinates.shape)

new_coordinates = np.vstack((new_coordinates, labels)).T


#dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "labels"))

#print(dataframe.head())

#g = sn.FacetGrid(dataframe, hue="labels",height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

#plt.show(g)

from sklearn import decomposition

pca = decomposition.PCA()

pca.n_components = 2

pca_data = pca.fit_transform(Standardized_data)

pca_data = np.vstack((pca_data.T, labels)).T

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "labels"))

sn.FacetGrid(pca_df, hue="labels",height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()
