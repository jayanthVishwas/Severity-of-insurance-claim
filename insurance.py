import warnings
warnings.filterwarnings('ignore')

import pandas as pd
dataset=pd.read_csv("C:/python/dataset/train.csv")

dataset_test=pd.read_csv("C:/python/dataset/test.csv")

ID = dataset_test['id']

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

print(dataset.shape)

dataset=dataset.iloc[:,1:]

#print(dataset.describe())


#print(dataset.skew())

import numpy

import seaborn as sns
import matplotlib.pyplot as plt

#range of features
split=116

#number of features considered
size=15

#dataframe with continuous features
data=dataset.iloc[:,split:]

cols=data.columns

#violine plots
n_cols = 2
n_rows = 7


# for i in range(n_rows):
    # fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12, 8))
    # for j in range(n_cols):
        # sns.violinplot(y=cols[i*n_cols+j], data=dataset, ax=ax[j])
        


dataset["loss"] = numpy.log1p(dataset["loss"])
#visualize the transformed column
# sns.violinplot(data=dataset,y="loss")  
# plt.show()

data_corr = data.corr()

threshold=0.5
corr_list=[]

for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j])


s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
# for v,i,j in s_corr_list:
    # print ("%s and %s = %.2f" % (cols[i],cols[j],v))


# for v,i,j in s_corr_list:
    # sns.pairplot(dataset, size=6, x_vars=cols[i],y_vars=cols[j] )
    # plt.show()
    
cols = dataset.columns

#Plot count plot for all attributes in a 29x4 grid
n_cols = 4
n_rows = 29
# for i in range(n_rows):
    # fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))
    # for j in range(n_cols):
        # sns.countplot(x=cols[i*n_cols+j], data=dataset, ax=ax[j])
        # plt.show()
        

import pandas

#cat1 to cat116 have strings. The ML algorithms we are going to study require numberical data
#One-hot encoding converts an attribute to a binary vector

#Variable to hold the list of variables for an attribute in the train and test data
labels = []

for i in range(0,split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))    
    

del dataset_test

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset.iloc[:,i])
    feature = feature.reshape(dataset.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)
 
    
encoded_cats = numpy.column_stack(cats)

# # Print the shape of the encoded data
print(encoded_cats.shape)

# #Concatenate encoded attributes with continuous attributes
dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)

del cats
del feature
del dataset
del encoded_cats
print(dataset_encoded.shape)

r, c = dataset_encoded.shape

#create an array which has indexes of columns
i_cols = []
for i in range(0,c-1):
    i_cols.append(i)

#Y is the target column, X has the rest
X = dataset_encoded[:,0:(c-1)]
Y = dataset_encoded[:,(c-1)]
del dataset_encoded

#Validation chunk size
val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

#Split the data into chunks
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, random_state=seed)
del X
del Y

#All features
X_all = []

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae = []

#Scoring parameter
from sklearn.metrics import mean_absolute_error

#Add this version of X to the list 
n = "All"
#X_all.append([n, X_train,X_val,i_cols])
X_all.append([n, i_cols])


##Linear regression

# from sklearn.linear_model import LinearRegression

# model=LinearRegression(n_jobs=-1)

# for name,i_cols_list in X_all:
   # model.fit(X_train[:,i_cols_list],Y_train)
   # predictions = numpy.expm1(model.predict(X_val[:,i_cols_list]))
   # result = mean_absolute_error(numpy.expm1(Y_val), predictions)
   # mae.append(result)
   # print(name + " %s" % result)

##Ridge regression

# from sklearn.linear_model import Ridge

# alpha_list=numpy.array([0.1,1,10.0,100])

# for alpha in alpha_list:
    # model= Ridge(alpha=alpha, random_state=seed)
    
    # for name,i_cols_list in X_all:
       # model.fit(X_train[:,i_cols_list],Y_train)
       # predictions = numpy.expm1(model.predict(X_val[:,i_cols_list]))
       # result = mean_absolute_error(numpy.expm1(Y_val), predictions)
       # mae.append(result)
       # print(name + " %s" % result) 
       # print("best alpha")
       # print(model.alpha)
        

##Lasso regression

from sklearn.linear_model import Lasso

alpha_list=numpy.array([0.1,1,10.0,100])

for alpha in alpha_list:
    #Set the base model
    model = Lasso(alpha=alpha,random_state=seed)
    
    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        predictions = numpy.expm1(model.predict(X_val[:,i_cols_list]))
        result = mean_absolute_error(numpy.expm1(Y_val), predictions)
        mae.append(result)
        print(name + " %s" % result)
  

with open("C:\python\dataset\submission_ridge.csv", "w") as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(predictions)):
        subfile.write("%s,%s\n"%(ID[i],pred))