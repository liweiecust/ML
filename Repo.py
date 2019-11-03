import matplotlib.pyplot as plt
import pandas as pd
# pip install pandas
from sklearn.datasets import load_iris
# pip install scikit-learn

from sklearn.model_selection import train_test_split

iris=load_iris()
iris.keys()
#dict_keys(['DESCR','data','target_names','feature_names','target'])
print(type(iris['DESCR']))
fig,ax=plt.subplots(3,3,figsize=(15,15))
plt.suptitle("iris_pairplot")

x_train, x_test, y_train, y_test=train_test_split(iris['data'],iris['target'],random_state=0)
for i in range(3):
    for j in range(3):
        print(x_train[:,j])
        print(x_train[:,i+1])
        print("\t")

for i in range(3):
    for j in range(3):
        ax[i,j].scatter(x_train[:,j],x_train[:,i+1],c=y_train,s=60)
        ax[i,j].set_xticks(())
        ax[i,j].set_yticks(())
        if i==2:
            ax[i,j].set_xlabel(iris['feature_names'][j])
        if j==0:
            ax[i,j].set_ylabel(iris['feature_names'][i+1])
        if j>1:
            ax[i,j].set_visible(False)
plt.show()

