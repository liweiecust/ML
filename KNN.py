from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# sample data
iris=load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris['data'],iris['target'],random_state=0)

# build the model
knn=KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=1,p=2,weights='uniform')
knn.fit(x_train,y_train)

# make predictions
x_new=np.array([[5,2.9,1,0.2]]) # conver the 1 D array to 2 D array
prediction=knn.predict(x_new)
print('prediction of the sample is',iris['target'][prediction])

# evaluate the model
y_pred=knn.predict(x_test)
#np.mean(y_pred,y_test)
accuracy=knn.score(x_test,y_test)
print('model accuracy is ', accuracy)

