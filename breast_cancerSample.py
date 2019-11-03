import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# pip install mglearn
import mglearn

x,y=mglearn.datasets.make_forge()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
fig,axes=plt.subplots(1,3,figsize=(10,3))

for n_neighbors,ax in zip([1,3,9], axes):
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(x,y)
    mglearn.plots.plot_2d_separator(clf,x,fill=True,eps=0.5,ax=ax,alpha=.4)
    ax.scatter(x[:,0],x[:,1],c=y,s=60,cmap=mglearn.cm2)
    ax.set_title("%d neighbors(s)" % n_neighbors)
plt.show()
