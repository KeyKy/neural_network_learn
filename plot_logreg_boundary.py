# example of plotting logistic regression decision boundary
# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
# http://matplotlib.org/api/pyplot_api.html for plt.scatter
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn import linear_model

np.random.seed(0)
# Make two interleaving half circles
X, y = sklearn.datasets.make_moons(200, noise=0.20)

logreg = linear_model.LogisticRegressionCV()
logreg.fit(X, y)

h = 0.02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy= np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()