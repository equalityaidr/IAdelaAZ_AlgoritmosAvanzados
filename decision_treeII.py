# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:00:42 2020

@author: DELL
"""
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

X_moons, y_moons = make_moons(n_samples=100, noise=0.25, random_state=53)

#modelo con 10 capas
deep_tree_clf1 = DecisionTreeClassifier(max_depth=10, random_state=42)
deep_tree_clf1.fit(X_moons, y_moons)

#Decsion boundaries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    plt.figure(figsize=(8, 4))
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

#plot (overfitting), outliers
plt.figure(figsize=(11, 4))
plot_decision_boundary(deep_tree_clf1, X_moons, y_moons, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("Maximum depth 10", fontsize=16)
plt.show()

#Regularización: Limitando capacidades del modelo para que se comporte mejor
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf2.fit(X_moons, y_moons)

#plot2, efecto de min_samples_leaf=4
plot_decision_boundary(deep_tree_clf2, X_moons, y_moons, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)

plt.show()

deep_tree_clf2.feature_importances_