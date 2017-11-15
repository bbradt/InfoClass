#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
SKLearn Classifier suite
=====================

Work off of arbitrary input saved as a numpy file
with X, y = data, labels

"""

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class PolynomialRegression():
    """"Wrapper Class for Polynomial Logistic Regression in SKLearn"""
    def __input__(self, **kwargs):
        deg = 2
        if 'degree' in kwargs:
            deg = kwargs.pop('degree', None)
        return(Pipeline([('poly', PolynomialFeatures(degree=deg)),
                         ('logistic',
                         LogisticRegression(**kwargs))]))


class Class_Suite():

    MODELS = {"Ridge Regression":        RidgeClassifier,
              "Logistic Regression":     LogisticRegression,
              "SGD":                     SGDClassifier,
              "Perceptron":              Perceptron,
              "Passive Aggressive":      PassiveAggressiveClassifier,
              "Polynomial Regression":   PolynomialRegression,
              "K Nearest Neighbors":     KNeighborsClassifier,
              "Radius Neighbors":        RadiusNeighborsClassifier,
              "Nearest Centroid":        NearestCentroid,
              "SVM":                     SVC,
              "NuSVM":                   NuSVC,
              "Gaussian Process":        GaussianProcessClassifier,
              "Decision Tree":           DecisionTreeClassifier,
              "Random Forest":           RandomForestClassifier,
              "AdaBoost":                AdaBoostClassifier,
              "Bagging":                 BaggingClassifier,
              "Gradient Tree Boosting":  GradientBoostingClassifier,
              "Gaussian Naive Bayes":    GaussianNB,
              "Multinomial Naive Bayes": MultinomialNB,
              "Bernoulli Naive Bayes":   BernoulliNB,
              "LDA":                     LinearDiscriminantAnalysis,
              "QDA":                     QuadraticDiscriminantAnalysis,
              "Neural Net":              MLPClassifier
              }

    ACCEPTABLE_NAMES = {name: set([name, name.lower(), name.upper(),
                                   name.strip(), name.strip().lower(),
                                   name.strip().upper(),
                                   ''.join([w[0] if len(name.split()) > 1
                                            else w[0:3].upper()
                                            for w in name.split()]),
                                   ''.join([w[0] if len(name.split()) > 1
                                            else w[0:3]
                                            for w in name.split()]).lower()])
                        for name in MODELS}

    def __init__(self, names, hyperparameters={}):
        real_names = {}
        for in_name in names:
            name = self._get_model_key(in_name)
            if name:
                real_names[name] = in_name
        self.names = set(real_names.keys())
        self.classifiers = []
        self.hyperparameters = {}
        for name in self.names:
            in_name = real_names[name]
            kwargs = {}
            if in_name in hyperparameters:
                kwargs = hyperparameters[in_name]
            self.classifiers.extend(self.MODELS(**kwargs))
            self.hyperparameters[name] = kwargs

    def _get_model_key(self, name):
        for model_name in self.ACCEPTABLE_NAMES:
            if name in self.ACCEPTABLE_NAMES[model_name]:
                return(model_name)
        return None

    def evaluate_model(self, in_name, dataset, max_iter=None, random_state=42):
        name = self._get_model_key(in_name)
        if not name:
            return
        clf = self.classifiers(name)
        X_train = dataset.x_train
        y_train = dataset.y_train
        X_test = dataset.x_test
        y_test = dataset.y_test
        clf.random_state = random_state
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        return score, y_hat
