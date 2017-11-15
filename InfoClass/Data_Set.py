#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:09:00 2017

@author: bbaker
"""
import os
import numpy as np


class Data_Set():

    def __init__(self, X=None, y=None, datafile=None, labelfile=None,
                 random_state=42):
        if not X and not datafile:
            raise Exception("You have either enter data or a file")
        if not y and not labelfile:
            raise Exception("You have to either enter labels or a file")
        self.__build(X, y, datafile, labelfile)

    def __build(self, X, y, datafile, labelfile):
        if X:
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if not y:
                y = X[:, 0]
                X = X[:, 1:]
        if y:
            if not isinstance(y, np.ndarray)
