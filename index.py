import numpy as np
import pandas as pd
import sklearn

a = np.array([1,2,3])
f = sklearn.LinearRegression(a)

def DoLR(a, f):
    y = f(a)
    return y
    