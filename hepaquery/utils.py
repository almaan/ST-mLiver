#!/usr/bin/env python3


import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression as LR
from functools import reduce
from skmisc.loess import loess

import matplotlib.pyplot as plt


from typing import *

def iprint(s : str,
           )->None:

    print("[INFO] : {}".format(s))

def eprint(s : str,
           )->None:

    print("[ERROR] : {}".format(s))




def smooth_fit(xs : np.ndarray,
               ys : np.ndarray,
               dist_thrs : float = 0,
               )->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    srt = np.argsort(xs)
    xs = xs[srt]
    ys = ys[srt]

    keep = np.abs(xs) < dist_thrs
    xs = xs[keep]
    ys = ys[keep]

    # generate loess class object
    ls = loess(xs,
               ys,
              )
    # fit loess class to data
    ls.fit()

    # predict on data
    pred =  ls.predict(xs,
                       stderror=True)
    # get predicted values
    ys_hat = pred.values
    # get standard error
    stderr = pred.stderr

    return (xs,ys,ys_hat,stderr)


