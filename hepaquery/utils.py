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
    """ info print

    Parameters:
    ----------

    s : str
        string to print with info wrapper

    """

    print("[INFO] : {}".format(s))

def eprint(s : str,
           )->None:

    """ error print

    Parameters:
    ----------

    s : str
        string to print with error wrapper

    """


    print("[ERROR] : {}".format(s))




def smooth_fit(xs : np.ndarray,
               ys : np.ndarray,
               dist_thrs : Optional[float] = None,
               )->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    """Smooth curve using loess

    will perform curve fitting using skmisc.loess,
    points above 'dist_thrs' will be excluded.

    Parameters:
    ----------

    xs : np.ndarray
        x values
    ys : np.ndarray
        y values
    dist_thrs : float
        exclude (x,y) tuples where x > dist_thrs

    Returns:
    -------
    A tuple with included x and y-values (xs',ys'), as well
    as fitted y-values (ys_hat) together with associated
    standard errors. The tuple is on the form
    (xs',ys',y_hat,std_err)

    """

    srt = np.argsort(xs)
    xs = xs[srt]
    ys = ys[srt]

    if dist_thrs is None:
        dist_thrs = np.inf

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


