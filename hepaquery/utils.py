#!/usr/bin/env python3


import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from functools import reduce
from skmisc.loess import loess
from scipy.stats import chi2


import matplotlib.pyplot as plt

from os import listdir
import os.path as osp


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


def load_genelist(dirname: str,
                  filter_tag: Optional[str],
                  include_all: bool = True,
                  )-> Dict[str,List[str]]:

    pths = listdir(dirname)

    if filter_tag is not None:
        pths = list(filter(lambda x: filter_tag in x,pths))

    genes = dict()

    for filename in pths:
        name = '.'.join(filename.split(".")[0:-1])
        with open(osp.join(dirname,filename),"r+") as f:
            _gs = f.readlines()
            _gs = [x.replace("\n","") for x in _gs]
            genes.update({name:_gs})

    if include_all:
        genes["all"] = reduce(lambda x,y : x + y,list(genes.values()))

    return genes

def likelihood_ratio_test(likelihoods: Dict[str,float],
                          dofs: Union[float,int,np.ndarray],
                          included_covariates: List[List[str]],
                          alpha: float = 0.05,
                          )->pd.DataFrame:

    n_features = len(likelihoods)
    features = list(likelihoods.keys())
    element_0 = list(likelihoods.values())[0]
    n_tests = len(element_0) - 1
	

    included_covariates = [[ic] if isinstance(ic,str) else ic\
                           for ic in included_covariates ]

    colnames = ["covariates_" + "_".join(ic) for ic in included_covariates]
    model_eval = pd.DataFrame(np.zeros((n_features,n_tests)),
                              columns = colnames,
                              index = features,
                              )

    if isinstance(dofs,float) or\
       isinstance(dofs,int):
        dofs = np.array([dofs] * n_tests)

    for gene,vals in likelihoods.items():
        pvals = np.zeros(n_tests)
        for k,(dof,v) in enumerate(zip(dofs,vals[1::])):
            rv = chi2(dof)
            D = -2 * (v - vals[0])
            pvals[k] = rv.sf(D)

        model_eval.loc[gene,:] = pvals

    for k,ic in enumerate(included_covariates):
        colname = "[full_model]_superior_to_[" + "_".join(ic) + "_model]"
        model_eval[colname] = model_eval.iloc[:,k].values < alpha

    return model_eval




