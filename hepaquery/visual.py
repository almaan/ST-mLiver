import numpy as np
import pandas as pd
import anndata as ad

from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression as LR
from functools import reduce

import matplotlib.pyplot as plt

from typing import *


def get_figure(n_elements : int,
               n_cols : Optional[int] = None,
               n_rows : Optional[int] = None,
               side_size : float = 3,
               sharey : bool = False,
               sharex : bool = False,
               )->Tuple[plt.Figure,plt.Axes]:

    n_rows,n_cols = get_plot_dims(n_elements,n_cols,n_rows)
    figsize = (n_cols * side_size,
               n_rows * side_size,
               )

    fig,ax = plt.subplots(n_rows,
                          n_cols,
                          figsize = figsize,
                          sharex = sharex,
                          sharey = sharey,
                          )

    ax = ax.flatten()

    for aa in ax[n_elements::]:
        aa.set_visible(False)

    return (fig,ax)


def get_plot_dims(n_elements : int,
                  n_cols : Optional[int] = None,
                  n_rows : Optional[int] = None,
                  )->Tuple[int,int]:

    round_fun = lambda x: int(np.ceil(n_elements/x))
    if n_cols is not None:
        n_rows = round_fun(n_cols)
    elif n_rows is not None:
        n_cols = round_fun(n_rows)
    else:
        n_rows  = np.sqrt(n_elements)
        if round(n_rows) == n_rows:
            n_cols = n_rows
        else:
            n_cols = n_rows +1

    return (n_rows,n_cols)


def plot_expression_by_distance(ax : plt.Axes,
                                data : Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray],
                                feature : Optional[str] = None,
                                include_background : bool = True,
                                curve_label : Optional[str]  = None,
                                flavor : str = "normal",
                                color_scheme : Optional[Dict[str,str]] = None,
                                ratio_order : List[str] = ["central","portal"],
                                list_flavor_choices : bool = False,
                                feature_type : str = "Feature",
                                distance_scale_factor : float = 1.0,
                                **kwargs,
                                )->None:

    flavors = ["normal","logodds","single_vein"]
    if list_flavor_choices:
        print("Flavors to choose from are : {}".format(', '.join(flavors)))
        return None
    if flavor not in flavors:
        raise ValueError("Not a valid flavor")

    if len(data) != 4:
        raise ValueError("Data must be (xs,ys,ys_hat,stderr)")

    if color_scheme is None:
        color_scheme = {}

    scaled_distances = data[0] * (distance_scale_factor if \
                                  flavor != "logodds" else 1.0)

    if include_background:
        ax.scatter(scaled_distances,
                   data[1],
                   s = 1,
                   c = color_scheme.get("background","gray"),
                   alpha = 0.4,
        )


    ax.fill_between(scaled_distances,
                    data[2] - data[3],
                    data[2] + data[3],
                    alpha = 0.2,
                    color = color_scheme.get("envelope","grey"),
                    )

    ax.set_title("{} : {}".format(feature_type,("" if feature is None else feature)),
                 fontsize = kwargs.get("title_font_size",kwargs.get("fontsize",15)),
                 )
    ax.set_ylabel("{} Value".format(feature_type),
                  fontsize = kwargs.get("label_font_size",kwargs.get("fontsize",15)),
                  )

    if flavor == "normal":
        unit = ("" if "distance_unit" not in\
                kwargs.keys() else " [{}]".format(kwargs["distance_unit"]))

        ax.set_xlabel("Distance to vein{}".format(unit),
                      fontsize = kwargs.get("label_font_size",kwargs.get("fontsize",15)),
                      )

    if flavor == "logodds":

        x_min,x_max = ax.get_xlim()

        ax.axvspan(xmin = x_min,
                   xmax = 0,
                   color = color_scheme.get(ratio_order[0],"red"),
                   alpha = 0.2,
                   )

        ax.axvspan(xmin = 0,
                   xmax = x_max,
                   color = color_scheme.get(ratio_order[1],"blue"),
                   alpha = 0.2,
                   )

        d1 = ratio_order[0][0]
        d2 = ratio_order[1][0]
        ax.set_xlabel(r"$\log(d_{}) - \log(d_{})$".format(d1,d2),
                      fontsize = kwargs.get("label_font_size",kwargs.get("fontsize",15)),
                      )

    ax.plot(scaled_distances,
            data[2],
            c = color_scheme.get("fitted","black"),
            linewidth = 2,
            label = ("none" if curve_label is None else curve_label),
            )

    if "tick_fontsize" in kwargs.keys():
        ax.tick_params(axis = "both",
                       which = "major",
                       labelsize = kwargs["tick_fontsize"])





def visualize_prediction_result(results : pd.DataFrame,
                                bar_width : float = 0.8,
                                accuracy_colname : str = "accuracy",
                                target_colname : str = "pred_on",
                                )->Tuple[plt.Figure,plt.Axes]:

    fig,ax = plt.subplots(1,1, figsize = (7,4))

    p1 = ax.bar(np.arange(results.shape[0]),
                results[accuracy_colname].values,
                bar_width,
                facecolor = "#8F88EC",
                edgecolor = "black",
                label = "correct",
                        )
    p2 = ax.bar(np.arange(results.shape[0]),
                1.0 - results[accuracy_colname].values,
                bar_width,
                bottom=results[accuracy_colname].values,
                facecolor = "lightgray",
                label = "incorrect",
            )

    p2 = ax.bar(np.arange(results.shape[0]),
                np.ones(results.shape[0]),
                bar_width,
                facecolor = "none",
                edgecolor = "black",
                linestyle = "dashed",
            )


    ax.axhline(y = 0.5,
               linestyle = "dashed",
               color = "red",
               label = r"$50\%$",
               )

    ax.set_xticks(np.arange(results.shape[0]))
    ax.set_xticklabels(results[target_colname],
                    rotation = 90,
                    fontsize = 10,
                    )

    ax.set_xlabel("Predicted on",fontsize =15)
    ax.set_ylabel("Accuracy",fontsize = 15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend()

    return (fig,ax)
