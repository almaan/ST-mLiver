import numpy as np
import pandas as pd
import anndata as ad

from scipy.spatial.distance import cdist
from functools import reduce

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as colormap

from statsmodels.regression.linear_model import OLS
from statsmodels.regression.linear_model import RegressionResultsWrapper as regres



from typing import *



def get_figure(n_elements : int,
               n_cols : Optional[int] = None,
               n_rows : Optional[int] = None,
               side_size : float = 3.0,
               sharey : bool = False,
               sharex : bool = False,
               )->Tuple[plt.Figure,plt.Axes]:


    """ Get figure and axes objects

    Parameters:
    ----------

    n_elements : int
        number of elements to plot
    n_cols : Optional[int] (None)
        desired number of columns
    n_rows : Optional[int] (None)
        desired number of rows
    side_size : float (3.0)
        size of each subplot (one per element)
    sharey : bool (False)
        set to True to share y-axis
    sharex : bool (False)
        set to True to share x-axis

    Returns:
    -------
    Tuple with Matplotlib Figure
    and Axes objects.

    """

    n_rows,n_cols = get_plot_dims(n_elements,
                                  n_cols,
                                  n_rows)
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

    """Get dimensions of plot

    will adjust row and column numbers
    according to the specified parameters. If
    n_cols and n_rows are both are None, then
    a square array of plots will be used.

    Parameters:
    ----------

    n_elements : int
        number of elements to plot
    n_cols : Optional[int] (None)
        desired number of columns
    n_rows : Optional[int] (None)
        desired number of rows

    Returns:
    --------
    A tuple with the form (n_rows,n_cols)

    """
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


def plot_veins(ax : plt.Axes,
               data : ad.AnnData,
               show_image : bool = False,
               show_spots : bool = False,
               show_id : bool = False,
               **kwargs,
               )->None:


    """Plot veins in 2D spatial space

    Parameters:
    ----------

    ax : plt.Axes
        Axes object to add plots to
    data : ad.AnnData
        AnnData object holding vein information in the
        uns['mask'] slot
    show_image : bool (False)
        set to True to show HE-image as background
    show_spots : bool (False)
        set to True to show spots
    show_id : bool (False)
        set to True to show vein id

    """

    if "alternative_colors" in kwargs:
        type_color = kwargs["alternative_colors"]
    else:
        types = np.unique(data.uns["mask"]["type"].values)
        types = np.sort(types)

        cti = {v:k for k,v in\
           enumerate(types)}

        type_color = data.uns["mask"]["type"].map(cti)
        type_color = type_color.values.flatten()


    if show_image:
        ax.imshow(data.uns["img"])
    if show_spots:
        ax.scatter(data.obs.x,
                   data.obs.y,
                   s = kwargs.get("spot_marker_size",80),
                   c = "none",
                   edgecolor = "black",
                   )

    ax.scatter(data.uns["mask"].x,
        data.uns["mask"].y,
        c = type_color,
        s = kwargs.get("node_marker_size",2),
        cmap = kwargs.get("cmap",plt.cm.Spectral_r),
        )

    if show_id:
        id_label = kwargs.get("id_label","id")
        assert "id" in data.uns["mask"].columns,\
            "must have vein ids"

        uni_veins = np.unique(data.uns["mask"][id_label].values)

        for vein in uni_veins:
            _pos = data.uns["mask"][id_label].values == vein
            _crds = data.uns["mask"][["x","y"]].values[_pos,:]
            _mns = _crds.min(axis=0).reshape(1,2)
            align = kwargs.get("id_align","top_right")

            if align == "center":
                _xy = crd.mean(axis = 0)
                _x = crd[0]
                _y = crd[1]
            else:
                _pxy = np.argmax(np.linalg.norm(_crds - _mns))
                _x = _crds[_pxy,0]
                _y = _crds[_pxy,1]

            ax.text(_x,
                    _y,
                    s = vein,
                    fontsize = kwargs.get("id_fontsize",10))


    ax.set_aspect("equal")
    ax.axis("off")




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

    """Generate feature by distance plots

    Function for seamless production of feature
    by distance plots.

    Parameters:
    ----------

    ax : plt.Axes
        axes object to plot data in
    data : Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]
        Tuple of data to plot, should be in form :
        (xs,ys,y_hat,std_err), the output which
        utils.smooth_fit produces.
    feature : Optional[str] (None)
        Name of plotted feature, set to None to exclude this
        information.
    include_background : bool (True)
        Set to True to include data points used
        to fit the smoothed data
    curve_label : Optional[str] (None)
        label of plotted data. Set to None to exclude
        legend.
    flavor : str = "normal",
        flavor of data, choose between 'normal' or
        'logodds'.
    color_scheme : Optional[Dict[str,str]] (None)
        dictionary providing the color scheme to be used.
        Include 'background':'color' to change color
        original data, include 'envelope':'color' to set
        color of envelope, include 'feature_class':'color' to
        set color of class.
    ratio_order : List[str] (["central","portal"])
        if logodds flavor is used, then specify which
        element was nominator (first element) and
        denominator (second element).
    list_flavor_choices : bool (False)
        set to True to list flavor choices
    feature_type : str ("Feature")
        Name of feature to plot, will be prepended to title
        as Feature : X. Set to None to only plot name X. Set
        to None to exclude feature type from being indicated
        in title and y-axis.
    distance_scale_factor : float (1.0)
        scaling factor to multiply distances with

    Returns:
    -------

    Tuple with Matplotlib Figure and Axes object, containing
    feature by distance plots.

    """

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

    ax.set_title("{} : {}".format(("" if feature_type is None else feature_type),
                                  ("" if feature is None else feature)),
                 fontsize = kwargs.get("title_font_size",
                                       kwargs.get("fontsize",15)),
                 )
    ax.set_ylabel("{} Value".format(("" if feature_type is None else feature_type)),
                  fontsize = kwargs.get("label_font_size",
                                        kwargs.get("fontsize",15)),
                  )

    if flavor == "normal":
        unit = ("" if "distance_unit" not in\
                kwargs.keys() else " [{}]".format(kwargs["distance_unit"]))

        ax.set_xlabel("Distance to vein{}".format(unit),
                      fontsize = kwargs.get("label_font_size",
                                            kwargs.get("fontsize",15)),
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

    """Visualize Prediction Results

    Generates a Bar Graph where the accuracy of each
    iteration in the cross validation.

    results : pd.DataFrame
        prediction results obtained from the
        structures.Model.cross_validation
        method
    bar_width : float (0.8)
        width of a single bar in the results display
    accuracy_colname : str ("accuracy")
        name of column where accuracy values are given
    target_colname : str ("pred_on")
        name of columns where target names are found

    Returns:
    --------
    Tuple of Matplotlib Figure and Axes object holding the
    bar graph illustrating the results.


    """

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




def expression_fields(xs : np.ndarray,
                    ys : np.ndarray,
                    results : regres,
                    n_ticks : int = 400,
                    )->Tuple[np.ndarray,np.ndarray,Tuple[int,int]]:

    mx = np.max((xs[:,1]))
    mn = np.min(xs[:,1])
    xx = np.linspace(mn,mx,n_ticks)
    mx = np.max((xs[:,2]))
    mn = np.min(xs[:,2])
    yy = np.linspace(mn,mx,n_ticks)
    X,Y = np.meshgrid(xx,yy)
    shape = X.shape
    Xf = X.flatten()
    Yf = Y.flatten()
    XY = np.hstack((np.ones((Xf.shape[0],1)),Xf[:,np.newaxis],Yf[:,np.newaxis]))
    Z = results.predict(XY)


    return (XY[:,1::],Z,shape)


def plot_field(ax : plt.Axes,
            Z : np.ndarray,
            XY : np.ndarray,
            cmap : colormap = plt.cm.magma,
            fontsize = 20,
            x_feature : str = "central",
            y_feature : str = "portal",
            )->None:

    im = ax.imshow(Z,cmap = cmap)
    ax.set_xlabel("Distance to central vein",fontsize = fontsize)
    ax.set_ylabel("Distance to portal vein", fontsize = fontsize)
    ax.set_xticks(np.linspace(0,Z.shape[0],10).round())
    ax.set_yticks(np.linspace(0,Z.shape[1],10).round())

    ax.set_xticklabels(np.linspace(XY[:,0].min(),XY[:,0].max(),10).round(),rotation = 90)
    ax.set_yticklabels(np.linspace(XY[:,1].min(),XY[:,1].max(),10).round())
    ax.invert_yaxis()

    plt.colorbar(im,ax=ax)


def bivariate_expression_plot(ax: plt.Axes,
                              data : [np.ndarray,np.ndarray],
                              feature: str,
                              feature_name: str = "Feature",
                              cmap: colormap = plt.cm.magma,
                              alpha: float = 0.05,
                              distance_scale_factor: float = 1,
                              **kwargs,
                              )->np.ndarray:


    xs = data[0]
    ys = data[1]

    model_full = OLS(ys,xs,hasconst = True)

    model_x1 = OLS(ys,xs[:,[0,1]])
    model_x2 = OLS(ys,xs[:,[0,2]])
    model_0 = OLS(ys,xs[:,0])

    results_full = model_full.fit()
    results_x1 = model_x1.fit()
    results_x2 = model_x2.fit()
    results_0 = model_0.fit()

    likelihood = np.array([results_full.llf,results_x1.llf,results_x2.llf,results_0.llf])

    insig = np.any(results_full.pvalues > alpha)

    XY,Z,reshape_shape = expression_fields(xs,ys,results_full)

    XY = XY * distance_scale_factor

    plot_field(ax,
               Z.reshape(reshape_shape),
               XY,
               fontsize=kwargs.get("label_fontsize",15),
               cmap = cmap)

    ax.set_title("{} : {}".format(feature_name,feature) + ("(*)" if insig else ""  ),
                 fontsize = kwargs.get("title_fontsize",25))

    return likelihood

