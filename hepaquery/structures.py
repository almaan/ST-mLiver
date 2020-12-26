import numpy as np
import pandas as pd
import anndata as ad

from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression as LR
from functools import reduce

from utils import iprint,eprint

from typing import *


class VeinData:
    def __init__(self,
                 data_set : Dict[str,ad.AnnData],
                 radius : float,
                 use_genes : Optional[Iterable[str]] = None,
                 verbose : bool = False,
                 weight_by_distance : bool = False,
                 sigma : float = 1.0,
                 individual_key : str = "sample",
                 )->None:


        """Class to hold data for classification

        Takes a dictionary with anndata objects (one file per sample) and will
        enable extraction of distances or expression of profiles from each
        sample. Will also compute the distance to each vein for every spot.


        Parameters:
        ----------

        data_set : Dict[str,ad.AnnData]
            Dictionary with AnnData files, one for each sample
        radius : float
            Distance that defines neighborhood of a veing. Spots
            within radius distance to a vein will be considered as
            residing within the neighborhood.
        use_genes : Optional[Iterable[str]] (None)
             specify a select set of gene which should be
             included in the analysis.
        verbose : bool (False)
             set to True if information should be printed
             during construction and extraction of class
             elements.
        weight_by_distance : bool (False)
             weight the assembled vein expression profiles
             by distance to profile (a form of weighted average)
        sigma : float (1.0) bandwidth to use when using the
             'weight_by_distance' optional, only impacts the results when
             'weight_by_distance' is set to True. The weight's an expression
             profile i used to compute a vein expression profile with distance
             d_i to the vein is defined as :

             w_i = exp(-[d_i / sigma])

        individual_key : str
             name of AnnData obs attribute that holds identifiers for each
             individual.

        """

        # TODO: currently in there's a mismatch between
        # naming in code and data. The mapping is:
        # h5ad-files : code
        # sample : individual
        # replicate : sample

        if use_genes is None:
            self.genes = pd.Index([])
            for data in data_set.values():
                self.genes = self.genes.union(data.var.index)
        else:
            self.genes = pd.Index([x for x in use_genes if x in data.var.index])

        self.data = {s:d.uns["mask"] for s,d in data_set.items()}
        self.samples = list(data_set.keys())
        self.n_samples = len(self.samples)
        self.individuals = list(set([d.uns[individual_key] for\
                                     d in data_set.values()]))

        self.sample_to_individual = {x:data_set[x].uns[individual_key] for\
                                     x in self.samples}

        self.individual_to_sample  : Dict[str,List[str]] = {d.uns[individual_key]:[] for\
                                                            d in data_set.values()}

        for sample in self.samples:
            self.individual_to_sample[self.sample_to_individual[sample]].append(sample)

        self._radius = radius
        self._sigma = sigma
        self._weight_by_distance
        self._verbose_state = verbose
        self._build(weight_by_distance = self._weight_by_distance,
                    sigma = self._sigma,
                    )

    def get_vein_crds(self,
                      vein_id : str,
                      )->np.ndarray:

        """Get coordinates of a specific vein

        Parameters:
        ----------
        vein_id : str
            id of vein to access coordinates for

        Returns:
        --------
        Numpy Array with coordinates of all pixels assigned
        to vein of said id

        """

        sample,idx = vein_id.split("_")
        sel = (self.data[sample]["id"] == int(idx)).values
        crds = self.data[sample][["x","y"]].values[sel,:]
        return crds

    def get_distance_to_vein(self,
                             crd : np.ndarray,
                             vein_id : str,
                             )->np.ndarray:

        """ Get distance for points to specific vein

        For a set of coordinates [n_points x 2 ] get the
        distance to a given vein.

        Parameters:
        ----------

        crd : np.ndarray
            coordinates to query for distance to vein
        vein_id : str
            id of vein to access coordinates for

        Returns:
        --------

        Array of distances for each coordinate pair
        to the given vein.

        """

        dists = cdist(crd,self.get_vein_crds(vein_id))
        dists = np.min(dists,axis = 1)
        return dists



    def _build(self,
               weight_by_distance : Optional[bool] = False,
               sigma : Optional[float] = 1,
               verbose : Optional[bool] = None,
               )->pd.DataFrame:

        """ build expression profiles


        weight_by_distance : bool (False)
             weight the assembled vein expression profiles
             by distance to profile (a form of weighted average)

        sigma : float (1.0) bandwidth to use when using the
             'weight_by_distance' optional, only impacts the results when
             'weight_by_distance' is set to True. The weight's an expression
             profile i used to compute a vein expression profile with distance
             d_i to the vein is defined as :

             w_i = exp(-[d_i / sigma])

        verbose : Optional[bool]
             set to true if verbose mode should be used


        Returns:
        --------

        Pandas DataFrame with NEPs for each vein

        """


        all_genes = pd.Index([])

        if verbose is None:
            verbose = self._verbose_state

        if weight_by_distance is None:
            weight_by_distance = self._weight_by_distance

        if sigma is None:
            sigma = self._sigma

        self.vein_expr_dict = dict()

        _id_to_type = {}

        for sample,data in self.data_set.items():
            # set variables for ease of access
            vein_data = data.uns["mask"]
            data_crd = data.obs[["x","y"]].values

            # list tp hold individual veins average 
            # expression levels
            vein_expr = list()

            # iterate over all individual veins
            for vein in np.unique(vein_data["id"]):
                # get index of vein members in the mask-data
                pos = vein_data["id"] == vein
                # get type of vein
                vc = vein_data["type"][pos].values[0]
                # update mapping between id and type
                _id_to_type.update({sample + "_" + str(vein):vc})

                # get coordinates of specific vein
                vein_crd = vein_data[["x","y"]].values[pos,:]

                # compute distances for spots
                # to pixels in specific vein
                dmat = cdist(data_crd,
                             vein_crd)

                # get minimum distances
                dmat = np.min(dmat,axis = 1)

                # get indices of spots
                # within specified radius
                in_box = dmat < self._radius
                n_in_box = sum(in_box)
                box_dists = dmat[in_box]
                in_box = data.obs.index[in_box]

                if verbose:
                    print("Sample : {} | Vein {} | Spots used : {}".format(sample,
                                                                           str(vein),
                                                                           str(n_in_box),
                                                                           ))

                # get average expression for 
                # neighborhood of vein
                tmp_data = pd.DataFrame(np.zeros((len(in_box),len(self.genes))),
                                        index = in_box,
                                        columns = self.genes,
                                        )

                inter_genes = pd.Index(self.genes).intersection(data.var.index)

                tmp_data.loc[in_box,inter_genes] = data.to_df()\
                                                       .loc[in_box,inter_genes]

                if weight_by_distance:
                    tmp_data.loc[:,:] = np.exp(-box_dists[:,np.newaxis]/sigma) *\
                        tmp_data.values

                x_vein = np.mean(tmp_data.values,axis = 0)

                # store average expression
                vein_expr.append(x_vein)

            # assemble individual vein expression df
            vein_expr = pd.DataFrame(vein_expr,
                                     columns = self.genes,
                                     index = [sample + "_" + str(x) for x\
                                              in np.unique(vein_data["id"])],
                                     )

            self.vein_expr_dict[sample] = vein_expr

        # function to map vein id to type
        self.__id_to_type = np.vectorize(lambda x: _id_to_type[x])

    def id_to_type(self,
                   x : np.ndarray,
                   )->np.ndarray:

        """Get type of vein specified by id

        Parameters:
        -----------
        x : np.ndarray
             vein id

        Returns:
        -------
        Numpy array with type of each vein

        """
        return self.__id_to_type(x)

    def get_expression(self,
                       ids : Union[List[str],pd.Index,str],
                       return_type : bool = False,
                       )->Union[pd.DataFrame,Tuple[pd.DataFrame,pd.DataFrame]]:

        """Get expression profiles from id


        Parameters:
        -----------

        ids : Union[List[str],pd.Index,str]
            List, Pandas Index or single string with
            id(s) of vein(s) to get expression of.
        return_type : bool (False)
            set to True if type of veins should be
            returned.

        Returns:
        --------
        Pandas DataFrame with NEPs for all specified veins,
        also returns Pandas DataFrame of each vein's type if
        return_type is set to True.

        """

        if isinstance(ids,str):
            if ids == "all":
                        tmp = pd.concat([v for v in self.vein_expr_dict.values()])
                        types = pd.DataFrame(self.id_to_type(tmp.index),
                                            index = tmp.index,
                                            columns = ["vein_type"],
                                            )
                        return (tmp,types)
            else:
                ids : List[str] = [ids]

        is_sample = all([x in self.vein_expr_dict.keys() for\
                         x in ids])

        is_individual = all([x in self.sample_to_individual.values() for\
                             x in ids])
        if is_individual: 
            all_ids = list(reduce(lambda x,y: x+y,[self.individual_to_sample[x] for\
                                                   x in ids]))

            tmp = pd.concat([self.vein_expr_dict[x] for\
                             x in all_ids])

        elif is_sample:
            tmp = pd.concat([self.vein_expr_dict[x] for\
                             x in ids])

        else:
            tmp = pd.concat(list(self.vein_expr_dict.values()))
            is_id =  all([x in tmp.index for x in ids])
            if not is_id:
                raise ValueError("Not a valid sample or vein id")
            else:
                tmp = tmp.loc[ids,:]
                if not return_type:
                    return tmp
                else:
                    types = self.id_to_type(tmp.index)
                    types = pd.DataFrame(types,
                                        index = ids,
                                        columns = ["vein_type"],
                                        )

                    return (tmp,types)

        if not return_type:
            return tmp
        else:
            types = self.id_to_type(tmp.index)
            types = pd.DataFrame(types,
                                index = tmp.index,
                                columns = ["vein_type"],
                                )
            return (tmp,types)

    def get_vein_ids(self,
                     ids : Union[List[str],str],
                     return_types : bool = False,
                     select_types : Optional[Union[List[str],str]] = None,
                     )->Union[List[str],Tuple[List[str],List[str]]]:

        """Get vein ids associated with
        a given sample(s).

        Parameters:
        ----------

        ids : Union[List[str],str]
             sample names
        return_types : bool = False
             set to True if vein types should be returned
        select_types : Optional[Union[List[str],str]] (None)
              Only get veins of types listed by select_types

        Returns:
        --------

        Pandas DataFrame with ids of veins associated with
        the specified sample. Also returns Pandas DataFrame with
        types of veins.



        """

        if isinstance(ids,str):
            ids = [ids]

        if isinstance(select_types,str):
            select_types = [select_types]

        if not all([x in self.vein_expr_dict.keys() for x in ids]):
            raise ValueError("Not a valid sample name")
        else:
            names = pd.Index([])
            for sample in ids:
                names = names.append(self.vein_expr_dict[sample].index)

        types = self.id_to_type(names).tolist()

        if select_types is not None:
            if all([x in types for x in select_types]):
                names,types = map(list,zip(*[(n,t) for n,t in zip(names,types) if\
                                             t in select_types]))
            else:
                raise ValueError("Not a valid type to select")

        if return_types:
            return (names,types)
        else:
            return names


class Model:
    """Model for vein classification

    Class used to conduct the training,
    prediction and classification.

    Arguments:
    ---------

    vein_data : VeinData
        VeinData object holding the data set
        to be used in the analysis. 
    verbose : bool (False)
        set to True to enable verbose mode

    Keyword Arguments:
    ------------------

    clf_params : Dict[str,Any]
         classifier parameters, should be
         compatible with the LogisticRegression
         class from sklearn.


    """
    def __init__(self,
                 vein_data : VeinData,
                 verbose : bool = False,
                 **kwargs,
                 )->None:

        self.data = vein_data
        self._verbose_state = verbose
        self.fitted = False

        clf_params = kwargs.get("clf_params",None)

        if clf_params is not None:
            self.clf = LR(**clf_params)
        else:
            self.clf = LR(random_state = 1337,
                          max_iter = 1000,
                          penalty ="l2",
                          fit_intercept = True,
                          )


    def fit(self,
            train_on : Union[List[str],pd.Index,str],
            exclude_class : Optional[Union[List[str],str,pd.Index]]=None,
            verbose : Optional[bool] = None,
            )->None:

        """Fit data set

        Parameters:
        -----------

        train_on : Union[List[str],pd.Index,str]
            vein or sample ids to train on
        exclude_class : Optional[Union[List[str],str,pd.Index]] (None)
            classes to exclude, will be subtracted from train on
        verbose : Optional[bool] (None)
            set to True to enable verbose mode

        """

        if verbose is None:
            verbose = self._verbose_state

        if verbose:
            iprint("Training on samples: {}"\
                  .format((', '.join(train_on) if \
                           isinstance(train_on,list) else \
                           train_on)))

        objs : List[pd.Index] = []
        _objs = [train_on,exclude_class]

        for obj in _objs:
            if isinstance(obj,str):
                if obj != "all":
                    objs.append(pd.Index([obj]))
            elif isinstance(obj,list):
                objs.append(pd.Index(obj))

        train_on,exclude_class = objs

        train_data,train_labels = self.data.get_expression(train_on,
                                                           return_type = True,
                                                           )

        if exclude_class is not None:
            if not all(exclude_class.isin(np.unique(train_labels.values))):
                iprint(">In Training :  some excluded classes not present in the data")

            keep = train_labels.isin(exclude_class).values.flatten()
            train_data = train_data.iloc[~keep,:]
            train_labels = train_labels.iloc[~keep,:]

        self.clf = self.clf.fit(train_data.values,
                                train_labels.values.flatten(),
                                )

        if verbose:
            acc = self.clf.score(train_data.values,
                                 train_labels.values.flatten(),
                                 )

            iprint("Accuracy on training data is >> {:0.3f}%".format(acc * 100))

        self.fitted = True

    def predict(self,
                predict_on : Union[List[str],str],
                exclude_class : Optional[Union[List[str],str]]=None,
                return_probs : bool = False,
                return_accuracy : bool = False,
                verbose : Optional[bool] = None,
                )->Union[np.ndarray,Tuple[np.ndarray,float]]:

        """Predict on specified data

        Predict the type of specified veins (sample or replicate),
        using a trained model. An Exception will be raised if the
        model is not fitted yet.


        Parameters:
        ----------

        predict_on : Union[List[str],str]
            samples, veins or individuals to predict on
        exclude_class : Optional[Union[List[str],str]] (None)
            types to exclude from the prediction
        return_probs : bool (False)
            return soft classification (probabilities)
        return_accuracy : bool (False)
            return accuracy values
        verbose : Optional[bool] (None)
            set to True to enable verbose mode


        Returns:
        -------
        Numpy Array with predicted classes or probability of
        belonging to respective class if 'return_probs' is
        set to True. Set accuracy values will also be
        returned (second argument) if 'return_accuracy' is
        set to True.


        """

        if verbose is None:
            verbose = self._verbose_state
        if verbose:
            iprint("Testing on samples: {}"\
                  .format((', '.join(predict_on) if \
                           isinstance(predict_on,list) else \
                           predict_on)))

        if not self.fitted:
            raise Exception("Model not fitted")

        _objs = [predict_on,exclude_class]
        objs : List[pd.Index] = []

        for obj in objs:
            if isinstance(obj,str):
                if obj != "all":
                    objs.append(pd.Index([obj]))
            elif isinstance(obj,list):
                objs.append(pd.Index(obj))

        predict_on,exclude_class = objs


        pred_data,pred_labels = self.data.get_expression(predict_on,
                                                         return_type = True,
                                                         )

        if exclude_class is not None:
            if not all(exclude_class.isin(np.unique(pred_labels.values))):
                iprint(">In Test | some excluded classes not present in the data")
            keep = pred_labels.isin(exclude_class).values.flatten()
            pred_data = pred_data.iloc[~keep,:]
            pred_labels = pred_labels.iloc[~keep,:]

        if return_probs:
            predicted = self.clf.predict_proba(pred_data.values)
            predicted = pd.DataFrame(predicted,
                                     columns = self.clf.classes_,
                                     index = pred_data.index,
                                     )
            return predicted
        else:
            predicted = self.clf.predict(pred_data.values)
            predicted = pd.DataFrame(predicted,
                                     index = pred_data.index,
                                     columns = ["vein_type"],
                                     )
            if return_accuracy:
                accuracy = np.sum(predicted.values.flatten() == pred_labels.values.flatten())
                accuracy /= len(predicted)

                return (predicted,accuracy)
            else:
                return predicted

    def eval(self,
             eval_on : Union[List[str],str],
             exclude_class : Optional[Union[List[str],str]]=None,
             verbose : Optional[bool] = None,
             )->float:

        """Evaluate prediction

        Calls predict on with 'return_accuracy' set to True,
        provided for improved interaction with API.

        Parameters:
        ----------

        eval_on : Union[List[str],str]
            samples, veins or individuals to evaluate
            prediction of
        exclude_class : Optional[Union[List[str],str]] (None)
            types to exclude from the prediction
        verbose : Optional[bool] (None)
            set to True to enable verbose mode

        Returns:
        --------
        Accuracy value computed over the set of specified
        veins.

        """


        _,accuracy = self.predict(eval_on,
                                  exclude_class = exclude_class,
                                  return_accuracy = True,
                                  verbose = verbose,
                                  )

        return accuracy


    def _generate_results_object(self,
                                 )->Dict[str,List[Union[str,float]]]:

        """Generate a results DataFrame (Helper)"""

        results_df : Dict[str, List[Union[str,float]]] = dict(pred_on = [],
                                                              train_on = [],
                                                              accuracy = [],
                                                              )

        return results_df


    def cross_validation(self,
                          k : int,
                          exclude_class : Optional[Union[List[str],str]]=None,
                          by : str = "sample",
                          verbose : Optional[bool] = None,
                          )->pd.DataFrame:

        """Cross validate model performance

        Performs validation by excluding 'k' samples or
        individual during training and then predicting on
        these samples.

        Parameters:
        ----------
        k : int
           number of samples/individuals to set
           aside during each iteration
        exclude_class : Optional[Union[List[str],str]] (None)
            name of classes to exclude during
            validation
        by : str (sample)
            validate by sample or individual
        verbose : Optional[bool] (None)
            set to True to enable Verbose mode

        Returns:
        --------

        Pandas DataFrame with results from cross
        validation.

        """

        from itertools import combinations

        valid_by = ["sample","individual"]

        if by not in valid_by:
            ValueError("by parameter must be one of {}"\
                       .format(", ".join(valid_by)))

        if by == "sample":
            if k > self.data.n_samples - 1:
                raise ValueError("k cannot be larger than number of samples")
            n_test = k
            n_train = self.data.n_samples - n_test
            combs = list(combinations(self.data.samples,n_train))

            name_list = self.data.samples

        elif by == "individual":
            by_individual : Dict[str,List[str]] = dict()
            for s in self.data.samples:
                sid = self.data.sample_to_individual[s]
                if sid in by_individual.keys():
                    by_individual[sid].append(s)
                else:
                    by_individual[sid] = [s]

            combs = combinations(list(by_individual.keys()),
                                 len(by_individual)-k)

            name_list = self.data.individuals

        results_df = self._generate_results_object()

        for comb in combs:
            train_on = list(comb)
            predict_on = list(filter(lambda x: x not in train_on,
                                     name_list))

            self.fit(train_on=train_on,
                     exclude_class=exclude_class,
                     verbose = verbose,
                     )

            acc = self.eval(eval_on=predict_on,
                            exclude_class=exclude_class,
                            verbose = verbose,
                            )

            results_df["pred_on"].append(', '.join(predict_on))
            results_df["train_on"].append(', '.join(train_on))
            results_df["accuracy"].append(acc)

        return pd.DataFrame(results_df)
