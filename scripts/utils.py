#!/usr/bin/env python3


import numpy as np
import pandas as pd
import anndata as ad

from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression as LR
from functools import reduce

from typing import Dict,Optional,Union,Tuple,List,Callable

def iprint(s : str,
           )->None:

    print("[INFO] : {}".format(s))

def eprint(s : str,
           )->None:

    print("[ERROR] : {}".format(s))

class VeinData:
    def __init__(self,
                 data_set : Dict[str,ad.AnnData],
                 radius : float,
                 get_individual_id : Optional[Callable] = None,
                 use_genes : Optional[str] = None,
                 verbose : bool = False,
                 )->None:


        if use_genes is None:
            self.genes = pd.Index([])
            for data in data_set.values():
                self.genes = self.genes.union(data.var.index)
        else:
            self.genes = use_genes


        self.samples = list(data_set.keys())
        self.n_samples = len(self.samples)

        if get_individual_id is None:
            get_individual_id = lambda x : x.split("-")[0]

        self.sample_to_individual = {x:get_individual_id(x) for x in self.samples}
        self.individual_to_sample = {v:[] for v in self.sample_to_individual.values()}
        for sample in self.samples:
            self.individual_to_sample[get_individual_id(sample)].append(sample)

        self.individuals = list(self.individual_to_sample.keys())


        self._radius = radius
        self._verbose_state = verbose
        self._build(data_set)


    def _build(self,
               data_set : Dict[str,ad.AnnData],
               verbose : bool = None,
               ):

        all_genes = pd.Index([])

        if verbose is None:
            verbose = self._verbose_state

        self.vein_expr_dict = dict()

        _id_to_type = {}

        for sample,data in data_set.items():
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
                in_box = data.obs.index[in_box]

                if verbose:
                    print("Sample : {} | Vein {} | Spots used : {}".format(sample,
                                                                           vein,
                                                                           sum(in_box),
                                                                           ))

                # get average expression for 
                # neighborhood of vein
                tmp_data = pd.DataFrame(np.zeros((len(in_box),len(self.genes))),
                                        index = in_box,
                                        columns = self.genes,
                                        )

                inter_genes = pd.Index(self.genes).intersection(data.var.index)

                tmp_data.loc[in_box,inter_genes] = data.to_df().loc[in_box,inter_genes]

                x_vein = np.mean(tmp_data.values,axis = 0)

                # store average expression
                vein_expr.append(x_vein)

            # assemble individual vein expression df
            vein_expr = pd.DataFrame(vein_expr,
                                     columns = self.genes,
                                     index = [sample + "_" + str(x) for x in np.unique(vein_data["id"])],
                                     )

            self.vein_expr_dict[sample] = vein_expr

        # function to map vein id to type
        self.__id_to_type = np.vectorize(lambda x: _id_to_type[x])

    def id_to_type(self,x)->np.ndarray:
        return self.__id_to_type(x)

    def get_expression(self,
                       ids : Union[List[str],pd.Index,str],
                       return_type : bool = False,
                       )->Union[pd.DataFrame,Tuple[pd.DataFrame,pd.DataFrame]]:

        if isinstance(ids,str):
            if ids == "all":
                        tmp = pd.concat([v for v in self.vein_expr_dict.values()])
                        types = pd.DataFrame(self.id_to_type(tmp.index),
                                            index = tmp.index,
                                            columns = ["vein_type"],
                                            )
                        return (tmp,types)
            else:
                ids = [ids]

        is_sample = all([x in self.vein_expr_dict.keys() for x in ids])
        is_individual = all([x in self.sample_to_individual.values() for x in ids])
        if is_individual: 
            all_ids = list(reduce(lambda x,y: x+y,[self.individual_to_sample[x] for x in ids]))
            tmp = pd.concat([self.vein_expr_dict[x] for x in all_ids])

        elif is_sample:
            tmp = pd.concat([self.vein_expr_dict[x] for x in ids])

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
                          max_iter = 100,
                          penalty ="l2",
                          solver = "liblinear",
                          fit_intercept = False,
                          )


    def fit(self,
            train_on : Union[List[str],pd.Index,str],
            exclude_class : Optional[Union[List[str],str,pd.Index]]=None,
            verbose : Optional[bool] = None,
            )->None:

        if verbose is None:
            verbose = self._verbose_state

        if verbose:
            iprint("Training on samples: {}"\
                  .format((', '.join(train_on) if \
                           isinstance(train_on,list) else \
                           train_on)))

        objs = [train_on,exclude_class]

        for k,obj in enumerate(objs):
            if isinstance(obj,str):
                if obj != "all":
                    objs[k]= pd.Index([obj])
            elif isinstance(obj,list):
                objs[k] = pd.Index(obj)

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
                verbose : bool = None,
                )->Union[np.ndarray,Tuple[np.ndarray,np.ndarray]]:

        if verbose is None:
            verbose = self._verbose_state
        if verbose:
            iprint("Testing on samples: {}"\
                  .format((', '.join(predict_on) if \
                           isinstance(predict_on,list) else \
                           predict_on)))

        if not self.fitted:
            raise Exception("Model not fitted")

        objs = [predict_on,exclude_class]

        for k,obj in enumerate(objs):
            if isinstance(obj,str):
                if obj != "all":
                    objs[k]= pd.Index([obj])
            elif isinstance(obj,list):
                objs[k] = pd.Index(obj)

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


        _,accuracy = self.predict(eval_on,
                                  exclude_class = exclude_class,
                                  return_accuracy = True,
                                  verbose = verbose,
                                  )

        return accuracy


    def _generate_results_object(self,
                                 )->Dict[str,List]:

        results_df = dict(pred_on = [],
                          train_on = [],
                          accuracy = [],
                          )

        return results_df


    def k_fold_validation(self,
                          k : int,
                          exclude_class : Optional[Union[List[str],str]]=None,
                          by : str = "sample",
                          )->pd.DataFrame:

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
            by_individual = dict()
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
                     exclude_class=exclude_class)

            acc = self.eval(eval_on=predict_on,
                            exclude_class=exclude_class)

            results_df["pred_on"].append(', '.join(predict_on))
            results_df["train_on"].append(', '.join(train_on))
            results_df["accuracy"].append(acc)

        return pd.DataFrame(results_df)


