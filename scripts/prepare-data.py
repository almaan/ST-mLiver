#!/usr/bin/env python3


import pandas as pd
import numpy as np
import anndata as ad

from sklearn.cluster import KMeans,DBSCAN
from scipy.spatial.distance import cdist

import sys
import os
import os.path as osp

from typing import Tuple, Union,Dict
import argparse as arp

from PIL import Image

from pathlib import Path

import yaml

class YAML_CONFIG:
    cnt = "count_data"
    spt = "spot_data"
    img = "image"
    mask = "mask"
    rgb = "rgb"
    def __init__():
        pass



def d1tod2(x : Union[int,np.ndarray],
           ncols :int,
          )->Union[np.ndarray,Tuple[int]]:

    r = x // ncols
    c = x % ncols

    return (r,c)


def prep_anndata(data : Union[Dict[str,pd.DataFrame],Dict[str,Path]],
                 n_types : int,
                 include_distances : bool = True,
                 segment : bool = True,
                 alpha : float = 0.05,
                 scale_factor : float = 1.0,
                 )->ad.AnnData:


    cnt,spt = data["cnt"],data["spt"]
    img,mask = data["img"],data["mask"]


    obs = dict(_x = spt["x"].values,
               _y = spt["y"].values,
               x = spt["pixel_x"].values *scale_factor,
               y = spt["pixel_y"].values *scale_factor,
               )

    obsm = dict()

    mask = Image.open(mask)
    mask = np.asarray(mask).astype(int)

    mx,my,mc = mask.shape
    cmask = mask.reshape(mx*my,mc)
    del mask

    km = KMeans(n_clusters = n_types + 1)
    cidx = km.fit_predict(cmask)
    del cmask


    background_idx = np.argmin(np.linalg.norm(km.cluster_centers_,
                                              axis =1))

    cidx = cidx.astype(str).astype(object)

    if "rgb" in data.keys():
        for name,clr in data["rgb"].items():
            c_to_n = np.argmin(cdist(km.cluster_centers_[:,0:3],
                                     clr.reshape(1,-1)).flatten(),
                               )
            pos = cidx == str(c_to_n)
            cidx[pos] = str(name)


    background_idx = np.argmin(np.linalg.norm(km.cluster_centers_,
                                              axis =1))
    background_idx = str(background_idx)

    include_idx = []
    mask_long = dict(x = [],
                     y = [],
                     type = [],
                     )

    for cluster in np.unique(cidx):
        if cluster != background_idx:
            pos = np.where(cidx == cluster)[0]
            _y,_x = d1tod2(pos,
                           ncols=my)

            mask_long["x"] += list(_x)
            mask_long["y"] += list(_y)
            mask_long["type"] += [cluster] * len(pos)
            include_idx.append(cluster)

    mask_long = pd.DataFrame(mask_long)

    if include_distances:
        dists = dict()
        for cluster in include_idx:
            pos = mask_long["type"].values == cluster

            cluster_crd = mask_long[["x","y"]].values[pos,:]

            dmat = cdist(spt[["pixel_x","pixel_y"]].values * scale_factor,
                          cluster_crd
                          )

            dmat[dmat == 0] = dmat.max()
            dmat = np.min(dmat,axis = 1)
            colname = "dist_type_{}".format(cluster)
            dists.update({colname : dmat})

        dists = pd.DataFrame(dists,
                             index = spt.index,
                             )
        
        obsm.update({"vein_distances":dists})

    if segment:
        segs = np.zeros(mask_long.shape[0]).astype(int)
        for k,ii in enumerate(include_idx):
            pos = mask_long["type"] == ii
            crd = mask_long[["x","y"]].values[pos,:]

            ncrd = crd.shape[0]
            if ncrd >= 1e3:
                sample_crd = np.random.choice(np.arange(ncrd),size=int(1e3))
                dist = cdist(crd[sample_crd,:],crd[sample_crd,:])
            else:
                dist = cdist(crd,crd)

            dist[dist == 0] = dist.max()
            eps = np.quantile(dist,alpha)
            seg = DBSCAN(eps).fit_predict(crd)

            if k > 0:
                mx = segs.max()
                mn = seg.min()
                diff = (mx + 1) - mn
                segs[pos] = seg + diff
            else:
                segs[pos] = seg

        mask_long["id"] = segs

        mask_long["id"] -= mask_long["id"].values.min()

    img = Image.open(img)
    img = np.asarray(img).astype(int)

    if "rgb" not in data.keys():
        mask_long["type"] -= mask_long["type"].values.min()


    uns = dict(img = img,
               mask = mask_long,
               )


    obs = pd.DataFrame(obs,
                       index = spt.index)

    var = pd.DataFrame(dict(gene = cnt.columns),
                       index = cnt.columns)

    adata = ad.AnnData(X = cnt.values,
                       var = var,
                       obs = obs,
                       obsm = obsm,
                       uns = uns,
                       )

    return adata


def read_yaml(filename : Path,
              )->Union[Dict[str,pd.DataFrame],
                       Dict[str,Path]]:


    with open(filename,"r+") as f:
        pths = yaml.load(f)


    for k,p in pths.items():
        if k != YAML_CONFIG.rgb:
            pths[k] = Path(p)

    cnt = pd.read_csv(pths[YAML_CONFIG.cnt],
                      sep = '\t',
                      header = 0,
                      index_col = 0)

    spt = pd.read_csv(pths[YAML_CONFIG.spt],
                      sep = '\t',
                      header = 0,
                      index_col = None)


    spt.index = pd.Index([str(x)+"x"+str(y) for x,y\
                          in zip(spt["x"].values,
                                 spt["y"].values)])


    inter = spt.index.intersection(cnt.index)

    assert len(inter) > 0,\
        "No matching entries between"\
        "count and spot data"

    cnt = cnt.loc[inter,:]
    spt = spt.loc[inter,:]

    data = dict(cnt = cnt,
                spt = spt,
                img = pths[YAML_CONFIG.img],
                mask = pths[YAML_CONFIG.mask]
                )

    if YAML_CONFIG.rgb in pths.keys():
        for k,v in pths[YAML_CONFIG.rgb].items():
            pths[YAML_CONFIG.rgb][k] = np.array(v) 
        data.update({"rgb":pths[YAML_CONFIG.rgb]})

    return data


def main():

    prs = arp.ArgumentParser()
    aa = prs.add_argument

    aa("-i",
       "--input",
       type = str,
       required = True,
       help = "data configuratio"\
       " file. Should be in YAML format",
       )

    aa("-n","--n_types",
       type = int,
       required = True,
       help = "number of different object types",
       )

    aa("-o","--out_dir",
       type = str,
       required = False,
       default = None,
       help = "output directory",
       )

    aa("-s","--segment",
       action = "store_true",
       default = False,
       help = "try to assign unique"\
       " id to each of the objects"
       )

    aa("-d","--include_distances",
       default = False,
       action = "store_true",
       help = "include minimal distances"\
       " for each spot to nearest object of"\
       " each type."
       )

    aa("-a","--alpha",
       type = float,
       default = 0.05,
       help = "quantile value to use"\
       " in segmentation."
       )

    aa("-sf","--scale_factor",
        type = float,
        default = 1,
        help = "scale factor",
        )



    args = prs.parse_args()


    supported_data = {"yaml": read_yaml}
    insplit = args.input.split(".")
    stem = '.'.join(insplit[0:-1])
    ext = insplit[-1]

    read_data = supported_data.get(ext,None)

    if read_data is None:
        print("[ERROR] : format {} is not supported".format(ext))
        sys.exit(-1)
    else:
        print("[INFO] : reading : {}".format(args.input))
        data = read_data(args.input)

    adata = prep_anndata(data,
                         n_types = args.n_types,
                         alpha = args.alpha,
                         segment = args.segment,
                         include_distances = args.include_distances,
                         scale_factor = args.scale_factor,
                         )

    if args.out_dir is None:
        args.out_dir = osp.dirname(osp.abspath(args.input))
    else:
        if not osp.exists(args.out_dir):
            os.mkdir(args.out_dir)

    out_pth = osp.join(args.out_dir,stem + ".h5ad")

    adata.write_h5ad(out_pth)


if __name__ == "__main__":
    main()
