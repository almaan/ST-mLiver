#!/usr/bin/env python3

# script to update h5ad files
# should not be necessary once prepare-data.py is updated

import anndata as ad
import pandas as pd
import sys
import os.path as osp


pth = sys.argv[1]

adata = ad.read_h5ad(pth)

sample,replicate = osp.basename(pth).split(".")[0].split("-")

adata.uns["sample"] = sample
adata.uns["replicate"] = replicate

adata.obs["sample"] = sample
adata.obs["replicate"] = replicate


adata.write_h5ad(osp.join(osp.dirname(pth),"new-" + sample + "-" + replicate + ".h5ad"))
