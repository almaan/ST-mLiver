import os
import os.path as osp
from operator import itemgetter

main_dir="/home/alma/w-projects/help/franzi/repo/data/stereoscope/st/set1"

cnt_dir ="raw_counts"
spt_dir = "spotfiles"
out_dir = "curated_counts"

cnts = os.listdir(osp.join(main_dir,cnt_dir))
cnts = {'_'.join(p.split("_")[0:2]):osp.join(main_dir,cnt_dir,p) for p in cnts}
spts = os.listdir(osp.join(main_dir,spt_dir))
spts= {'_'.join(itemgetter(0,2)(p.split("_"))):osp.join(main_dir,spt_dir,p) for p in spts}


for sample in list(cnts.keys()):
    _cnt = pd.read_csv(cnts[sample],sep = '\t',header= 0,index_col = 0)
    _spt = pd.read_csv(spts[sample],sep = '\t',header= 0)

    new_idx = pd.Index([str(x)+"x"+str(y) for x,y in zip(_spt["x"],_spt["y"])])
    _spt.index = new_idx

    inter = _cnt.index.intersection(_spt.index)

    _cnt = _cnt.loc[inter,:]

    _cnt.to_csv(osp.join(main_dir,out_dir,"ut_"+osp.basename(cnts[sample])),sep = '\t')
