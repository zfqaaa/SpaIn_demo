import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import modify_spagcn as spg
#In order to read in image data, we need to install some package. Here we recommend package "opencv"
#inatll opencv in python
#!pip3 install opencv-python
import cv2
import STAGATE
import tensorflow as tf
from GraphST import GraphST
from GraphST.utils import clustering
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
os.environ['R_HOME'] = 'E:\R\R-4.3.2'
os.environ['R_USER'] = 'E:/R/Rwork'
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rcParams
config={
    "font.family":'serif',
    "font.size":12,
    "font.serif":['SimSun'],
    "mathtext.fontset":'stix',
    'axes.unicode_minus':False
}
rcParams.update(config)
counts_file = os.path.join('F:/ST_data/SEDR_analyses-master/data/Stero-seq/RNA_counts.tsv')
coor_file = os.path.join('F:/ST_data/SEDR_analyses-master/data/Stero-seq/position.tsv')
counts = pd.read_csv(counts_file, sep='\t', index_col=0,error_bad_lines = False)
coor_df = pd.read_csv(coor_file, sep='\t')
print(counts.shape, coor_df.shape)
counts.columns = ['Spot_'+str(x) for x in counts.columns]
coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
coor_df = coor_df.loc[:, ['x','y']]
print(coor_df.head())
adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
adata.obsm["spatial"] = coor_df.to_numpy()
sc.pp.calculate_qc_metrics(adata, inplace=True)
plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
plt.title("")
plt.axis('off')
plt.show()

used_barcode = pd.read_csv(os.path.join('F:/ST_data/SEDR_analyses-master/data/Stero-seq/used_barcodes.txt'), sep='\t', header=None)
used_barcode = used_barcode[0]
adata = adata[used_barcode,]
print(adata)


# plt.rcParams["figure.figsize"] = (5,4)
# sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
# plt.title("")
# plt.axis('off')
# plt.show()

sc.pp.filter_genes(adata, min_cells=50)
print('After flitering: ', adata.shape)

#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata.shape)
#
# STAGATE.Cal_Spatial_Net(adata, rad_cutoff=50)
# STAGATE.Stats_Spatial_Net(adata)
#
# tf.compat.v1.disable_eager_execution()
# x = tf.compat.v1.placeholder(tf.float32, [None, 784])
# adata = STAGATE.train_STAGATE(adata, alpha=0)



# sc.pp.pca(adata,n_comps=30)
# sc.pp.neighbors(adata, use_rep='X_pca')
# sc.tl.umap(adata)


'''训练NewgraphST'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GraphST.GraphST(adata, device=device)
# train model
adata = model.train()

# set radius to specify the number of neighbors considered during refinement
radius = 50
print(adata)
tool = 'mclust' # mclust, leiden, and louvain
# tool = 'louvain'
# the number of clusters
n_clusters = 8

# clustering

# clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.

clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)


print(adata)

sc.pp.neighbors(adata, use_rep='emb_pca',n_neighbors=10)
sc.tl.umap(adata)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.embedding(adata, basis="spatial", color="mclust", s=6, show=False, title='SpaInGNN')
plt.axis('off')
plt.savefig('F:/results/SpaInGNNste_clust.tif',dpi=300,bbox_inches='tight')
plt.show()

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata, color='mclust', title='SpaInGNN',show=False)
plt.savefig('F:/results/SpaInGNNste_ump.tif',dpi=300,bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(2,5,figsize=(1.7*5, 1.5*2), sharex=True, sharey=True)
axes = axes.ravel()

for i in range(n_clusters):
    sub = adata[adata.obs['mclust'] == i+1]
    sc.pl.spatial(sub, spot_size=30, color='mclust', ax=axes[i], legend_loc=None, show=False)
    axes[i].set_title(i)


xmin = adata.obsm['spatial'][:, 0].min()
xmax = adata.obsm['spatial'][:, 0].max()
ymin = adata.obsm['spatial'][:, 1].min()
ymax = adata.obsm['spatial'][:, 1].max()

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

plt.subplots_adjust(wspace=0, hspace=0.05)
plt.tight_layout()
plt.savefig('F:/results/SpaInGNN_Steroclust.tif',dpi=300,bbox_inches='tight')
plt.show()