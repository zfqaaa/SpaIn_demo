import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp

from GraphST import GraphST

dataset = 'Mouse_Embryo'
# Run device，by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# the location of R, which is necessary for mclust algorithm. Please replace it with local R installation path
os.environ['R_HOME'] = 'E:\R\R-4.3.2'

# the number of clusters
n_clusters = 22

# read data
file_path = 'F:/ST_data/Stereo/' #please replace 'file_path' with the download path
adata = sc.read_h5ad(file_path + 'E9.5_E1S1.MOSTA.h5ad')
adata.var_names_make_unique()
# define model
model = GraphST.GraphST(adata, datatype='Stereo', device=device)

# run model
adata = model.train()
# clustering
from GraphST.utils import clustering

tool = 'mclust' # mclust, leiden, and louvain

# clustering
from GraphST.utils import clustering

if tool == 'mclust':
   clustering(adata, n_clusters, method=tool)
elif tool in ['leiden', 'louvain']:
   clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)

import matplotlib.pyplot as plt
adata.obsm['spatial'][:, 1] = -1*adata.obsm['spatial'][:, 1]
plt.rcParams["figure.figsize"] = (3, 4)
plot_color=["#F56867","#556B2F","#C798EE","#59BE86","#006400","#8470FF",
           "#CD69C9","#EE7621","#B22222","#FFD700","#CD5555","#DB4C6C",
           "#8B658B","#1E90FF","#AF5F3C","#CAFF70", "#F9BD3F","#DAB370",
          "#877F6C","#268785", '#82EF2D', '#B4EEB4']

ax = sc.pl.embedding(adata, basis="spatial",
                    color="domain",
                    s=30,
                    show=False,
                    palette=plot_color,
                    title='GraphST')
ax.axis('off')
ax.set_title('Mouse Embryo E9.5')
plt.show()