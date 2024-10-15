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
import newGATE
import tensorflow as tf
from GraphST import GraphST
from GraphST.utils import clustering
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
os.environ['R_HOME'] = 'E:\R\R-4.3.2'
os.environ['R_USER'] = 'E:/R/Rwork'

from matplotlib import rcParams
# from keras import layers
# from keras import Input
config={
    "font.family":'serif',
    "font.size":12,
    "font.serif":['SimSun'],
    "mathtext.fontset":'stix',
    'axes.unicode_minus':False
}
rcParams.update(config)

from scanpy import read_10x_h5
#read_10x_h5返回的是带注释的数据矩阵，基因表达矩阵x obs_names细胞名称 var_names基因名称 var [基因ID 特征类型]
# adata = read_10x_h5("../tutorial/data/151673/sample_data.h5")
section_id = '151673'
input_dir = os.path.join('E:/ST_Data', section_id)
adata = sc.read_visium(path=input_dir, count_file=section_id+'_filtered_feature_bc_matrix.h5')

#读取空间坐标的位置
spatial=pd.read_csv("../tutorial/data/151673/tissue_positions_list.csv",sep=",",header=None,na_filter=False,index_col=0)

adata.obs["x1"]=spatial[1]
adata.obs["x2"]=spatial[2]
adata.obs["x3"]=spatial[3]
adata.obs["x4"]=spatial[4]
adata.obs["x5"]=spatial[5]
adata.obs["x_array"]=adata.obs["x2"]
adata.obs["y_array"]=adata.obs["x3"]
adata.obs["x_pixel"]=adata.obs["x4"]
adata.obs["y_pixel"]=adata.obs["x5"]
#Select captured samples
adata=adata[adata.obs["x1"]==1]
adata.var_names=[i.upper() for i in list(adata.var_names)]
adata.var["genename"]=adata.var.index.astype("str")
adata.write_h5ad("../tutorial/data/151673/sample_data.h5ad")
adata=sc.read("../tutorial/data/151673/sample_data.h5ad")
#Read in hitology image
img=cv2.imread("../tutorial/data/151673/histology.tif")
x_array=adata.obs["x_array"].tolist()
y_array=adata.obs["y_array"].tolist()
x_pixel=adata.obs["x_pixel"].tolist()
y_pixel=adata.obs["y_pixel"].tolist()

#Test coordinates on the image
#每个坐标周围20个像素设置为黑色
img_new=img.copy()
for i in range(len(x_pixel)):
    x=x_pixel[i]
    y=y_pixel[i]
    img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0

cv2.imwrite('./sample_results/151673_map.jpg', img_new)
s=1
b=49
#计算图像的邻接矩阵
adj=spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=True)
print(adj.shape)
# #If histlogy image is not available, spatrain can calculate the adjacent matrix using the fnction below
# adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
np.savetxt('../tutorial/data/151673/adj.csv', adj, delimiter=',')
adata=sc.read("../tutorial/data/151673/sample_data.h5ad")
adj=np.loadtxt('../tutorial/data/151673/adj.csv', delimiter=',')
adata.var_names_make_unique()
spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
#去除基因表达矩阵中的常数行和列，以及在所有样本中都没有表达的行和列
spg.prefilter_specialgenes(adata)
#Normalize and take log for UMI
# sc.pp.normalize_per_cell(adata)
# sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(adata)
# read the annotation
Ann_df = pd.read_csv(os.path.join('E:/ST_Data', '151673', '151673'+'_truth.txt'), sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"])
tf.compat.v1.disable_eager_execution()
newGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
newGATE.Stats_Spatial_Net(adata)

# print(adj)
#
# from newGATE.Train_STAGATE import prepare_graph_data
#
# prepare_graph_data(adj)
# from transformers import pipeline
adata = newGATE.train_STAGATE(adata, alpha=0.5, pre_resolution=0.6,
                              n_epochs=1000, save_attention=True)
print(adata)
# pre-clustering result
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color="expression_louvain_label", size=1.0, title='pre-clustering result')
# sc.pp.neighbors(adata, use_rep='STAGATE')




adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)#计算质量控制指标
# import seaborn as sns
# ###保留 total_counts 在5000到35000的细胞，下一步，我们保留线粒体基因 pct_counts_mt 占比小于20%的细胞。最后，做基因上的过滤：
# fig, axs = plt.subplots(1, 4, figsize=(15, 4))
# sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
# sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])#kde设置为true为启动内核密度图和Dist plot
# sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
# sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000], kde=False, bins=60, ax=axs[3])
# plt.show()



# Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GraphST.GraphST(adata, device=device)
# train model
adata = model.train()
# set radius to specify the number of neighbors considered during refinement
radius = 50
print(adata)
tool = 'mclust' # mclust, leiden, and louvain
# the number of clusters
n_clusters = 7
dataset = '151673'
# read data
# please replace 'file_fold' with the download path
file_fold = 'E:/ST_data/' + str(dataset)
# clustering
if tool == 'mclust':
   clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
elif tool in ['leiden', 'louvain']:
   clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)
 # add ground_truth
# df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
# df_meta_layer = df_meta['layer_guess']
# adata.obs['ground_truth'] = df_meta_layer.values
# filter out NA nodes
adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
adata


print(adata.obs['domain'])
print(adata.obs['Ground Truth'])
# calculate metric ARI
ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['Ground Truth'])
NMI= metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['Ground Truth'])
AMI = metrics.adjusted_mutual_info_score(adata.obs['domain'], adata.obs['Ground Truth'])
adata.uns['ARI'] = ARI
adata.uns['NMI'] = NMI
adata.uns['AMI'] = AMI

print('Dataset:', dataset)
print('ARI:', ARI)
print('NMI:', NMI)
print('AMI:', AMI)
# plotting spatial clustering result
sc.pl.spatial(adata,
              img_key="hires",
              color=["Ground Truth", "domain"],
              title=["Ground truth", "ARI=%.4f"%ARI],
              show=True)
# plotting predicted labels by UMAP
sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=10)
sc.tl.umap(adata)
# sc.pl.umap(adata, color='domain', title=['Predicted labels'], show=True)

plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata, color=["domain", "Ground Truth"], title=['new(ARI=%.2f)'%ARI, "Ground Truth"], show=True)

# ###簇的marker基因
# sc.tl.rank_genes_groups(adata, "domain", method="t-test")
# sc.pl.rank_genes_groups_heatmap(adata, groups="4", n_genes=10, groupby="domain")
# plt.show()

sc.tl.draw_graph(adata)
sc.tl.paga(adata, groups='mclust')
print(adata.var_names)
sc.pl.paga(adata, color=['mclust'])





# sc.tl.umap(adata)
# adata = newGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=7)
# obs_df = adata.obs.dropna()
# ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
# print('Adjusted rand index = %.2f' %ARI)
# plt.rcParams["figure.figsize"] = (3, 3)
# sc.pl.umap(adata, color=["mclust", "Ground Truth"], title=['STAGATE (ARI=%.2f)'%ARI, "Ground Truth"])
# plt.rcParams["figure.figsize"] = (3, 3)
# sc.pl.spatial(adata, color=["mclust", "Ground Truth"], title=['STAGATE (ARI=%.2f)'%ARI, "Ground Truth"])
# #louvain聚类
# sc.tl.umap(adata)
# sc.tl.louvain(adata, resolution=1)
# adata.uns['louvain_colors']=['#aec7e8', '#9edae5', '#d62728', '#dbdb8d', '#ff9896',
#                              '#8c564b', '#696969', '#778899', '#17becf', '#ffbb78',
#                              '#e377c2', '#98df8a', '#aa40fc', '#c5b0d5', '#c49c94',
#                              '#f7b6d2', '#279e68', '#b5bd61', '#ad494a', '#8c6d31',
#                              '#1f77b4', '#ff7f0e']
# plt.rcParams["figure.figsize"] = (4, 4)
# sc.pl.spatial(adata, img_key="hires", color="louvain", size=1.0)
# plt.rcParams["figure.figsize"] = (3, 3)
# sc.pl.umap(adata, color="louvain", legend_loc='on data', s=10)
# obs_df = adata.obs.dropna()
# ARI = adjusted_rand_score(obs_df['louvain'], obs_df['Ground Truth'])
# print('Adjusted rand index = %.2f' %ARI)
# plt.rcParams["figure.figsize"] = (3, 3)
# sc.pl.umap(adata, color=["louvain", "Ground Truth"], title=['new (ARI=%.2f)'%ARI, "Ground Truth"])
#
# #设置超参数p社区贡献的总表达百分比 l:控制p的参数
# p=0.5
# #Find the l value given p
# l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
#
# n_clusters=7
# #Set seed
# r_seed=t_seed=n_seed=100
# #Seaech for suitable resolution
# '''这段Python代码是一个名为`search_res`的函数，它接受以下参数：
# - `l`：一个整数，表示拉普拉斯矩阵的类型。
# - `target_num`：一个整数，表示目标聚类数。
# - `start`：一个浮点数，表示搜索的起始分辨率。
# - `step`：一个浮点数，表示搜索的步长。
# - `tol`：一个浮点数，表示收敛容差。
# - `lr`：一个浮点数，表示学习率。
# - `max_epochs`：一个整数，表示最大迭代次数。
# 该函数使用了SpaGCN算法，通过调整分辨率来搜索最佳聚类数。它返回一个浮点数，表示推荐的分辨率。在函数中，它使用了随机数生成器来确保结果的可重复性，并使用了PyTorch和NumPy库来进行计算
# '''
# res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
# clf=spg.SpaGCN()
# clf.set_l(l)
# #Set seed
# random.seed(r_seed)
# torch.manual_seed(t_seed)
# np.random.seed(n_seed)
# #Run
# clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
# y_pred, prob=clf.predict()
# adata.obs["pred"]= y_pred
# adata.obs["pred"]=adata.obs["pred"].astype('category')
# #Do cluster refinement(optional)
# #shape="hexagon" for Visium data, "square" for ST data.
# # adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
# # refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
# # adata.obs["refined_pred"]=refined_pred
# # adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
# #Save results
# adata.write_h5ad("../tutorial/sample_results/results.h5ad")
# adata=sc.read("../tutorial/sample_results/results.h5ad")
# #Set colors used
# plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
# #Plot spatial domains
# domains="pred"
# num_celltype=len(adata.obs[domains].unique())
# adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
# ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=3.0)
# ax.set_aspect('equal', 'box')
# ax.axes.invert_yaxis()
# plt.savefig("../tutorial/sample_results/pred.png", dpi=600)
# plt.close()
# sc.tl.umap(adata)
# sc.tl.louvain(adata, resolution=1)
# obs_df = adata.obs.dropna()
# ARI = adjusted_rand_score(obs_df['louvain'], obs_df['Ground Truth'])
# print('Adjusted rand index = %.2f' %ARI)
# plt.rcParams["figure.figsize"] = (3, 3)
# sc.pl.umap(adata, color=["louvain", "Ground Truth"], title=['new (ARI=%.2f)'%ARI, "Ground Truth"])
#
# #Plot refined spatial domains
# domains="refined_pred"
# num_celltype=len(adata.obs[domains].unique())
# adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
# ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
# ax.set_aspect('equal', 'box')
# ax.axes.invert_yaxis()
# plt.savefig("../tutorial/sample_results/refined_pred.png", dpi=600)
# plt.close()
#
#Read in raw data
# raw=sc.read("../tutorial/data/151673/sample_data.h5ad")
# raw.var_names_make_unique()
# raw.obs["pred"]=adata.obs["pred"].astype('category')
# raw.obs["x_array"]=raw.obs["x2"]
# raw.obs["y_array"]=raw.obs["x3"]
# raw.obs["x_pixel"]=raw.obs["x4"]
# raw.obs["y_pixel"]=raw.obs["x5"]
# #Convert sparse matrix to non-sparse
# raw.X=(raw.X.A if issparse(raw.X) else raw.X)
# raw.raw=raw
# sc.pp.log1p(raw)
#
# #Use domain 0 as an example
# target=0
# #Set filtering criterials
# min_in_group_fraction=0.8
# min_in_out_group_ratio=1
# min_fold_change=1.5
# #Search radius such that each spot in the target domain has approximately 10 neighbors on average
# adj_2d=spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
# start, end= np.quantile(adj_2d[adj_2d!=0],q=0.001), np.quantile(adj_2d[adj_2d!=0],q=0.1)#表示adj_2d中所有非零元素的第0.1%和第99.9%分位数
# r=spg.search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(), x=x_array, y=y_array, pred=adata.obs["pred"].tolist(), start=start, end=end, num_min=10, num_max=14,  max_run=100)
# #Detect neighboring domains
# nbr_domians=spg.find_neighbor_clusters(target_cluster=target,
#                                    cell_id=raw.obs.index.tolist(),
#                                    x=raw.obs["x_array"].tolist(),
#                                    y=raw.obs["y_array"].tolist(),
#                                    pred=raw.obs["pred"].tolist(),
#                                    radius=r,
#                                    ratio=1/2)
#
# nbr_domians=nbr_domians[0:3]
# de_genes_info=spg.rank_genes_groups(input_adata=raw,
#                                 target_cluster=target,
#                                 nbr_list=nbr_domians,
#                                 label_col="pred",
#                                 adj_nbr=True,
#                                 log=True)
# #Filter genes
# de_genes_info=de_genes_info[(de_genes_info["pvals_adj"]<0.05)]
# filtered_info=de_genes_info
# filtered_info=filtered_info[(filtered_info["pvals_adj"]<0.05) &
#                             (filtered_info["in_out_group_ratio"]>min_in_out_group_ratio) &
#                             (filtered_info["in_group_fraction"]>min_in_group_fraction) &
#                             (filtered_info["fold_change"]>min_fold_change)]
# filtered_info=filtered_info.sort_values(by="in_group_fraction", ascending=False)
# filtered_info["target_dmain"]=target
# filtered_info["neighbors"]=str(nbr_domians)
# print("SVGs for domain ", str(target),":", filtered_info["genes"].tolist())
#
# print(filtered_info)
# #Plot refinedspatial domains
# color_self = clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370',"#EAE7CC","#FD1593"], N=256)
# for g in filtered_info["genes"].tolist():
#     raw.obs["exp"]=raw.X[:,raw.var.index==g]
#     ax=sc.pl.scatter(raw,alpha=1,x="y_pixel",y="x_pixel",color="exp",title=g,color_map=color_self,show=False,size=100000/raw.shape[0])
#     ax.set_aspect('equal', 'box')
#     ax.axes.invert_yaxis()
#     plt.savefig("./sample_results/"+g+".png", dpi=600)
#     plt.close()
#
#
# #Use domain 2 as an example
# target=2
# meta_name, meta_exp=spg.find_meta_gene(input_adata=raw,
#                     pred=raw.obs["pred"].tolist(),
#                     target_domain=target,
#                     start_gene="GFAP",
#                     mean_diff=0,
#                     early_stop=True,
#                     max_iter=3,
#                     use_raw=False)
#
# raw.obs["meta"]=meta_exp
#
# #Plot meta gene
# g="GFAP"
# raw.obs["exp"]=raw.X[:,raw.var.index==g]
# ax=sc.pl.scatter(raw,alpha=1,x="y_pixel",y="x_pixel",color="exp",title=g,color_map=color_self,show=False,size=100000/raw.shape[0])
# ax.set_aspect('equal', 'box')
# ax.axes.invert_yaxis()
# plt.savefig("E:/PythonCode/SpaGCN-master/tutorial/sample_results/"+g+".png", dpi=600)
# plt.close()
#
# raw.obs["exp"]=raw.obs["meta"]
# ax=sc.pl.scatter(raw,alpha=1,x="y_pixel",y="x_pixel",color="exp",title=meta_name,color_map=color_self,show=False,size=100000/raw.shape[0])
# ax.set_aspect('equal', 'box')
# ax.axes.invert_yaxis()
# plt.savefig("E:/PythonCode/SpaGCN-master/tutorial/sample_results/meta_gene.png", dpi=600)
# plt.close()