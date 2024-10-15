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

from scanpy import read_10x_h5
#read_10x_h5返回的是带注释的数据矩阵，基因表达矩阵x obs_names细胞名称 var_names基因名称 var [基因ID 特征类型]
# adata = read_10x_h5("../tutorial/data/151508/sample_data.h5")

# section_id = '151508'
# input_dir = os.path.join('F:/SpatialLIBD', section_id)
# adata = sc.read_visium(path=input_dir, count_file=section_id+'_filtered_feature_bc_matrix.h5')
folder_path = "F:/SpatialLIBD/data"
save_path = "F:/SpatialLIBD/tutorial/data/"
results_path = "F:/SpatialLIBD/tutorial/results"
file_names = os.listdir(folder_path)
for file_name in file_names:
    input_dir = os.path.join(folder_path, file_name)
    adata = sc.read_visium(path=input_dir, count_file=file_name + '_filtered_feature_bc_matrix.h5')
    detail_path = os.path.join(folder_path, file_name)
    #读取空间坐标的位置
    spatial=pd.read_csv(detail_path + '/spatial/tissue_positions_list.csv',sep=",",header=None,na_filter=False,index_col=0)
    # "F:/SpatialLIBD/151508/spatial/tissue_positions_list.csv"

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
    adata.write_h5ad(save_path + file_name +"/sample_data.h5ad")
    adata=sc.read(save_path + file_name +"/sample_data.h5ad")
    print(file_name+":已读入")


    #Read in hitology image
    img=cv2.imread(detail_path +"/histology.tif")
    x_array=adata.obs["x_array"].tolist()
    y_array=adata.obs["y_array"].tolist()
    x_pixel=adata.obs["x_pixel"].tolist()
    y_pixel=adata.obs["y_pixel"].tolist()
    print(file_name+":组织信息读入")

    #Test coordinates on the image
    #每个坐标周围20个像素设置为黑色
    img_new=img.copy()
    for i in range(len(x_pixel)):
        x=x_pixel[i]
        y=y_pixel[i]
        img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0

    cv2.imwrite(results_path+"/"+file_name +'/'+file_name+'_map.jpg', img_new)
    s=1
    b=49
    #计算图像的邻接矩阵
    adj=spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=True)
    print(file_name , adj.shape)
    #If histlogy image is not available, spatrain can calculate the adjacent matrix using the fnction below
    #adj=calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
    np.savetxt(save_path + file_name +'/adj.csv', adj, delimiter=',')
    # adata=sc.read("../tutorial/data/151508/sample_data.h5ad")
    adj=np.loadtxt(save_path + file_name +'/adj.csv', delimiter=',')
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
    # print(adata)
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(folder_path, file_name, file_name+'_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"])
    tf.compat.v1.disable_eager_execution()
    newGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    newGATE.Stats_Spatial_Net(adata)
    adata = newGATE.train_STAGATE(adata, alpha=0.5, pre_resolution=0.6,
                                  n_epochs=1000, save_attention=True)
    # print(adata)
    # pre-clustering result
    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.spatial(adata, img_key="hires", color="expression_louvain_label", size=1.0, title='pre-clustering result')
    # sc.pp.neighbors(adata, use_rep='STAGATE')



    # Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GraphST.GraphST(adata, device=device)
    # train model
    adata = model.train()
    # set radius to specify the number of neighbors considered during refinement
    radius = 50

    tool = 'mclust' # mclust, leiden, and louvain
    # the number of clusters
    char_list = ['151669', '151670', '151671', '151672']
    if file_name in char_list:
        n_clusters = 5
    else:
        n_clusters = 7
    # dataset = '151508'
    # read data
    # please replace 'file_fold' with the download path
    # file_fold = 'F:/SpatialLIBD/' + str(dataset)
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
    # adata
    # calculate metric ARI
    # ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['Ground Truth'])
    # adata.uns['ARI'] = ARI
    #
    # print('Dataset:', file_name)
    # print('ARI:', ARI)
    # calculate metric ARI
    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['Ground Truth'])
    NMI = metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['Ground Truth'])
    AMI = metrics.adjusted_mutual_info_score(adata.obs['domain'], adata.obs['Ground Truth'])
    adata.uns['ARI'] = ARI
    adata.uns['NMI'] = NMI
    adata.uns['AMI'] = AMI

    print('Dataset:', file_name)
    print('ARI:', ARI)
    print('NMI:', NMI)
    print('AMI:', AMI)

    # plotting spatial clustering result
    ax = sc.pl.spatial(adata,
                  img_key="hires",
                  color=["Ground Truth", "domain"],
                  title=[file_name + "Ground truth", "ARI=%.4f"%ARI],
                  show=False)

    # ax.set_aspect('equal', 'box')
    # ax.axes.invert_yaxis()
    # plt.savefig(results_path + '/' + file_name + "/" + file_name + "_clust.png", dpi=600)
    '''plt.savefig(results_path + '/' + file_name + "/" + file_name + "_clust.png", dpi=600)
    plt.show()'''
    results_txt = 'F:/SpatialLIBD/tutorial/results/results_txt'
    with open(results_txt, "a") as f:
        f.write("\n")
        f.write(file_name +"ARI:"+ str(ARI) + "\n")
        f.write(file_name + "NMI:" + str(NMI) + "\n")
        f.write(file_name + "AMI:" + str(AMI) + "\n")


    # ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=3.0)
    # ax.set_aspect('equal', 'box')
    # ax.axes.invert_yaxis()
    # plt.savefig("../tutorial/sample_results/pred.png", dpi=600)
    # plt.close()
    # plotting predicted labels by UMAP
    # sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=10)
    # sc.tl.umap(adata)
    # sc.pl.umap(adata, color='domain', title=['Predicted labels'], show=True)
    # sc.tl.draw_graph(adata)
    # sc.tl.paga(adata, groups='mclust')
    # print(adata.var_names)
    # sc.pl.paga(adata, color=['mclust', 'NOC2L', 'SAMD11', 'PRMT2'])
    '''sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=10)
    sc.tl.umap(adata)
    # sc.pl.umap(adata, color='domain', title=['Predicted labels'], show=True)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "Ground Truth"], title=['new(ARI=%.2f)' % ARI, "Ground Truth"], show=True)
    sc.tl.draw_graph(adata)
    sc.tl.paga(adata, groups='mclust')
    print(adata.var_names)
    sc.pl.paga(adata, color=['mclust'])'''