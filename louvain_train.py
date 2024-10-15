import matplotlib as mpl
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams
import os
config={
    "font.family":'serif',
    "font.size":12,
    "font.serif":['SimSun'],
    "mathtext.fontset":'stix',
    'axes.unicode_minus':False
}
rcParams.update(config)

folder_path = "F:/SpatialLIBD/data"
save_path = "F:/SpatialLIBD/tutorial/data/"
results_path = "F:/SpatialLIBD/tutorial/results"
file_names = os.listdir(folder_path)
for file_name in file_names:
    input_dir = os.path.join(folder_path, file_name)
    adata = sc.read_visium(path=input_dir, count_file=file_name + '_filtered_feature_bc_matrix.h5')
    detail_path = os.path.join(folder_path, file_name)
    adata=sc.read(save_path + file_name +"/sample_data.h5ad")
    print(file_name+":已读入")

    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # read the annotation
    Ann_df = pd.read_csv(os.path.join(folder_path, file_name, file_name+'_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    # sc.tl.louvain(adata, key_added="clusters")
    sc.tl.louvain(adata, resolution=0.8113)

    plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.umap(adata, color=['louvain'], title=['Louvain'], show=False)
    # plt.savefig('F:/results/Louvain_umap.tif',dpi=300,bbox_inches='tight')
    sc.pl.spatial(adata, img_key="hires", color=['louvain'], title=['Louvain'], show=True)
    # adata.obs['louvain'] = adata.obs['louvain'].replace('0', '7')
    print(adata)
    print(adata.obs['louvain'])
    print(adata.obs['Ground Truth'])
    # adata.obs['louvain']=adata.obs['louvain'].reset_index()
    from sklearn import metrics

    # ARI = metrics.adjusted_rand_score(adata.obs['louvain'], adata.obs['Ground Truth'])
    ARI = metrics.adjusted_rand_score(adata.obs['louvain'], adata.obs['Ground Truth'])
    NMI = metrics.normalized_mutual_info_score(adata.obs['louvain'], adata.obs['Ground Truth'])
    AMI = metrics.adjusted_mutual_info_score(adata.obs['louvain'], adata.obs['Ground Truth'])
    adata.uns['ARI'] = ARI
    adata.uns['NMI'] = NMI
    adata.uns['AMI'] = AMI

    print('Dataset:', file_name)
    print('ARI:', ARI)
    print('NMI:', NMI)
    print('AMI:', AMI)
    results_txt = 'F:/SpatialLIBD/tutorial/results/louvain_clust_txt'
    with open(results_txt, "a") as f:
        f.write("\n")
        f.write(file_name +"ARI:"+ str(ARI) + "\n")
        f.write(file_name + "NMI:" + str(NMI) + "\n")
        f.write(file_name + "AMI:" + str(AMI) + "\n")
