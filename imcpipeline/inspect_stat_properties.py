#! /usr/bin/env python

import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.ndimage as ndi
import skimage
import skimage.measure

import tifffile
import imctools
import imctools.io.mcdparser
from anndata import AnnData
import scanpy as sc

from imcpipeline.utils import get_grid_dims, metal_order_to_channel_labels, cleanup_channel_names

matplotlib.rcParams['svg.fonttype'] = "none"


fkwargs = dict(dpi=100, bbox_inches="tight")

sample_name = '20191015'

output_prefix = pjoin("processed", "stats", sample_name)
os.makedirs(pjoin("processed", "stats"), exist_ok=True)


mcd_file = pjoin('data', sample_name, '20191015_I19_1801_A1', '20191015_I19_1801_A1.mcd')
mcd = imctools.io.mcdparser.McdParser(mcd_file)

segmentation_file = pjoin('processed', sample_name, 'tiffs', '20191015_I19_1801_A1_s1_p3_r1_a1_ac_ilastik_s2_Probabilities_mask.tiff')
segmentation = tifffile.imread(segmentation_file)

ac = mcd.get_imc_acquisition('1')

arr = ac.data

labels = cleanup_channel_names(pd.Series(ac.channel_labels))

m, n = get_grid_dims(len(labels))

# Visualize images
fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n), gridspec_kw=dict(wspace=0, hspace=0))
axes = axes.ravel()
for i, label in enumerate(labels):
    a = ac.get_img_by_label(label)
    axes[i].imshow(np.log1p(a), rasterized=True)
    axes[i].set_title(labels[i])
    axes[i].axis("off")
for ax in axes[i + 1:]:
    ax.axis("off")
fig.savefig(output_prefix + ".all_channels.svg", **fkwargs)

# since the acquisition seems to have stopped abruptly, I'm going to mask
# pixels in the last row which are 0 in the pos control channel
# otherwise this could be seen as a kind of zero-inflation
background = ac.get_img_by_label(labels[0])


# Visualize distributions
fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n))
axes = axes.ravel()
for i, label in enumerate(labels):
    a = ac.get_img_by_label(label)
    a = np.ma.array(a, mask=background <= 0)  # mask from background
    # sns.distplot(a.ravel(), kde=False, ax=axes[i])
    sns.distplot(a.ravel().astype(int), kde=False, ax=axes[i])
    axes[i].set_yscale("log")
    axes[i].set_title(labels[i])
for ax in axes[i + 1:]:
    ax.axis("off")
fig.savefig(output_prefix + '.channel_dists_per_pixel.histogram.svg', **fkwargs)


# # Try to fit poisson, NB, etc


# Quantify cells keeping count data by summing up
cells = np.unique(segmentation)
res = np.zeros((len(cells) - 1, arr.shape[0]), dtype=int)  # result = shape(cells, channels)
for channel in range(arr.shape[0]):
    res[:, channel] = [
        x.intensity_image.sum()
        for x in skimage.measure.regionprops(segmentation, arr[channel])]
x = pd.DataFrame(res[:, 3:], columns=labels)
x = x.drop(['', '80ArAr', '124Xe', '127I'], axis=1)
x = x.reindex(sorted(x.columns[x.columns != '190BCKG'].tolist()) + ['190BCKG'], axis=1)
x.columns = cleanup_channel_names(x.columns)
# x = x.loc[x.sum(1) > 0]
# x = x.drop(0, axis=0)
# background = np.where(segmentation == 0, arr, 0).mean(axis=(1, 2))
x.to_csv(output_prefix + ".quantification_sum.csv.gz")


# Do the same with tiff to compare the "lossiness"
tiff_file = pjoin('processed', sample_name, 'tiffs', '20191015_I19_1801_A1_s1_p3_r1_a1_ac_full.tiff')
tiff = tifffile.imread(tiff_file)

metal_csv = pjoin('processed', sample_name, 'tiffs', '20191015_I19_1801_A1_s1_p3_r1_a1_ac_full.csv')
channel_metadata = pjoin('processed', sample_name, 'ometiff', '20191015_I19_1801_A1', '20191015_I19_1801_A1_AcquisitionChannel_meta.csv')
roi_number = 1
tiff_labels = metal_order_to_channel_labels(metal_csv, channel_metadata, roi_number)
tiff_labels = cleanup_channel_names(tiff_labels.str.extract(r"(.*)\(").squeeze())

tiff_res = np.zeros((len(cells) - 1, tiff.shape[0]), dtype=int)  # result = shape(cells, channels)
for channel in range(tiff.shape[0]):
    tiff_res[:, channel] = [
        x.intensity_image.mean()
        for x in skimage.measure.regionprops(segmentation, tiff[channel])]
tiff_x = pd.DataFrame(tiff_res, columns=tiff_labels)
tiff_x = tiff_x.sort_index(axis=1)
tiff_x.to_csv(output_prefix + ".quantification_mean.csv.gz")



# # # additional measurements
# n_measurements = 6
# cells = np.unique(segmentation)
# res2 = np.zeros((len(cells) - 1, n_measurements, arr.shape[0]), dtype=float)  # result = shape(cells, channels)
# for channel in range(arr.shape[0]):
#     for j, x in enumerate(skimage.measure.regionprops(segmentation, arr[channel])):
#         res2[j, :, channel] = x.equivalent_diameter, x.perimeter, x.area, x.convex_area, x.eccentricity, x.euler_number
# # # # reduce across channels somhow
# shape_form = pd.DataFrame(
#     res2.mean(axis=2),
#     columns=['diamater', 'perimeter', 'area', 'convex_area', 'eccentricity', 'euler_number'])



fig, axes = plt.subplots(n, m, figsize=(3 * m, 3 * n))
axes = axes.ravel()
for i, label in enumerate(x.columns):
    # sns.distplot(a.ravel(), kde=False, ax=axes[i])
    sns.distplot(x[label].astype(int), kde=False, ax=axes[i])
    axes[i].set_yscale("log")
    axes[i].set_title(labels[i])
for ax in axes[i + 1:]:
    ax.axis("off")
fig.savefig(output_prefix + '.channel_dists_per_cell.histogram.svg', **fkwargs)


# Quickly check channel correlation
grid = sns.clustermap(x.corr(), xticklabels=True, yticklabels=True, cbar_kws=dict(label="Pearson correlation"))
grid.savefig(output_prefix + '.cell.sum.channel_correlation.clustermap.svg', **fkwargs)

grid = sns.clustermap(tiff_x.corr(), xticklabels=True, yticklabels=True, cbar_kws=dict(label="Pearson correlation"))
grid.savefig(output_prefix + '.cell.mean.channel_correlation.clustermap.svg', **fkwargs)


ckws = dict(rasterized=True, yticklabels=False, metric="correlation")
grid = sns.clustermap(np.log1p(x.iloc[:5000, :]), cbar_kws=dict(label="Signal sum"), **ckws)
grid.savefig(output_prefix + ".cell.log_sum.clustermap.top5000cells.svg", **fkwargs)
grid = sns.clustermap(np.log1p(tiff_x.iloc[:5000, :]), cbar_kws=dict(label="Signal mean"), **ckws)
grid.savefig(output_prefix + ".cell.log_mean.clustermap.top5000cells.svg", **fkwargs)


ckws = dict(rasterized=True, yticklabels=False, metric="correlation", z_score=1, cmap="RdBu_r", center=0, robust=True)
grid = sns.clustermap(np.log1p(x.iloc[:5000, :]), cbar_kws=dict(label="Signal sum (Z-score)"), **ckws)
grid.savefig(output_prefix + ".cell.log_sum.clustermap.top5000cells.zscore.svg", **fkwargs)
grid = sns.clustermap(np.log1p(tiff_x.iloc[:5000, :]), cbar_kws=dict(label="Signal mean (Z-score)"), **ckws)
grid.savefig(output_prefix + ".cell.log_mean.clustermap.top5000cells.zscore.svg", **fkwargs)


ans = dict()
for label, df in [("sum", x), ("mean", tiff_x)]:
    # Start usual single cell analysis
    a = AnnData(df)
    a.raw = a
    ans[label] = a

    a.obs['n_counts'] = a.X.sum(axis=1).astype(int)
    a.obs['log_counts'] = np.log10(a.obs['n_counts'])

    tech_variables = ['log_counts']

    spec_markers = [
        'CD20', 'BCL6', 'MUM1',
        'CD3', 'CD4', 'CD8a', 'FoxP3',
        'HLAClassI', 'ColTypeI', 'S100', 'Vimentin',
        'CD31',
        'CD10', 'CD11b', 'GranzymeB',
        'CD68', 'CD163',
        'H3K27me3', 'Ki67']

    for var_label, markers in [("all_vars", a.var.index.tolist()), ("red_markers", spec_markers)]:
        a = a[:, a.var.index.isin(markers)]

        # norm
        sc.pp.log1p(a)
        sc.pp.normalize_per_cell(a, counts_per_cell_after=1e4)
        sc.pp.scale(a)


        sc.pl.heatmap(a, a.var.index, use_raw=False, vmin=-1, vmax=1, show=False)
        plt.gca().figure.savefig(output_prefix + f".cell.{label}.norm_scaled.heatmap.svg", **fkwargs)


        # dim res
        sc.pp.pca(a)

        sc.pp.neighbors(a, n_neighbors=20, use_rep='X')
        sc.tl.umap(a)
        sc.tl.diffmap(a)

        sc.tl.leiden(a)


        def raster(fig):
            for ax in fig.axes:
                for c in ax.get_children():
                    if not isinstance(c, (matplotlib.text.Text, matplotlib.axis.XAxis, matplotlib.axis.YAxis)):
                        if not c.get_children():
                            c.set_rasterized(True)
                        for cc in c.get_children():
                            if not isinstance(c, (matplotlib.text.Text, matplotlib.axis.XAxis, matplotlib.axis.YAxis)):
                                cc.set_rasterized(True)


        raw = a.raw.copy()

        sc.pp.log1p(a.raw.X)

        # Plot
        kwargs = dict(color=tech_variables + ['leiden'] + a.var.index.tolist(), show=False, return_fig=True, use_raw=True)
        fig = sc.pl.pca(a, **kwargs)
        raster(fig)
        fig.savefig(output_prefix + f'.cell.{label}.{var_label}.pca.svg', **fkwargs)

        fig = sc.pl.umap(a, **kwargs)
        raster(fig)
        fig.savefig(output_prefix + f'.cell.{label}.{var_label}.umap.svg', **fkwargs)

        fig = sc.pl.diffmap(a, **kwargs)
        raster(fig)
        fig.savefig(output_prefix + f'.cell.{label}.{var_label}.diffmap.svg', **fkwargs)

        # Test
        sc.tl.rank_genes_groups(a, groupby='leiden', method='logreg')
        sc.pl.rank_genes_groups(a)
        sc.pl.rank_genes_groups_dotplot(a, n_genes=4)


        axs = sc.pl.rank_genes_groups_matrixplot(a, n_genes=1, standard_scale='var', cmap='Blues')


        tiff_x.index = tiff_x.index.astype(str)
        mean_expr = tiff_x.join(a.obs['leiden']).groupby('leiden').mean()
        grid = sns.clustermap(mean_expr.drop('32'), z_score=1, center=0, cmap="RdBu_r")

        grid = sns.clustermap(mean_expr.drop('32'), z_score=0, center=0, cmap="RdBu_r")


        # Fish out a cell cluster to visualize in image
        coi = a.obs.index[a.obs['leiden'] == '0'].astype(int)
