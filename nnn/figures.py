"""
For specific figures
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json
import seaborn as sns
from scipy.stats import chi2, pearsonr, norm
from sklearn.metrics import r2_score
from ipynb.draw import draw_struct

from .util import *
from . import util, plotting

sns.set_style('ticks')
sns.set_context('paper')

#############################
######### Figure 1 ##########
#############################

#--------- Raw image ----------#

def get_tile_filename(imagedir, condition, tile):
    found_filename = [fn for fn in os.listdir(os.path.join(imagedir, condition)) if '_tile%d_'%tile in fn]
    assert len(found_filename) == 1, print(found_filename)
    return os.path.join(imagedir, condition, found_filename[0])

def get_cropped_tif(filename, crop_tuple=(1024,1024,256,256)):
    x0, y0, dx, dy = crop_tuple
    im = np.array(Image.open(filename))[x0:x0+dx, y0:y0+dy]
    return im

def normalize_image(img, norm_max_min, celsius):
    fmax, fmin = norm_max_min.loc[celsius, :]
    return (img - fmin) / (fmax - fmin)

def plot_fig1_raw_images(figname, normalize=True):
    datadir = '/scratch/groups/wjg/kyx/NNNlib2b_Nov11/data/'
    experiment = '20220314'
    imagedir = os.path.join(datadir, 'images_' + experiment)
    condition = ['Green10_20', 'Green14_30', 'Green18_40','Green22_50', 'Green26_60']
    T = [20, 30, 40, 50, 60]
    tile = 9
    
    norm_max_min = pd.read_table(os.path.join(datadir, 'series_normalized', 'NNNlib2b_DNA_%s_norm_max_min.tsv'%experiment))
    xdata = pd.read_table(os.path.join(datadir, 'series_normalized', 'NNNlib2b_DNA_%s_xdata.txt'%experiment), header=None) - 273.15
    norm_max_min.index = np.array(xdata).squeeze()
    
    fig, ax = plt.subplots(1,5,figsize=(15,3))
    for i in range(5):
        im = im = get_cropped_tif(filenames[i], (512, 1024, 256,256))
        
        if normalize:
            norm_im = normalize_image(im, norm_max_min, T[i])
            ax[i].imshow(norm_im, cmap='gray', vmin=300, vmax=2200)
        else:
            ax[i].imshow(im, cmap='gray', vmin=0, vmax=500)
            
        ax[i].axis('off')
        ax[i].set_title('%d °C'%T[i])
        
    util.save_fig(figname)
    

#############################
######### Figure 2 ##########
#############################

def plot_fig2_comparison_by_series(vf, param, suffix = '_NUPACK_salt_corrected',
    annotation=None, lim=None):

    df = vf.copy()
    if annotation is not None:
        df = df.join(annotation)
        
    df.loc[df.Series == 'External', 'Series'] = 'Control'
    df.loc[df.Series == 'TRIloop', 'Series'] = 'Hairpin Loops'
    df.loc[df.Series == 'TETRAloop', 'Series'] = 'Hairpin Loops'

    series = df.query('Series != "Control"').groupby('Series').apply(len).sort_values(ascending=True)
    l = np.abs(lim[1] - lim[0])

    fig, ax = plt.subplots(2,2,figsize=(10,10), sharex=False, sharey=False)
    ax = ax.flatten()

    for i, s in enumerate(series.index[:4]):
        series_df = df.query('Series == "%s"'%s)
        print('Series %s,  %d variants' % (s, len(series_df)))
        ax[i].plot(lim, lim, '--', c='gray', zorder=0)
        if len(series_df) > 100:
            plotting.plot_colored_scatter_comparison(data=series_df, x=param+suffix, y=param,
                lim=lim, palette=plotting.generate_color_palette(i), ax=ax[i])
            pearson, _ = pearsonr(series_df[param+suffix], series_df[param])
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.15*l, r"$Pearson's\ r = %.3f$"%pearson, va='bottom')
        else:
            sns.scatterplot(data=series_df, x=param+suffix, y=param,
                color=palette[i % len(palette)], ax=ax[i])

        ax[i].set_xlim(lim)
        ax[i].set_ylim(lim)    
        ax[i].set_xlabel('NUPACK $dG_{37}$ (kcal/mol)')
        ax[i].set_ylabel('MANifold $dG_{37}$ (kcal/mol)')
        ax[i].set_title('%s, N=%d'%(s, series[s]))

    plt.suptitle(param)

    return fig, ax


def plot_fig2_nupack_distance(vf, param='dG_37', suffix='_NUPACK_salt_corrected'):
    l = 4
    df = vf.copy()
    df['s.e. of $dG_{37}$'] = pd.qcut(df[param+'_se'], q=4)
    df['ddG[NUPACK, MANIfold]'] = df[param+suffix] - df[param]
        
    df.loc[df.Series == 'External', 'Series'] = 'Control'
    df.loc[df.Series == 'TRIloop', 'Series'] = 'Hairpin Loops'
    df.loc[df.Series == 'TETRAloop', 'Series'] = 'Hairpin Loops'
    series = df.query('Series != "Control"').groupby('Series').apply(len).sort_values(ascending=True)

    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=True)
    ax = ax.flatten()
    bins = np.arange(-l, l, 0.02)

    for i, s in enumerate(series.index[:4]):
        sigma = 0.33 # empiracal, std of ddG between replicates
        norm_pdf = norm.pdf(bins, 0, sigma)
        ax[i].fill_between(bins, norm_pdf, 0, where=(norm_pdf>=0), interpolate=True, color='gray')
        sns.kdeplot(data=df.query(f'Series == "{s}"'), x='ddG[NUPACK, MANIfold]', hue='s.e. of $dG_{37}$', 
            common_norm=False, linewidth=4, palette='magma', ax=ax[i])
        ax[i].set_xlabel('ddG[NUPACK, MANIfold]' + ' (kcal/mol)')
        ax[i].set_title(s)
        ax[i].set_xlim([-l, l])
        if i != 1:
            ax[i].get_legend().remove()


def plot_fig2_nupack_distance_ridge(vf, param='dG_37', suffix='_NUPACK_salt_corrected'):
    df = vf.copy()
    df['$dG_{37}$ bin'] = pd.qcut(df[param+'_se'], q=4)
    df['ddG[NUPACK, MANIfold] (kcal/mol)'] = df[param+suffix] - df[param]
    l = 3
    bins = np.arange(-l, l, 0.1)
    norm_pdf = norm.pdf(bins, 0, 1)

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row='$dG_{37}$ bin', hue='$dG_{37}$ bin',
        aspect=3, height=1, palette='magma', xlim=[-l,l])

    # Draw the densities in a few steps
    # g.map(plt.fill_between, bins, norm_pdf, 0, interpolate=True, color='gray')
    g.map(sns.kdeplot, 'ddG[NUPACK, MANIfold] (kcal/mol)',
        bw_adjust=.5, clip_on=False,
        fill=True, alpha=1, linewidth=1.5)
    # g.map(sns.kdeplot, 'ddG[NUPACK, MANIfold]', clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    # g.map(label, 'ddG[NUPACK, MANIfold] (kcal/mol)')

    # Set the subplots to overlap
    # g.figure.subplots_adjust(hspace=-.1)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

def plot_fig2_nupack_zscore(vf, param='dG_37', suffix='_NUPACK_salt_corrected'):
    l = 4
    df = vf.copy()
    # df['s.e. of $dG_{37}$'] = pd.qcut(df[param+'_se'], q=4)
        
    df.loc[df.Series == 'External', 'Series'] = 'Control'
    df.loc[df.Series == 'TRIloop', 'Series'] = 'Hairpin Loops'
    df.loc[df.Series == 'TETRAloop', 'Series'] = 'Hairpin Loops'
    series = df.query('Series != "Control"').groupby('Series').apply(len).sort_values(ascending=True)
    
    df['ddG_NUPACK'] = df[param+suffix] - df[param]
    std_df = df[['Series', 'ddG_NUPACK']].groupby('Series').apply(np.std)
    std_df.columns = ['ddG_NUPACK_std']
    mean_df = df[['Series', 'ddG_NUPACK']].groupby('Series').apply(np.mean)
    mean_df.columns = ['ddG_NUPACK_mean']
    df =  df.join(std_df, on='Series').join(mean_df, on='Series')
    df['zscore'] = (df['ddG_NUPACK'] - df['ddG_NUPACK_mean']) / df['ddG_NUPACK_std']

    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=True)
    ax = ax.flatten()
    bins = np.arange(-l, l, 0.02)

    for i, s in enumerate(series.index[:4]):
        norm_pdf = norm.pdf(bins, 0, 1)
        ax[i].fill_between(bins, norm_pdf, 0, where=(norm_pdf>=0), interpolate=True, color='gray')
        sns.kdeplot(data=df.query(f'Series == "{s}"'), x='zscore',
            linewidth=4, color=cc.glasbey_light[i], ax=ax[i])
        ax[i].set_xlabel('$ddG_{37}$ [NUPACK, MANIfold]' + ' (kcal/mol)')
        ax[i].set_title(s)
        ax[i].set_xlim([-l, l])


#############################
######### Figure 3 ##########
#############################

def plot_triloop_linear_model():
    pass