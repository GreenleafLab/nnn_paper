import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import seaborn as sns
import colorcet as cc
import json, os, pickle
from collections import defaultdict
from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder
from ipynb.draw import draw_struct
import nupack
import sklearn
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from tqdm import tqdm
import itertools

from nnn import util, fileio, mupack, plotting, processing, uv, variance, modeling
import nnn.motif_fit as mf
from nnn.arraydata import ArrayData

# suppress warnings 
import warnings
warnings.filterwarnings("ignore")

# plotting settings
palette = cc.glasbey_dark
cm = 1/2.54
px = 1/plt.rcParams['figure.dpi']
tick_font_size = 5
label_font_size = 6

sns.set_style('ticks')
sns.set_context('paper')
matplotlib.rc('axes',edgecolor='k')

# make sure the text is editable in illustrator
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# set font to arial
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

# some constants
kB = 0.0019872 # Bolzman constant
C2T = 273.15 # conversion from celsius to kalvin
bp_dict = dict(A='T', C='G', T='A', G='C')