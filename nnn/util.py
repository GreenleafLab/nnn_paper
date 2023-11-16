"""
Functions that all other modules use.
    - save_fig
    - get_* for calculating things
    - format conversion functions
    - other handy lil functions
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json
import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')
#import colorcet as cc
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy.stats import chi2, pearsonr, norm
from scipy import optimize as opt
from sklearn.preprocessing import MaxAbsScaler

# from ipynb.draw import draw_struct
import nupack
from matplotlib.backends.backend_pdf import PdfPages
# from arnie.free_energy import free_energy
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

palette=[
    '#201615',
    '#4e4c4f',
    '#4c4e41',
    '#936d60',
    '#5f5556',
    '#537692',
    '#a3acb1']#cc.glasbey_dark

complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', '-':'-'}
N_A = 6.023 * 1e23
R = 0.00198717 # gas constant in kcal/mol not J
kB = 1.380649*1e-23
cm = 1/2.54

####################
##### Plotting #####
####################

def save_fig(filename, fig=None):

    figdir, _ = os.path.split(filename)
    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    if fig is None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        
def save_multi_image(filename, figs=None):
    pp = PdfPages(filename)
    if figs is None:
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
def set_size(w,h, ax=None):
    """ 
    Use to set ax size. will auto get current axis
    w, h: width, height in inches 
    """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    

def get_cycle(cmap, N=None, use_index="auto"):
    """
    Make catagorical cmap work with plt.plot()
    Usage:
        plt.rcParams["axes.prop_cycle"] = get_cycle("Dark2")
    """
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return cycler("color",colors)

def beutify_all_ax(ax, **kwargs):
    for a in ax:
        beutify(a, **kwargs)
        
def beutify(ax, x_locator=None, y_locator=None,
            force_same_xy=False, add_margin=0,
            shrink=False, do_not_resize=False):
    sns.despine()
    matplotlib.rc('axes',edgecolor='k', linewidth=.5)
    ax.tick_params(colors='k', width=.5)
    ax.set_clip_on(False)
    
    if force_same_xy:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        lim_min = min(xlim[0], ylim[0])
        lim_max = max(xlim[1], ylim[1])
        
        if add_margin > 0:
            lim_min -= (lim_max - lim_min) * add_margin
            lim_max += (lim_max - lim_min) * add_margin
            
        new_lim = (lim_min, lim_max)
        ax.set_xlim(new_lim)
        ax.set_ylim(new_lim)
        
    if x_locator is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_locator))
    
    if y_locator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_locator))
        
    if shrink:
        if not do_not_resize:
            ax.figure.set_size_inches(4.5*cm, 3.25*cm)
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.xaxis.label.set_size(5)
        ax.yaxis.label.set_size(5)

#################
#### General ####
#################

def absolute_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
            
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df
    
def rcompliment(seq):
    return "".join(complement.get(base, base) for base in reversed(seq))

def nrcompliment(seq):
    return "".join(complement.get(base, base) for base in seq)
   
def replace_at_index(s:str, newstring:str, idx:int, idy:int=np.nan):
    """
    Only replace at one location if idy is not given
    """
    if np.isnan(idy):
        return s[:idx] + newstring + s[idx + 1:]
    else:
        return s[:idx] + newstring + s[idy:]

def get_reordered_ind(clustergrid, pivot_mat):
    """
    Get the new indices from sns.clustermap
    Args:
        clustergrid - returned object of sns.clustermap
    """
    reordered_index = pivot_mat.index.values[clustergrid.dendrogram_row.reordered_ind]
    reordered_columns = pivot_mat.columns.values[clustergrid.dendrogram_col.reordered_ind]
    return reordered_index, reordered_columns
    
  
def get_index_isinlist(df, mylist):
    return df.reset_index(names='myindex').query('myindex in @mylist').set_index('myindex')

       
######################            
#### NNN specific ####
######################
    
def format_refseq(refseq):
    if isinstance(refseq, str) and '[' in refseq:
        return eval(refseq)
    else:
        return refseq
        
            
def convert_santalucia_motif_representation(motif):
    strand = motif.split('_')
    return(f'{strand[0]}_{strand[1][::-1]}')



def filter_variant_table(df, variant_filter):
    filtered_df = df.query(variant_filter)
    print('%.2f%% variants passed the filter %s' % (100 * len(filtered_df) / len(df), variant_filter))
    return filtered_df


def get_GC_content(seq):
    return 100 * np.sum([s in ['G','C'] for s in str(seq)]) / len(str(seq))


def get_Na_adjusted_Tm(Tm, dH, GC, Na=0.088, from_Na=1.0):
    # Tmadj = Tm + (-3.22*GC/100 + 6.39)*(np.log(Na/from_Na))
    Tmadj_inv = (1 / (Tm + 273.15) + (4.29 * GC/100 - 3.95) * 1e-5 * np.log(Na / from_Na)
        + 9.4 * 1e-6 * (np.log(Na)**2 - np.log(from_Na)**2))
    Tmadj = 1 / Tmadj_inv - 273.15
    
    return Tmadj

def get_dG(dH, Tm, celsius):
    return dH * (1 - (273.15 + celsius) / (273.15 + Tm))
    
def get_Tm(dH, dG, celsius=37):
    return (273.15 + celsius) / (1 - dG / dH) - 273.15

def get_dS(dH, Tm):
    return dH / (Tm + 273.15)
    
def get_dG_err(dH, dH_err, Tm, Tm_err, celsius):
    """
    Error propagation
    dG = dH - TdS
    dG_err = sqrt(dH_err^2 + (TdS)_err^2)
           = sqrt(dH_err^2 + (T*dS_err)^2)
    """
    dS_err = get_dS_err(dH, dH_err, Tm, Tm_err)
    return np.sqrt(dH_err**2 + ((celsius + 273.15) * dS_err)**2)
    
def get_dS_err(dH, dH_err, Tm, Tm_err):
    """
    Error propagation
    dS = dH / Tm
    dS_err = - dS * sqrt((dH_err / dH)^2 + (Tm_err / Tm)^2)
    """
    dS = get_dS(dH, Tm)
    return  - dS * np.sqrt((dH_err / dH)**2 + (Tm_err / (Tm + 273.15))**2)
    
def get_Na_adjusted_dG(Tm, dH, GC, celsius, Na=0.088, from_Na=1.0):
    Tm_adjusted = get_Na_adjusted_Tm(Tm, dH, GC, Na, from_Na)
    return get_dG(dH, Tm_adjusted, celsius)
    
def get_Na_adjusted_dG_37(Tm, dH, GC, Na=0.088, from_Na=1.0):
    Tm_adjusted = get_Na_adjusted_Tm(Tm, dH, GC, Na, from_Na)
    return get_dG(dH, Tm_adjusted, 37)

def get_Na_adjusted_param(Na=1.0, from_Na=0.088, **data_dict):
    """
    data_dict - dict, with keys dH, Tm, and seq
    """
    dH, Tm = data_dict['dH'], data_dict['Tm']
    GC_content = get_GC_content(data_dict['seq'])
    Tm_adjusted = get_Na_adjusted_Tm(Tm, dH, GC_content, Na, from_Na)
    dG_37_adjusted = get_dG(dH, Tm_adjusted, 37)
    return dict(dH=dH, dS=dH/Tm_adjusted, Tm=Tm_adjusted, dG_37=dG_37_adjusted)
    
def get_dof(row):
    n_T = len(np.arange(20, 62.5, 2.5))
    dof = row['n_clusters'] * n_T - 4 + row['enforce_fmax'] + row['enforce_fmin']
    return dof

def get_red_chisq(row):
    return row['chisq'] / row['dof']

def pass_chi2(row):
    cutoff = chi2.ppf(q=.99, df=row['dof'])
    return row['chisq'] < cutoff

def add_dG_37(df):
    df['dG_37'] = get_dG(df['dH'], df['Tm'], t=37)
    df['dG_37_ub'] = get_dG(df['dH_ub'], df['Tm_lb'], t=37)
    df['dG_37_lb'] = get_dG(df['dH_lb'], df['Tm_ub'], t=37)

def add_chisq_test(df):
    df['red_chisq'] = df.apply(get_red_chisq, axis=1)
    df['pass_chi2'] = df.apply(pass_chi2, axis=1)
    
    n_pass = sum(df['pass_chi2'])
    print('%d / %d, %.2f%% varaints passed the chi2 test' % (n_pass, len(df), 100 * n_pass / len(df)))
    

    return df

def add_p_unfolded_NUPACK(df, T_celcius, sodium=1.0):
    """
    Calculate p_unfolded at T_celcius and add as a column to df
    """
    def get_p_unfolded(row):
        Tm_NUPACK = get_Na_adjusted_Tm(row.Tm_NUPACK, row.dH_NUPACK, get_GC_content(row.RefSeq), Na=sodium)
        return 1 / (1 + np.exp(row.dH_NUPACK/0.00198*((Tm_NUPACK + 273.15)**-1 - (T_celcius+273.15)**-1)))

    df['p_unfolded_%dC'%T_celcius] = df.apply(get_p_unfolded, axis=1)

    return df
    
def dotbracket2edgelist(dotbracket_str:str):

    assert dotbracket_str.count('(') == dotbracket_str.count(')'), \
        'Number of "(" and ")" should match in %s' % dotbracket_str

    # Backbone edges
    N = len(dotbracket_str)
    edge_list = [[i, i+1] for i in range(N-1)]

    # Hydrogen bonds
    flag3p = N - 1
    for i,x in enumerate(dotbracket_str):
        if x == '(':
            for j in range(flag3p, i, -1):
                if dotbracket_str[j] == ')':
                    edge_list.append([i, j])
                    flag3p = j - 1
                    break

    return edge_list
    

def get_ddX(df, param='dG_37', by='ConstructType'):
    class_median = df.groupby(by).apply('median')[param]
    return df.apply(lambda row: row[param] - class_median[row[by]], axis=1)
    

def get_symmetric_struct(len_seq, len_loop):
    return '('*int((len_seq - len_loop)/2) + '.'*len_loop + ')'*int((len_seq - len_loop)/2)

def get_target_struct(row):

    def get_mismatch_struct(seq, mismatch_list):
        """
        Assumes all mismatches constructs are symmetric
        """
        target_struct_list = list(get_symmetric_struct(len(seq), 4))
        for i in range(int((len(seq)-4)/2)):
            if seq[i] + seq[-1-i] in mismatch_list:
                target_struct_list[i] = '.'
                target_struct_list[-1-i] = '.'
        target_struct = ''.join(target_struct_list)
        return target_struct

    series = row['Series']
    construct_type = row['ConstructType']
    seq = row['RefSeq']
    if series in ['WatsonCrick', 'TETRAloop'] or construct_type in ['SuperStem']:
        target_struct = get_symmetric_struct(len(seq), 4)
    elif series == 'TRIloop':
        target_struct = get_symmetric_struct(len(seq), 3)
    elif series == 'VARloop':
        target_struct = get_symmetric_struct(len(seq), len(seq) - 4)
    elif series in ['REPeatControls', 'PolyNTControls']:
        target_struct = '.'* len(seq)
    elif construct_type.startswith('StemDangle'):
        topScaffold = row['topScaffold']
        stem_pos = seq.find(topScaffold[:len(topScaffold)//2])
        target_struct = '.'*stem_pos + get_symmetric_struct(len(topScaffold) + 4, 4) + '.'*int(len(seq) - stem_pos - len(topScaffold) - 4)
    elif (series == 'MisMatches') or (construct_type == 'BaeControls'):
        mismatch_list = []
        for x in 'ATCG':
            for y in 'ATCG':
                if not x+y in ['AT', 'TA', 'CG', 'GC']:
                    mismatch_list.append(x+y)
        
        target_struct = get_mismatch_struct(seq, mismatch_list)
    else:
        # generated elsewhere
        target_struct = '.'*len(seq)

    return target_struct

def get_curve_pred(row, conds=None):
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*(Tm**-1 - x)))
    if conds is None:
        conds = [x for x in row.keys() if x.endswith('_norm')]
        
        errs = [x for x in row.keys() if x.endswith('_norm_std')]
    else:
        errs = [x+'_std' for x in conds]

    vals = np.array(row[conds].values,dtype=float) 
    errors = np.array(row[errs].values / np.sqrt(row['n_clusters']),dtype=float)

    T_celsius=[float(x.split('_')[1]) for x in conds]
    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    pred_fit = function(row['dH'],row['Tm'], row['fmax'], row['fmin'], T_inv)

    return pred_fit, vals, errors


def get_mfe_struct(seq, return_free_energy=False, verbose=False, celsius=0.0, sodium=1.0, param_set='dna04'):
    my_model = nupack.Model(material=param_set, celsius=celsius, sodium=sodium, magnesium=0.0)
    mfe_structures = nupack.mfe(strands=[seq], model=my_model)
    mfe_struct = str(mfe_structures[0].structure)

    if verbose:
        print('Number of mfe structures:', len(mfe_structures))
        print('Free energy of MFE proxy structure: %.2f kcal/mol' % mfe_structures[0].energy)
        print('MFE proxy structure in dot-parens-plus notation: %s' % mfe_structures[0].structure)

    if return_free_energy:
        return mfe_struct, mfe_structures[0].energy
    else:
        return mfe_struct


def get_seq_ensemble_dG(seq, celsius, sodium=1.0, verbose=False, param_set='dna04'):
    my_model = nupack.Model(material=param_set, celsius=celsius, sodium=sodium, magnesium=0.0)
    _, dG = nupack.pfunc(seq, model=my_model)
    return dG


def get_seq_structure_dG(seq, structure, celsius, sodium=1.0, param_set='dna04', **kwargs):
    """
    **kwargs passed to nupack.Model
    """
    my_model = nupack.Model(material=param_set, celsius=celsius, sodium=sodium, magnesium=0.0, **kwargs)
    if isinstance(seq, str):
        try:
            return nupack.structure_energy([seq], structure=structure, model=my_model)
        except:
            print(seq, structure, sodium)
    else:
        return nupack.structure_energy(seq, structure=structure, model=my_model)


def get_seq_end_pair_prob(seq:str, celsius:float, sodium=1.0, n_pair:int=2, param_set='dna04') -> float:
    """
    Pr[either last or second to last basepair in the hairpin paired]
    """
    my_model = nupack.Model(material=param_set, celsius=celsius, sodium=sodium, magnesium=0.0)
    pair_mat = nupack.pairs([seq], model=my_model).to_array()
    if n_pair == 1:
        return pair_mat[0,-1]
    elif n_pair == 2:
        try:
            p1, p2 = pair_mat[0,-1], pair_mat[1,-2]
        except:
            p1, p2 = np.nan, np.nan

        return p1 + p2 - p1 * p2
    else:
        raise ValueError("n_pair value not allowed")


def calculate_distance_2_equilibrium(seq, DNA_conc, model, target_struct=None, verbose=False):
    '''Calculate unpaired fraction either using structure free energies
    '''
    
    if isinstance(seq, str):
        A = nupack.Strand(seq, name='A')
    else:
        A = nupack.Strand(seq[0], name='A')
        B = nupack.Strand(seq[1], name='B')
    
    if target_struct is None:
        duplex = '(' * A.nt() + '+' + ')' * A.nt()
        # unpaired = '.' * A.nt() + '+' + '.' * A.nt()
    else:
        duplex = target_struct
        # unpaired = target_struct.replace('(','.').replace(')','.')

    # secondary structure free energies
    if isinstance(A, str):
        # energies = [nupack.structure_energy([A, ~A], s, model=model) for s in (unpaired, duplex)]
        energy = nupack.structure_energy([A, ~A], duplex, model=model)
    else:
        # energies = [nupack.structure_energy([A, B], s, model=model) for s in (unpaired, duplex)]
        energy = nupack.structure_energy([A, B], duplex, model=model)

    if verbose:
        print('dG =', energy)
    
    # get reaction quotient Q
    # factor = np.exp(-model.beta * (energies[1] - energies[0])) * DNA_conc / nupack.constants.water_molarity(model.temperature)
    lnQ = - energy * model.beta
    if verbose:
        print('lnQ =', lnQ)

    # structure free energies are assuming distinguishable strands, so if homodimer need to divide by 2 for indistinguishability
    # Equilibrium constant is affected by whether the strands are distinguishable
    # If identical strands, K = 1/conc; if non-identical strands, K = 2/conc
    if str(A) == str(~A):
        lnK = np.log(1/DNA_conc)
    else:
        lnK = np.log(2/DNA_conc)
        
    if verbose:
        print('lnK =', lnK)
    return lnQ - lnK # to get the difference of reaction quotient to equilibrium constant


    
def calculate_tm(seq, target_struct, sodium, DNA_conc, param_set):
    '''Simple Tm calculation using bisection to find where unpaired = paired population'''

    lo, hi = 0, 100
    get_nupack_model = lambda T: nupack.Model(material=param_set, celsius=T, sodium=sodium)

    if calculate_distance_2_equilibrium(seq, DNA_conc, get_nupack_model(lo), target_struct=target_struct) < 0:
        return lo

    if calculate_distance_2_equilibrium(seq, DNA_conc, get_nupack_model(hi), target_struct=target_struct) > 0:
        return hi

    while True:
        T = (lo + hi) / 2
        Q_K = calculate_distance_2_equilibrium(seq, DNA_conc, get_nupack_model(T), target_struct=target_struct)

        if abs(Q_K) < 1e-2:
            return T
        elif Q_K < 0:
            hi = T
        else:
            lo = T
            

def get_nupack_dH_dS_Tm_dG_37(seq, struct, sodium=1.0, return_dict=False, **kwargs):
    '''
    Return dH (kcal/mol), dS(kcal/mol), Tm (˚C), dG_37(kcal/mol)
    Use the better sodium correction equation
    '''
    T1=0
    T2=50

    dG_37 = get_seq_structure_dG(seq, struct, 37, **kwargs)

    dG_1 = get_seq_structure_dG(seq, struct, T1, **kwargs)
    dG_2 = get_seq_structure_dG(seq, struct, T2, **kwargs)

    dS = -1*(dG_2 - dG_1)/(T2 - T1)
    assert((dG_1 + dS*(T1+273.15)) - (dG_2 + dS*(T2+273.15)) < 1e-6)
    
    dH = dG_1 + dS*(T1+273.15)
    
    if dS != 0:
        if not '+' in struct:
            Tm = (dH/dS) - 273.15 # to convert to ˚C
        else:
            # duplex
            # kwargs: DNA_conc, param_set
            Tm = calculate_tm(seq, sodium=sodium, **kwargs)
            
        Tm = get_Na_adjusted_Tm(Tm=Tm, dH=dH, GC=get_GC_content(seq), 
                                    Na=sodium, from_Na=1.0)
        dG_37 = get_dG(Tm=Tm, dH=dH, celsius=37)
        dS = dH / (Tm + 273.15)
    else:
        Tm = np.nan
    if return_dict:
        return dict(dH=dH, dS=dS, Tm=Tm, dG_37=dG_37)
    else:
        return [dH, dS, Tm, dG_37]



def is_diff_nupack(df, param):
    return df.apply(lambda row: (row[param+'_lb'] > row[param+'_NUPACK_salt_corrected']) or (row[param+'_ub'] < row[param+'_NUPACK_salt_corrected']), axis=1)

def add_intercept(arr):
    """
    Helper function for fitting with LinearRegressionSVD
    """
    arr_colvec = arr.reshape(-1, 1)
    return np.concatenate((arr_colvec, np.ones_like(arr_colvec)), axis=1)

class LinearRegressionSVD(LinearRegression):
    """
    The last feature is intercept
    self.intercept_ is set to 0 to keep consistency
    y does not have nan ('raise' behavior)
    """
    def __init__(self, param='dG_37', **kwargs):
        super().__init__(fit_intercept=False, **kwargs)
        self.intercept_ = 0.0
        self.param = param
                
    def fit(self, X:np.array, y:np.array, y_err:np.array, sample_weight=None, 
            feature_names=None, singular_value_rel_thresh:float=1e-15, skip_rank:bool=False):
        """
        Params:
            skip_rank - if True, do not calculate the rank of the matrix
        """
        if sample_weight is not None:
            assert len(sample_weight) == len(y)
            y_err = 1 / sample_weight
            
        A = X / (y_err.reshape(-1,1))
        
        if not skip_rank:
            rank_A = np.linalg.matrix_rank(A)
            n_feature = A.shape[1]
            if rank_A < n_feature:
                print('Warning: Rank of matrix A %d is smaller than the number of features %d!' % (rank_A, n_feature))
            
        b = (y / y_err).reshape(-1,1)
        u,s,vh = np.linalg.svd(A, full_matrices=False)
        s_inv = 1/s
        s_inv[s < s[0]*singular_value_rel_thresh] = 0
        self.coef_se_ = np.sqrt(np.sum((vh.T * s_inv.reshape(1,-1))**2, axis=1))
        self.coef_ = (vh.T @ np.diag(s_inv) @ u.T @ b).flatten()
        
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        
        self.metrics = self.calc_metrics(X, y, y_err)
    
    
    def calc_metrics(self, X, y, y_err=None):        
        """
        Returns:
            metrics - Dict[str: float], ['rsqr', 'rmse', 'dof', 'chisq', 'redchi']
        """
        y_pred = self.predict(X=X)
        ss_total = np.nansum((y - y.mean())**2)
        ss_error = np.nansum((y - y_pred)**2)
        
        rsqr = 1 - ss_error / ss_total
        rmse = np.sqrt(ss_error / len(y))
        mean_abs_error = mae(y_pred, y)
        dof = len(y) - len(self.coef_)
        if y_err is not None:
            chisq = np.sum((y - y_pred.reshape(-1,1))**2 / y_err**2)
            redchi = chisq / dof
        else:
            chisq, redchi = np.nan, np.nan

        metrics = dict(rsqr=rsqr, rmse=rmse, mae=mean_abs_error, dof=dof, chisq=chisq, redchi=redchi)
        
        return metrics
    
    
    def fit_with_some_coef_fixed(self, X:np.array, y:np.array, y_err:np.array,
            feature_names, fixed_feature_names, coef_df, coef_se_df=None,# index_col='motif',
            singular_value_rel_thresh=1e-15, debug=False):
        """
        Fix a given list of coef of features and fit the rest. Calls self.fit()
        Args:
            feature_names - list like, ALL the feats (column names of X)
            fixed_feature_names - list like, names of the indices to look up from coef_df
            coef_df - pd.DataFrame, indices are feature names to look up, only one column with the coef
            coef_se_df - pd.DataFrame, indices are feature names to look up, only one column with the coef, 
                optional. If not given, se of the known parameters are presumably set to 0
        """
        
        A = X / (y_err.reshape(-1,1))
        b = (y / y_err).reshape(-1,1)
        
        known_param = [f for f in feature_names if f in fixed_feature_names]
        known_param_mask = np.array([(f in fixed_feature_names) for f in feature_names], dtype=bool)
        A_known, A_unknown = A[:, known_param_mask], A[:, ~known_param_mask]
        if debug:
            print('known_param_mask: ', np.sum(known_param_mask), known_param)
            print('A_unknown, A_known: ', A_unknown.shape, A_known.shape)
        n_feature = A.shape[1]
        n_feature_to_fit = A_unknown.shape[1]
        # x_unknown is to be solved; x_known are the known parameters
        print(coef_df)
        print(known_param)
        x_known = coef_df.loc[known_param, :].values.flatten()
        if debug:
            print('x_known: ', x_known.shape)
        b_tilde = b - A_known @ x_known.reshape(-1, 1)
        
        rank_A1 = np.linalg.matrix_rank(A_unknown)
        if rank_A1 < n_feature_to_fit:
            print('Warning: Rank of matrix A_unknown, %d, is smaller than the number of features %d!' % (rank_A1, n_feature_to_fit))
        
        # initialize
        self.coef_ = np.zeros(n_feature)
        self.coef_se_ = np.zeros(n_feature)
        
        u,s,vh = np.linalg.svd(A_unknown, full_matrices=False)
        s_inv = 1/s
        s_inv[s < s[0]*singular_value_rel_thresh] = 0
        x_unknown = (vh.T @ np.diag(s_inv) @ u.T @ b_tilde).flatten()
        x_se_unknown = np.sqrt(np.sum((vh.T * s_inv.reshape(1,-1))**2, axis=1))
        
        self.coef_[known_param_mask] = x_known
        self.coef_[~known_param_mask] = x_unknown
        self.coef_se_[~known_param_mask] = x_se_unknown
        if coef_se_df is not None:
            self.coef_se_[known_param_mask] = coef_se_df.loc[known_param, :].values.flatten()
        
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        
        self.metrics = self.calc_metrics(X, y, y_err)
    
        
    def predict_err(self, X):
        return (X @ self.coef_se_ .reshape(-1,1)).flatten()
        
        
    def fit_intercept_only(self, X, y):
        self.coef_[-1] = np.mean(y - X[:,:-1] @ self.coef_[:-1].reshape(-1,1))
        
        
    def set_coef(self, feature_names, coef_df, index_col='index'):
        """
        Force set coef of the model to that supplied in `coef_df`,
        sorted in the order of `feature_names`.
        For instance, external parameter sets like SantaLucia.
        `coef_se_` is set to 0.
        Args:
            feature_names - array-like of str, **WITH** the 'intercept' at the end of feature matrices
            coef_df - df, with a column named `self.param` e.g. dG_37
                and a columns called `index_col` to set the names of the coef to.
            index_col - str. col name to set names of the coef to.
                if 'index', use index.
        """
        if index_col != 'index':
            coef_df = coef_df.set_index(index_col)
            
        self.coef_ = np.append(coef_df.loc[feature_names[:-1],:][self.param].values, 0)
        self.coef_se_ = np.zeros_like(self.coef_)
        self.feature_names_in_ = feature_names
        
    @property
    def intercept(self):
        return self.coef_[-1]
    @property
    def intercept_se(self):
        return self.coef_se_[-1]
    @property
    def coef_df(self):
        return pd.DataFrame(index=self.feature_names_in_,
                            data=self.coef_.reshape(-1,1), columns=[self.param])
    @property
    def coef_se_df(self):
        return pd.DataFrame(index=self.feature_names_in_,
                            data=self.coef_se_.reshape(-1,1), columns=[self.param + '_se'])


class LinearRegressionRegularized(LinearRegressionSVD):
    """
    Regularized to a set of prior parameter values.
    Using `scipy.optimize.minimize()`
    A wrapper class for `fit_regularized_linear_regression()` function to make life easier
    The last feature is intercept
    self.intercept_ is set to 0 to keep consistency
    """
    def __init__(self, param='dG_37', reg_lambda:float=1e-2, **kwargs):
        """
        Args:
            reg_lambda - regularization scaling factor,
                just setting the default here, can be updated in the `fit()` method
        """
        self.reg_lambda = reg_lambda
        super().__init__(param=param, **kwargs)
        
    @staticmethod   
    def fit_regularized_linear_regression(X, y, coef_p, reg_lambda):
        def objective_func(coef, X, y, coef_p, reg_lambda):
            """
            X·coef = y
            """
            residual = X @ coef - y
            pred_cost = np.mean(residual**2)
            reg_cost = reg_lambda * np.mean(np.square(coef - coef_p))

            return pred_cost + reg_cost
            
        coef0 = coef_p.copy()
        result = opt.minimize(fun=objective_func, x0=coef0,
                            args=(X, y, coef_p, reg_lambda),
                            method='L-BFGS-B')
        return result.x
        
    def fit(self, X:np.array, y:np.array, 
            coef_prior=None, feature_names=None, coef_df=None,
            reg_lambda=None, normalize_X=False):
        """
        Params:
            X - (n_sample, n_feat)
            y - (n_sample, n_dim)
            coef_prior - (n_feat,) either provide this or `feature_names` and `coef_df`
            feature_names - (n_feat,) list like, 
                looks up in coef_df
            coef_df - prior feature coefficients, 
                index is a super set of feature_names
        Updates:
            self.feat_norm - (n_feat,) scaling factors for feature normalization
                `X * self.feat_norm = X_norm`
            self.coef_
            self.coef_se_ (set to None)
            self.feature_names_in_
            self.metrics
            * self.scaler - MaxAbsScaler() instance if `normalize_X` set to True
                Note that MinMaxScaler() would shift the data and disturb scarcity
                Really shouldn't use in NN model context but implemented for the remote
                likelihood that we need it for some really skewed feature range distribution
        """
        # Update `self.reg_lambda` value
        if reg_lambda is not None:
            self.reg_lambda = reg_lambda
            
        # Prepare `coef_prior`
        if coef_prior is not None:
            assert len(coef_prior) == X.shape[1]
        else:
            assert (feature_names is not None) and (coef_df is not None)
            coef_prior = coef_df.loc[feature_names,:].values.flatten()
            print("Coef prior:")
            
        # Normalize X if needed
        if normalize_X:
            self.scaler = MaxAbsScaler()
            X = self.scaler.fit_transform(X)
        
        # Fitting
        self.coef_ = self.fit_regularized_linear_regression(X, y, coef_prior, self.reg_lambda)    
        self.coef_se_ = None
        self.feature_names_in_ = feature_names
        self.metrics = self.calc_metrics(X, y)             
        
        
    def fit_with_some_coef_fixed(self, X:np.array, y:np.array,
            feature_names, fixed_feature_names, coef_df, 
            debug=False):
        """
        Params:
            X - (n_sample, n_feat)
            y - (n_sample, n_dim)
            feature_names - (n_feat,) list like, 
                looks up in coef_df, has to match X.shape[1]
            coef_df - prior feature coefficients, 
                index is a super set of `feature_names` and `fixed_feature_names`
                those in `fixed_feature_names` are fixed during fitting
                those not in `fixed_feature_names` are used as prior
        Updates:
            self.coef_
            self.coef_se_ (set to None bc I'm lazy)
            self.feature_names_in_
            self.metrics
        """
        assert len(feature_names) == X.shape[1]
        known_param = [f for f in feature_names if f in fixed_feature_names]
        unknown_param = [f for f in feature_names if not f in fixed_feature_names]
        
        known_param_mask = np.array([(f in fixed_feature_names) for f in feature_names], dtype=bool)
        X_known, X_unknown = X[:, known_param_mask], X[:, ~known_param_mask]
        if debug:
            print('known_param_mask: ', np.sum(known_param_mask), known_param)
            print('X_unknown, X_known: ', X_unknown.shape, X_known.shape)
            
        n_feature = X.shape[1]
        
        # coef_unknown is to be solved, here just extracted as a prior; 
        # coef_known are the known parameters to be fixed
        coef_known = coef_df.loc[known_param, :].values.flatten()
        coef_unknown = coef_df.loc[unknown_param, :].values.flatten()
        try:
            y_tilde = (y.reshape(-1,1) - X_known @ coef_known.reshape(-1, 1)).flatten()
        except:
            # if debug:
            print('shape of coef_known: ', coef_known.shape)
            print("X_unknown, y_tilde, coef_unknown: \n", X_unknown)
            print(y_tilde)
            print(coef_unknown)

        fitted_coef_unknown  = self.fit_regularized_linear_regression(
            X_unknown, y_tilde, coef_p=coef_unknown, reg_lambda=self.reg_lambda)    
        
        if debug:
            print('fitted_coef_unknown: ', fitted_coef_unknown)
        
        # Update attributes
        self.coef_ = np.zeros(n_feature)
        self.coef_[known_param_mask] = coef_known
        self.coef_[~known_param_mask] = fitted_coef_unknown
        self.coef_se_ = None
        self.feature_names_in_ = feature_names
        self.metrics = self.calc_metrics(X, y)
        
#### Fluor simulation related ####

def get_fluor_distance_from_structure(structure: str):
    return len(structure) - len(structure.strip('.'))
    
    
def get_duplex_row(hairpin_row):
    """
    Assume loop size is 4
    """
    row = hairpin_row.copy()
    stem_len = int((len(row.RefSeq) - 4.0) / 2.0)
    row.TargetStruct = row.TargetStruct.replace('....','+')
    row.RefSeq = row.RefSeq[:stem_len] + '+' + row.RefSeq[-stem_len:]
    return row
"""
# Old nupack dH dS Tm code, has problems
def calc_dH_dS_Tm(seq, package='nupack',dna=True):
    '''Return dH (kcal/mol), dS(kcal/mol), Tm (˚C)'''

    T1=0
    T2=50

    dG_37C = free_energy(seq, T=37, package=package, dna=dna)

    dG_1 = free_energy(seq, T=T1, package=package, dna=dna)
    
    dG_2 = free_energy(seq, T=T2, package=package, dna=dna)
    
    dS = -1*(dG_2 - dG_1)/(T2 - T1)
    
    assert((dG_1 + dS*(T1+273.15)) - (dG_2 + dS*(T2+273.15)) < 1e-6)

    dH = dG_1 + dS*(T1+273.15)
    
    if dS != 0:
        Tm = (dH/dS) - 273.15 # to convert to ˚C
    else:
        Tm = np.nan
    #print((dH - dS*273),free_energy(seq, T=0, package=package, dna=dna))
    
    return dH, dS, Tm, dG_37C
"""

def rmse(y1, y2):
    return np.sqrt(np.mean(np.square(y1 - y2)))
def mae(y1, y2):
    return np.mean(np.abs(y1 - y2))
def pearson_r(y1, y2):
    return pearsonr(y1, y2)[0]