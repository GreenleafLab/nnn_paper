from tkinter import N
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json
import seaborn as sns
# from sklearn import ensemble
sns.set_style('ticks')
sns.set_context('paper')
from ipynb.draw import draw_struct
import nupack
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed
from lmfit import Parameters, minimize
from typing import Set, Tuple, Dict
from . import util

np.random.seed(42)
kB = 0.0019872
##############################
###### Distance Functon ######
##############################

def distance_2_norm_fluor(x, a=93):
    norm_fluor = 1.0 / (1.0 + (a * x**(-3.0)))
    norm_fluor[np.isinf(norm_fluor)] = 0.0
    return norm_fluor

def get_transform_curve(max_len=50, a=93):
    nt_range = np.arange(max_len)
    transform_curve = distance_2_norm_fluor(nt_range, a=a)
    return transform_curve

######################
###### Sampling ######
######################

def simulate_nupack_curve_p_closing(seq, sodium=1.0, T=np.arange(20,62.5,2.5)):
    """
    Simulate P[closing base pair] as a function of temperatures
    returns P[open base pair] to mimic melt curve in an experiment
    """

    p_close = np.zeros_like(T)
    for i,celsius in enumerate(T):
        p_close[i] = util.get_seq_end_pair_prob(seq, celsius=celsius, sodium=sodium, n_pair=2)
    
    return 1 - p_close


def sample_nupack_nt_distance(seq, num_sample=100, sodium=1.0, celsius=37, verbose=False):
    """
    Sample distances at one temperature point.
    """
    my_model = nupack.Model(material='DNA', celsius=celsius, sodium=sodium, magnesium=0.0)
    sampled_structures = nupack.sample(strands=[seq], num_sample=num_sample, model=my_model)
    if verbose:
        print(sampled_structures)
    nt_distances = np.array([util.get_fluor_distance_from_structure(str(s)) for s in sampled_structures], dtype=int)
    
    return nt_distances


def sample_nupack_curve_distance(seq, num_sample=100, sodium=1.0, T=np.arange(20, 62.5, 2.5), verbose=False):
    """
    [Deprecated]
    Simulates a melt curve of distances in nt
    Returns:
        nt_distances - (n_sample, n_temperature) np.array
    """
    nt_distances = np.zeros((num_sample, len(T)), dtype=int)
    
    for i,celsius in enumerate(T):
        nt_distances[:,i] = sample_nupack_nt_distance(seq, num_sample, sodium, celsius, verbose)
    
    return nt_distances


def simulate_nupack_curve_by_sampling(seq, num_sample=1000, sodium=1.0, T=np.arange(20, 62.5, 2.5),
                          transform_param={'a': 93}, transform_curve=None):
    """
    Args:
        transform_param - param for the transform curve
        transform_curve - array like, `transform_curve[n]` is the normalized fluorescence 
    Returns:
        simulated fluorescence curve (macroscopic mean of the ensemble)
    """
    if transform_curve is None:
        # if did not supply existing transform_curve array, requiring transform_param dict
        nt_range = np.arange(50)
        transform_curve = distance_2_norm_fluor(nt_range, a=transform_param['a'])
        
    nt_distances = sample_nupack_curve_distance(seq, num_sample=num_sample, T=T, sodium=sodium)
    nt_distances = transform_curve[nt_distances]

    return np.mean(nt_distances, axis=0)
        
        
def plot_nupack_curve_by_sampling(seq, num_sample=1000, sodium=1.0, T=np.arange(20, 62.5, 2.5),
                      transform_curve=None, ax=None):
    """
    Args:
        transform_curve - array like, `transform_curve[n]` is the normalized fluorescence 
        expected at a distance of n nt. If not given, only plot the raw distance.
    """
    nt_distances = sample_nupack_curve_distance(seq, num_sample=num_sample, T=T, sodium=sodium)
    if transform_curve is not None:
        nt_distances = transform_curve[nt_distances]

    df = pd.DataFrame(nt_distances, columns=T).melt()
    df.columns = ['temperature', 'distance_nt']
    # plt.errorbar(T, np.median(nt_distances, axis=1), yerr=np.std(nt_distances, axis=1), fmt='k.')
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,4))
    sns.violinplot(data=df, x='temperature', y='distance_nt', palette='magma', 
                   inner=None, linewidth=0, ax=ax)
    # ax.plot(np.median(nt_distances, axis=0), 'orange', linewidth=2, zorder=9, label='median')
    ax.plot(np.mean(nt_distances, axis=0), 'purple', linewidth=4, zorder=10, label='mean')
    ax.legend(loc='lower right')

    if transform_curve is not None:
        ax.set_ylabel('Normalized Fluorescence')

#########################
###### Probability ######
#########################

def simulate_nupack_fluor(seq, distance_curve, sodium=1.0, celsius=37, 
                             energy_gap=0.5, verbose=False, energy_key='energy', **kwargs):
    """
    Simulates distances at one temperature point.
    Args:
        energy_gap - float, in kcal/mol
        distance_curve - array
        energy_key - str, {'energy', 'stack_energy'}
        energy_offset - float, offsets the unfolded structure in kcal/mol
    Returns:
        ensemble_fluorescence - float, in normalized fluorescence unit a.u.
    """
    my_model = nupack.Model(material='dna04', celsius=celsius, sodium=sodium, magnesium=0.0, ensemble='nostacking')
    # only consider structures energetically close enough to the fully folded / mfe structure
    subopt_struct = nupack.subopt(strands=[seq], energy_gap=energy_gap, model=my_model)
    if verbose:
        print(subopt_struct)
        
    structures = [str(s.structure) for s in subopt_struct]
    energies = [getattr(s, energy_key) for s in subopt_struct]
    
    # only consider structures close enough to the unfolded structure
    # potential_struct = nupack.subopt(strands=[seq], energy_gap=-np.min(energies), model=my_model)
    # all_structures = structures + [str(s.structure) for s in potential_struct if ((getattr(s, energy_key) > -energy_gap) and not (str(s.structure) in structures))]
    # # print('T =', celsius, ':', len(all_structures) - len(structures))
    # all_energies = energies + [getattr(s, energy_key) for s in potential_struct if ((getattr(s, energy_key) > -energy_gap) and not (str(s.structure) in structures))]
    
    # all_energies = energies
    # all_structures = structures
    
    # unfolded_stuct = '.'*len(seq)
    # if not unfolded_stuct in structures:
    #     structures.append(unfolded_stuct)
    #     energies = np.append(energies, 0.0)
        
    distances = [util.get_fluor_distance_from_structure(s) for s in structures]
    fluors = distance_curve[distances]

    weights = np.exp(-np.array(energies))
    ensemble_fluorescence = np.sum(weights * fluors) / np.sum(weights)    
    return ensemble_fluorescence


def simulate_nupack_curve(seq, sodium=1.0, T=np.arange(5, 95, 1), 
                          dG_gap_kcal_mol=1.0, dG_gap_kT=None,
                          distance_param={'a': 93}, distance_curve=None, **kwargs):
    """
    Args:
        seq - str, up to 50 nt
        dG_gap_kT - float, if given, overwrites `dG_gap_kcal_mol`
        distance_param - param for the distance curve
        distance_curve - array like, `distance_curve[n]` is the normalized fluorescence 
    Returns:
        curve - (n_T,) array, simulated fluorescence curve (macroscopic mean of the ensemble)
    """
    if distance_curve is None:
        # if did not supply existing distance_curve array, requiring distance_param dict
        nt_range = np.arange(50)
        distance_curve = distance_2_norm_fluor(nt_range, a=distance_param['a'])
        
    if dG_gap_kT is not None:
        curve = np.zeros_like(T)
        for i, celsius in enumerate(T):
            energy_gap = kB * (celsius + 273.15) * dG_gap_kT
            curve[i] = simulate_nupack_fluor(seq, distance_curve, sodium=sodium, celsius=celsius, energy_gap=energy_gap, **kwargs)
    else:
        curve = np.array([simulate_nupack_fluor(seq, distance_curve, sodium=sodium, celsius=c, energy_gap=dG_gap_kcal_mol, **kwargs) for c in T])
        
    return curve


################################################
###### Enumerate all secondary structures ######
################################################

def get_all_secondary_structures(seq, T=np.arange(5, 97.5, 2.5), sodium=1.0,
                                 dG_gap_kcal_mol=1.0, dG_gap_kT=None) -> Set[str]:
    """
    All possible secondary structures within a certain range of temperature
    Returns:
        all_structures - Set[str]
    """
    all_stuctures = set()
    all_stuctures.add('.'*len(seq))

    for celsius in T:
        if dG_gap_kT is not None:
            energy_gap = kB * (celsius + 273.15) * dG_gap_kT
        else:
            energy_gap = dG_gap_kcal_mol
            
        assert np.isfinite(energy_gap)
        my_model = nupack.Model(material='dna04', celsius=celsius, sodium=sodium, magnesium=0.0, ensemble='stacking')
        sec_structs = nupack.subopt(strands=[seq], energy_gap=energy_gap, model=my_model)
        all_stuctures.update([str(s.structure) for s in sec_structs])
    
    return all_stuctures
    
    
def calc_nupack_param_of_secondary_structure(seq, struct, sodium=1.0) -> Dict:
    '''Helper function. Return dH (kcal/mol), dS(kcal/mol), Tm (˚C), Tm_corrected (°C)'''
    T1=0
    T2=50

    dG_1 = util.get_seq_structure_dG(seq, struct, T1, sodium=1.0, ensemble='stacking')
    dG_2 = util.get_seq_structure_dG(seq, struct, T2, sodium=1.0, ensemble='stacking')

    dS = -1 * (dG_2 - dG_1) / (T2 - T1)
    assert((dG_1 + dS*(T1+273.15)) - (dG_2 + dS*(T2+273.15)) < 1e-6)
    
    dH = dG_1 + dS*(T1+273.15)
    
    if dS != 0:
        Tm = (dH/dS) - 273.15 # to convert to ˚C
        GC = util.get_GC_content(seq)
        Tm_corrected = util.get_Na_adjusted_Tm(Tm, dH, GC, Na=sodium)
    else:
        Tm = np.nan
        Tm_corrected = np.nan
    
    nupack_param = dict(dH=dH, dS=dS, Tm=Tm, Tm_corrected=Tm_corrected)
    return nupack_param
    
    
def get_dG_curve_of_secondary_structure(seq, struct, sodium=1.0, T=np.arange(5, 97.5, 2.5), return_dict=False):
    """
    Calls `get_nupack_param_of_secondary_structure` first
    Args:
        return_dict - bool. If True, return a dict otherwise a np.array
    """
    p = calc_nupack_param_of_secondary_structure(seq, struct, sodium)
    dS_corrected = p['dH'] / p['Tm_corrected']
    # dG_curve = p['dH'] - (T + 273.15) * dS_corrected
    dG_curve = [util.get_seq_structure_dG(seq, struct, celsius, sodium, ensemble='stacking') for celsius in T]
    if return_dict:
        return dict(zip(T, dG_curve))
    else:
        return dG_curve
    
    
def simulate_nupack_curve_all_struct(seq:str, target_struct:str=None, sodium:float=1.0, T:np.array=np.arange(5, 95, 1),
                                     dG_gap_kT:float=1.0,
                                     distance_param:Dict={'a': 93}, distance_curve:np.array=None, **kwargs) -> np.array:
    """
    Args:
        seq - str, up to 50 nt
        dG_gap_kT - float, if given, overwrites `dG_gap_kcal_mol`
        distance_param - param for the distance curve
        distance_curve - array like, `distance_curve[n]` is the normalized fluorescence 
    Returns:
        curve - (n_T,) array, simulated fluorescence curve (macroscopic mean of the ensemble)
    """
    # def cutoff_dG(dG_df, target_struct):
    #     """
    #     TODO: Set dG to 0 if falls outside cutoff using certain rules
    #     """
    #     return dG_df
        
    if distance_curve is None:
        # if did not supply existing distance_curve array, requiring distance_param dict
        nt_range = np.arange(50)
        distance_curve = distance_2_norm_fluor(nt_range, a=distance_param['a'])
            
    all_structures = get_all_secondary_structures(seq, T, sodium, dG_gap_kT=dG_gap_kT)
    distances = [util.get_fluor_distance_from_structure(s) for s in all_structures]
    fluors = distance_curve[distances]
    
    dG_df = pd.DataFrame(index=all_structures, columns=T)
    for s in all_structures:
        dG_df.loc[s, :] = get_dG_curve_of_secondary_structure(seq, s, sodium, T)

    # dG_cutoff_df = cutoff_dG(dG_df, target_struct)
    exp_dG = np.exp(-dG_df.values.astype(float))
    prob_T = exp_dG / np.sum(exp_dG, axis=0).reshape(1,-1)
    
    curve = np.sum(fluors.reshape(-1, 1) * prob_T, axis=0)
    
    return curve

######################
###### Scale up ######
######################

def simulate_CPseries(annotation, sodium=0.075, T=np.arange(20, 62.5, 2.5), n_jobs:int=2, dG_gap_kT:float=1.0, **kwargs):
    """
    Simulates melt curves for the entire library.
    Args:
        transform_param - param for the transform curve
    Returns:
        series_df - (n_seq, n_temp) simulated fluorescence curve (macroscopic mean of the ensemble)
    """
    # n_seq = annotation.shape[0]
    # n_temp = len(T)
    transform_curve = get_transform_curve(max_len=50, a=93.0)
    refseqs = annotation.RefSeq
    
    if n_jobs == 1:
        print("No parallization")
        curves = np.zeros((len(refseqs), len(T)))
        for i,seq in enumerate(refseqs):
            curves[i, :] = simulate_nupack_curve(seq, sodium=sodium, T=T, transform_curve=transform_curve, dG_gap_kT=dG_gap_kT, **kwargs)
    else:
        curves = Parallel(n_jobs=n_jobs)(delayed(simulate_nupack_curve)(seq,
                                     sodium=sodium, T=T, transform_curve=transform_curve,  dG_gap_kT=dG_gap_kT, **kwargs) for seq in refseqs)
    
    return curves
    
    
#####################
###### Fitting ######
#####################

def residual(params, x, data):
    dH = params['dH']
    Tm = params['Tm']
    kB = 0.0019872
    
    model = 1 / (1 + np.exp((dH/kB) * (1/(Tm + 273.15) - 1/(x + 273.15))))

    return model - data

def fit_curve(y, T, plot=False, ylim=False):
    params = Parameters()
    params.add('dH', value=-40, max=0.0, min=-200.0)
    params.add('Tm', value=50, max=300, min= -100)
    out = minimize(residual, params, args=(T, y))
    
    dH = out.params['dH'].value
    Tm = out.params['Tm'].value
    dG_37 = util.get_dG(dH, Tm, 37.0)
    rmse = np.sqrt(np.mean(out.residual**2))
    chisq = out.chisqr
    
    if plot:
        plt.plot(T, y, 'o')
        plt.plot(T, residual(out.params, T, y) + y, 'x', label='best fit')
        if ylim:
            plt.ylim([-.05, 1.05])
        plt.legend()
        plt.show()
    
    return np.array([dH, Tm, dG_37, rmse, chisq], dtype=float)
    
    
def fit_simulated_nupack_series(series_df, n_jobs=1, T_subset_ind=None):
    """
    Takes the series df from output of simulate_CPseries
    Returns a df with dH, Tm, dG_37, rmse and chisq
    Args:
        T_subset_ind - array, indeces. only fit using a subset of temperature points
    """
    
    T = series_df.columns
    if T_subset_ind is None:
        T_subset_ind = np.arange(len(T))
        
    result_col = ['dH', 'Tm', 'dG_37', 'rmse', 'chisq']
    results = Parallel(n_jobs=n_jobs)(delayed(fit_curve)(y[T_subset_ind], T[T_subset_ind]) for _,y in series_df.iterrows())
    
    return pd.DataFrame(data=results, columns=result_col, index=series_df.index)