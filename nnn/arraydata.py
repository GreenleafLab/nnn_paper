"""
The ArrayData class reads and stores all replicates, annotation and combined data
for an array library.

Yuxi Ke, Feb 2022
"""

from distutils.log import error
from pickletools import float8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, Dict
from . import fileio, processing


class ErrorAdjust(object):
    """
    Error adjust from intra- to inter- experiment
    \Sigma (\sigma) = A \sigma ^{k}
    """
    def __init__(self, param='dG_37') -> None:
        self.A = 1.0
        self.k = 1.0
        self.param = param

    def adjust_sigma(self, sigma):
        return self.A * sigma**self.k

class ArrayData(object):
    """
    Contains all replicates of the same condition,
    combined and error adjusted.
    Attributes:
        name - str
        lib_name - str, 'nnn_lib2b'
        n_rep - int
        buffer - Dict[str: value], {'sodium': float, in M}
        data_all - df, all shitty columns
        data - df, just the clean combined parameters
        curve - dict of pd.DataFrame, keys are replicate,
                levels are variant-replicate-median/se-temperature. 
                Green normed data.
        p_unfold - pd.DataFrame, normalized to p_unfold and combined across reps.
        celsius - dict of array, temperatures for the curves of each replicate.
        annotation - pd.DataFrame, designed library
        replicate_df - pd.Dataframe with the information of replicate locations
                     No actual data is loaded
    

    """
    def __init__(self, replicate_df, annotation_file, name='',
                 lib_name='NNNlib2b', filter_misfold: bool=False,
                 learn_error_adjust_from: Tuple[str]=None, error_adjust: ErrorAdjust=None ) -> None:
        """
        Args:
            replicate_df - df, cols ['name', 'replicate', 'chip', 'filename', 'get_cond', 'notes']
            TODO filter_misfold - whether to filter out misfolded secondary structures
            learn_error_adjust_from - Tuple[str], 2 replicates to learn adjustion from
                required if error_adjust is None
            error_adjust - if given, use error adjust function already determined externally
        """
        self.name = name
        self.lib_name = lib_name
        self.replicate_df = replicate_df
        if len(replicate_df.shape) == 2:
            self.n_rep = len(replicate_df)
        else:
            self.n_rep = 1
            

        if self.n_rep > 1:
            assert np.all(np.isclose(replicate_df['sodium'], replicate_df['sodium'][0])), "Sodium concentration not equal in the replicates"
            self.buffer = {'sodium': replicate_df['sodium'][0]}
        else:
            self.buffer = {'sodium': replicate_df['sodium']}
        
        self.annotation = fileio.read_annotation(annotation_file, sodium=self.buffer['sodium'])
        if filter_misfold:
            pass
        
        # load data
        if self.n_rep > 1:
            self.data_all, self.curve, self.curve_se, self.p_unfold = self.read_data()
        else:
            self.data_all, self.curve, self.curve_se, self.p_unfold = self.read_data_single()

        # celsius
        if not hasattr(self, 'celsius'):
            self.celsius = dict()
            for rep in self.curve:
                self.celsius[rep] = np.array([float(x.split('_')[1]) for x in self.curve[rep].columns])
        
        # data accounting for num variant per series after each step of processing
        steps = ['designed', 'fitted', 'passed2state']
        series = np.unique(self.annotation.Series)
        self.accounting_df = pd.DataFrame(index=steps, columns=series)
        self.accounting_df.loc['designed', :] = self.annotation.groupby('Series').apply(len)
        self.accounting_df.loc['fitted', :] = self.data_all.join(self.annotation).groupby('Series').apply(len)

        if error_adjust is not None:
            self.error_adjust = error_adjust
        else:
            if learn_error_adjust_from is None:
                # OK to have no adjust
                self.error_adjust = ErrorAdjust()
                # raise Exception('Need to give `learn_error_adjust_from` if no ErrorAdjust is given!')
            else:
                self.error_adjust = self.learn_error_adjust_function(learn_error_adjust_from)

        self.data = self.data_all[[c for c in self.data_all.columns if not ('-' in c)]]

    @staticmethod
    def calc_p_unfold(rep_name, rep, cond):
        cols = ['%s_%.1f'%(rep_name, float(x.split('_')[1])) for x in cond]
        melt_curve = (rep[cond].values - rep.fmin.values.reshape(-1,1)) / (rep.fmax - rep.fmin).values.reshape(-1,1)
        melt_curve = np.clip(melt_curve, a_min=0, a_max=1)
        return pd.DataFrame(data=melt_curve, index=rep.index, columns=cols)

    def read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        reps = [fileio.read_fitted_variant(fn, add_chisq_test=False, annotation=None) for fn in self.replicate_df.filename] # List[DataFrame]
        
        conds = []
        for rep, drop_last, reverse in zip(reps, self.replicate_df.drop_last, self.replicate_df.reverse):
            cond = [x for x in rep.columns if x.endswith('_norm')]
            if drop_last:
                cond = cond[:-1]
            if reverse:
                cond = cond[::-1]
            conds.append(cond)

        data = processing.combine_replicates(reps, self.replicate_df['name'].tolist(), verbose=False)
        curves = {rep_name: rep[cond] for rep, rep_name, cond in zip(reps, self.replicate_df['name'], conds)}
        curves_se = {rep_name: rep[[c+'_std' for c in cond]].rename(columns=lambda c: c.replace("_std", "_se")) / np.sqrt(rep['n_clusters']).values.reshape(-1,1)
            for rep, rep_name, cond in zip(reps, self.replicate_df['name'], conds)}
        
        self.celsius = dict()
        for rep in curves:
            self.celsius[rep] = np.array([float(x.split('_')[1]) for x in curves[rep].columns])

        p_unfold_dict = {rep_name: self.calc_p_unfold(rep_name, rep, cond) for rep, rep_name, cond in zip(reps, self.replicate_df['name'], conds)}
        # print(self.celsius)
        p_unfold = processing.combine_replicate_p_unfold(p_unfold_dict, self.celsius)
        return data, curves, curves_se, p_unfold
        

    def read_data_single(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        rep = fileio.read_fitted_variant(self.replicate_df.filename, add_chisq_test=False, annotation=None)
        
        cond = [x for x in rep.columns if x.endswith('_norm')]
        if self.replicate_df.drop_last:
            cond = cond[:-1]
        if self.replicate_df.reverse:
            cond = cond[::-1]

        data = processing.combine_replicates([rep], self.replicate_df['name'], verbose=False)
        curves = {self.replicate_df['name']: rep[cond]}
        curves_se = {self.replicate_df['name']: rep[[c+'_std' for c in cond]].rename(columns=lambda c: c.replace("_std", "_se")) / np.sqrt(rep['n_clusters']).values.reshape(-1,1)}
        p_unfold = {self.replicate_df['name']: self.calc_p_unfold(self.replicate_df['name'], rep, cond)}
        return data, curves, curves_se, p_unfold


    def get_replicate_data(self, replicate_name: str, attach_annotation: bool = False, verbose=True) -> pd.DataFrame:
        """
        Lower level, returns the original df of fitted variant data
        Compatible with older functions
        """
        # assert replicate_name in self.replicate_df['name'], "Invalid replicate name"
        filename = self.replicate_df.loc[self.replicate_df.name == replicate_name, 'filename'].values[0]
        if verbose:
            print('Load from file', filename)
        if attach_annotation:
            annotation = self.annotation
        else:
            annotation = None
            
        return fileio.read_fitted_variant(filename, filter=True, annotation=annotation, sodium=self.buffer['sodium'])
            
    def get_replicate_curves(self, replicate_name: str, verbose: bool = True):
        """
        Return the normalized values and std of a replicate experiment.
        """
        repdata = self.get_replicate_data(replicate_name=replicate_name, verbose=verbose)
        values = repdata[[c for c in repdata.columns if c.endswith('_norm')]]
        se = repdata[[c for c in repdata.columns if c.endswith('_norm_std')]] / np.sqrt(repdata['n_clusters'].values.reshape(-1,1))
        xdata = np.array([c.split('_')[1] for c in repdata.columns if c.endswith('_norm_std')], dtype=float)
        xdata += 273.15
        
        return xdata, values, se


    def learn_error_adjust_function(self, learn_error_adjust_from, debug=False, figdir='./fig/error_adjust'):
        """
        Args:
            learn_error_adjust_from - Tuple of 2 str, replicate names
        """
        r1_name, r2_name = learn_error_adjust_from
        r1, r2 = self.get_replicate_data(r1_name), self.get_replicate_data(r2_name)

        error_adjust = ErrorAdjust()
        error_adjust.A, error_adjust.k = processing.correct_interexperiment_error(r1, r2, figdir=figdir, return_debug=debug)

        return error_adjust
    
    def get_two_state_df(self, force_recalculate:bool=False,
                         myfilter:str="dH_err_rel < 0.2 & Tm_err_abs < 2 & redchi < 1.5 & n_inlier > 10") -> pd.DataFrame:
        """
        Load the line fit files for a df with replicate level information
        """
        if (not hasattr(self, 'two_state_df')) or force_recalculate:
            line_fit = []
            for i in range(self.n_rep):
                rep = self.get_replicate_data(self.replicate_df.iloc[i]['name'], verbose=False)
                result_df = pd.read_table(self.replicate_df.iloc[i]['line_fit_filename']).set_index('SEQID')
                df = rep.join(result_df, lsuffix='_curve', rsuffix='_line')
                
                
                df['Tm_err_abs'] = np.abs(df.Tm_curve - df.Tm_line)
                df['dH_err_rel'] = - np.abs(df.dH_curve - df.dH_line) / df.dH_line
                df['n_inlier'] = df['dof_line'] + 2
                
                df['two_state'] = df.eval(myfilter)
                line_fit.append(df)
                
            cols = ['dH_line', 'dH_se_line', 'dG_37_line', 'dH_err_rel', 'Tm_err_abs', 'redchi', 'n_inlier', 'two_state']
            self.two_state_df = pd.concat([d[cols] for d in line_fit], axis=1, keys=self.replicate_df['name'].tolist())
        
        return self.two_state_df
        
        
    def filter_two_state(self, min_rep_pass:int=2, force_recalculate:bool=False, overwrite_dH:bool=False,
                         myfilter:str="dH_err_rel < 0.2 & Tm_err_abs < 2 & redchi < 1.5 & n_inlier > 10",
                         inplace=False):
        """
        Judge final two state behavior based on the replicate level information
        Args:
            overwrite_dH - bool, overwrite the curve fitting dH with line fitting dH
                otherwise use column name 'dH_line'
            inplace - bool, if True, filter self.data inplace
                otherwise simply append it to the df
        """
    
        two_state_df = self.get_two_state_df(myfilter=myfilter, force_recalculate=force_recalculate)
        n_pass = np.nansum(two_state_df.xs('two_state', axis=1, level=1), axis=1)
        pass_df = pd.DataFrame(data=(n_pass >= min_rep_pass), index=two_state_df.index, columns=['two_state'])
        
        self.accounting_df.loc['passed2state', :] = pass_df.query('two_state').join(self.annotation).groupby('Series').apply(len)
        
        fig, ax = plt.subplots()
        sns.histplot(two_state_df[n_pass >= min_rep_pass]['r1','dH_err_rel'].dropna(), 
                    bins=np.arange(0,1,0.02), alpha=.3,# stat='density',
                    color='g')
        sns.histplot(two_state_df[n_pass < min_rep_pass]['r1','dH_err_rel'].dropna(), 
                    bins=np.arange(0,1,0.02), alpha=.3,# stat='density',
                    color='r')
        plt.legend(['pass', 'not pass'])

        plt.title('%.2f%% variants pass the two state filter' % (np.sum(n_pass >= min_rep_pass) / len(n_pass) * 100))
        plt.show()
            
        # use the dH from line fitting for reduced uncertainty
        dH_line, dH_se_line = processing.get_combined_param(
                self.two_state_df.xs('dH_line', axis=1, level=1),
                self.two_state_df.xs('dH_se_line', axis=1, level=1))

        fig, ax = plt.subplots(1, 2, figsize=(6,2))
        histargs = dict(bins=np.arange(0,10,.2), kde=False, ax=ax[0])
        sns.histplot(dH_se_line,color='cornflowerblue', label='line fit', **histargs)
        sns.histplot(self.data['dH_se'], color='c',label='curve fit', **histargs)
 
        histargs = dict(bins=np.arange(-60,0,1), kde=False, ax=ax[1])
        sns.histplot(dH_line,color='cornflowerblue', label='line fit', **histargs)
        sns.histplot(self.data['dH'], color='c',label='curve fit', **histargs)
        plt.legend()
        plt.show()
        if overwrite_dH:
            suffix = ''
        else:
            suffix = '_line'
       
        if 'two_state' in self.data_all.columns:
            self.data_all.drop('two_state', axis=1, inplace=True)
            self.data.drop('two_state', axis=1, inplace=True)
        
        dH, dH_se = self.data['dH'], self.data['dH_se']
        self.data_all = self.data_all.join(pass_df)
        self.data_all['dH'+suffix], self.data_all['dH_se'+suffix] = dH_line, dH_se_line
        self.data_all['dH'+suffix] = self.data_all['dH'+suffix].fillna(dH)
        self.data_all['dH_se'+suffix] = self.data_all['dH_se'+suffix].fillna(dH_se)
        
        self.data = self.data.join(pass_df)
        self.data['dH'+suffix], self.data['dH_se'+suffix] = dH_line, dH_se_line
        self.data['dH'+suffix] = self.data['dH'+suffix].fillna(dH)
        self.data['dH_se'+suffix] = self.data['dH_se'+suffix].fillna(dH_se)

        if inplace:
            self.data = self.data.query('two_state == 1')
            
        return pass_df