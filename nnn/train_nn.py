import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from . import util, fileio, mupack, modeling
from . import motif_fit as mf
from tqdm import tqdm

import wandb

def model_pipeline(hyperparameters, tags=None):

    # tell wandb to get started
    with wandb.init(project="NN", config=hyperparameters, tags=tags):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        lr_dict = train(config)
      
        if config['use_model_from'] == 'lr_dict':
            test(config, lr_dict=lr_dict)
        elif config['use_model_from'] == 'json':
            json_file = './models/%s.json' % wandb.run.name
            test(config, json_file=json_file)


    return lr_dict
           
class MyData(object):
    def __init__(self, config=None) -> None:
        self.param_set_template_file = './models/dna04.json'
        if config is not None:
            if 'struct_pred_param_file' in config:
                self.param_set_template_file = config['struct_pred_param_file']
        self.config = config

        
    def load_everything(self):
        # Array
        self.arr = pd.read_csv('./data/models/raw/arr_v1_n=27732.csv', index_col=0)
        self.arr_adj = pd.read_csv('./data/models/processed/arr_v1_adjusted_n=27732.csv', index_col=0)

        # UV melt
        agg_result_file='./data/uv_melt/uvmelt_agg.csv'
        self.uv_df = pd.read_csv(agg_result_file, index_col=0).set_index('SEQID')
        self.uv_df.columns = [x.replace('_uv', '') for x in self.uv_df.columns]
        self.ecl_oligo_df = pd.read_csv('./data/uv_melt/ECLTables/ECLOligos.csv', index_col=0)    

        # Literature
        # SL parameters
        self.santa_lucia = fileio.read_santalucia_df('./data/literature/SantaLucia.tsv')
        
        # Tm validation dataset from the literature
        self.ov_df = fileio.read_Oliveira_df('./data/literature/Oliveira_2020_mismatches.csv')
        self.lit_uv_df = pd.read_csv('./data/literature/compiled_DNA_Tm_348oligos.csv', index_col=0)
        self.lit_uv_df['RefSeq'] = self.lit_uv_df['RefSeq'].apply(eval)
        self.ov_data_split = fileio.read_json('./data/models/raw/data_split_Oliveira.json')
        self.lit_uv_data_split = fileio.read_json('./data/models/raw/data_split_348oligos.json')

        # Annotation
        self.annotation = pd.read_table(
            './data/annotation/NNNlib2b_annotation_2024_duplicates_dropped.tsv', index_col=0)
        
    @staticmethod
    def get_df_by_split(df, data_split_dict, data_split:str):
        """Helper function """
        myind = [x for x in data_split_dict[data_split+'_ind'] if x in df.index]
        return df.loc[myind,:]
    
    def prepare_val_df(self, test_mode=None):
        """
        Args:
            test_mode - 'val' or 'test'
        """
        if test_mode is None:
            test_mode = self.config['test_mode']
            
        arr_val_df = self.get_df_by_split(self.arr_adj, self.data_split_dict, data_split=test_mode)
        lit_uv_val_df = self.get_df_by_split(self.lit_uv_df, self.lit_uv_data_split, data_split=test_mode)
        ov_val_df = self.get_df_by_split(self.ov_df, self.ov_data_split, data_split=test_mode)
        
        return dict(arr=arr_val_df, lit_uv=lit_uv_val_df, ov=ov_val_df)        
    
    def get_arr_1M_with_mfe_struct(self, struct_pred_param_file):
        preprocessed_arr_1M_file = struct_pred_param_file.replace('.json', '_arr_1M.csv')
        if os.path.isfile(preprocessed_arr_1M_file):
            return pd.read_csv(preprocessed_arr_1M_file, index_col=0)
        else:
            arr_1M = pd.read_csv('./data/models/processed/arr_v1_1M_n=27732.csv', index_col=0)
            for i,row in tqdm(arr_1M.iterrows()):
                seq = row.RefSeq
                arr_1M.loc[i, 'TargetStruct'] = util.get_mfe_struct(seq, param_set=struct_pred_param_file)
                
            arr_1M.to_csv(preprocessed_arr_1M_file)
            return arr_1M
        
    @property
    def arr_1M(self):
        try:
            if self.config['secondary_struct'] == 'mfe':
                arr_1M = self.get_arr_1M_with_mfe_struct(self.config['struct_pred_param_file'])
                arr_1M['fluor_dist'] = arr_1M.TargetStruct.apply(util.get_fluor_distance_from_structure)
                arr_1M = arr_1M.query('fluor_dist == 0')
            elif self.config['secondary_struct'] == 'target':
                arr_1M = pd.read_csv('./data/models/processed/arr_v1_1M_n=27730.csv', index_col=0)
        except:
            print('Using default TargetStruct arr_1M file')
            self.config['secondary_struct'] = 'target'
            arr_1M = pd.read_csv('./data/models/processed/arr_v1_1M_n=27732.csv', index_col=0)
        
        return arr_1M
        
    @property
    def data_split_dict(self):
        data_split_dict = fileio.read_json('./data/models/raw/data_split.json')
        return data_split_dict        
        
    
def train(config):
    """
    Logs training error
    Returns `lr_dict`
    """
    wandb.init()
    mydata = MyData(config)
    
    ### Extract Features ###
    if config['feature_method'] == 'get_feature_list':
        feature_style = 'nnn'
        feature_kwargs = dict(symmetry=config['symmetry'], 
                              sep_base_stack=True, 
                              hairpin_mm=False, 
                              ignore_base_stack=False,
                              stack_size=config['stack_size'])
    elif config['feature_method'] == 'get_nupack_feature_list':
        feature_style = 'nupack'
        feature_kwargs = dict(
            directly_fit_3_4_hairpin_loop=False,
        )
    else:
        raise "Check config['feature_method']"
        
    # Shared `feature_kwargs`
    feature_kwargs.update(dict(fit_intercept=config['fit_intercept']))
    
    feats = mf.get_feature_count_matrix(mydata.arr_1M, 
                                        feature_method=config['feature_method'],
                                        feature_style=feature_style, 
                                        **feature_kwargs,
                                        )
    print('feats', feats.shape, feats.columns[:3])

    ### Train model ###
    if config['fix_some_coef']:
        """ Fixed parameters during training """
        fixed_coef_df, fixed_feature_names = mupack.get_fixed_params(
            param_set_template_file=mydata.param_set_template_file, 
            fixed_pclass=config['fixed_pclass'])
        fixed_feature_names = [x for x in fixed_feature_names if x in feats.columns]
        print('fixed_feature_names:', fixed_feature_names)
        fix_coef_kwargs=dict(
            fixed_feature_names=fixed_feature_names,
            coef_df=None, # have to set to None here cause we will set it to either dH or dG later :( 
        )
        wandb.log(dict(n_fixed_feat=len(fixed_feature_names)))
    else:
        fix_coef_kwargs=dict()
        wandb.log(dict(n_fixed_feat=0))
        
    train_kwargs = dict(
        feats=feats,
        train_only=True, 
        method=config['fit_method'],
        use_train_set_ratio=config['use_train_set_ratio'],
        fix_some_coef=config['fix_some_coef'],
        fix_coef_kwargs=fix_coef_kwargs
    )
    
    lr_dict = dict(dH=None, dG=None)
    # I have to manually select the col in coef_df because how it's structured :(
    param_name_dict = dict(dH='dH', dG='dG_37')
    
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    for i,param in enumerate(param_name_dict):
        train_kwargs['fix_coef_kwargs']['coef_df'] = fixed_coef_df[[param]]
        lr_dict[param] = mf.fit_param(mydata.arr_1M, mydata.data_split_dict, 
                                      param=param_name_dict[param],
                                      **train_kwargs
                                    )
        
    wandb.log(dict(
        n_feat=feats.shape[1],
        train_dH_mae=lr_dict['dH'].metrics['mae'],
        train_dG_mae=lr_dict['dG'].metrics['mae'],
        train_dH_rsqr=lr_dict['dH'].metrics['rsqr'],
        train_dG_rsqr=lr_dict['dG'].metrics['rsqr'],
    ))
    
    ### Save to json ###
    if feature_style == 'nupack':
        param_set_file = './models/%s.json' % wandb.run.name

        mupack.lr_dict_2_nupack_json(
            lr_dict, 
            template_file=mydata.param_set_template_file, 
            out_file=param_set_file,                  
            lr_step='full', 
            center_new_parameters=True,
            comment='wandb')
        wandb.log(dict(param_set_file=param_set_file))
    
        if config['fit_intercept']:
            wandb.log(dict(
                dH_intercept = lr_dict['dH'].coef_df.loc['intercept#intercept', :].values[0],
                dG_intercept = lr_dict['dH'].coef_df.loc['intercept#intercept', :].values[0],
            ))
    
    return lr_dict


def test(config, lr_dict=None, json_file=None, 
         debug=False, log_wandb=True, 
         save_val_result_df=True, save_metric_json=True):
    """
    Val or Test
    """
    ### Validation or test data ###
    mydata = MyData(config)
    mydata.load_everything()
    val_df_dict = mydata.prepare_val_df()
    val_result_df_dict = {k: None for k in val_df_dict}
    
    if debug:
        val_df_dict = {
            'arr': val_df_dict['arr'].sample(5),
            'lit_uv': val_df_dict['lit_uv'].sample(5),
            'ov' : val_df_dict['ov'].sample(5)}
    
    ### Run prediction ###
    if config['use_model_from'] == 'lr_dict':
        ### LinearRegression object ###
        assert lr_dict is not None
        
        for dataset_name, val_df in val_df_dict.items():
            # Special validation settings for duplexes
            model_kwargs = {key: config[key] for key in 
                                  ['feature_method', 'fit_intercept', 'symmetry', 'sep_base_stack']}
            model_kwargs['lr_dict'] = lr_dict
            if dataset_name in {'lit_uv', 'ov'}:
                model_kwargs.update({'DNA_conc': val_df['DNA_conc'].values})
                val_kwargs = dict(
                    sodium = 'varied',
                    model_kwargs=model_kwargs
                )
            else:
                val_kwargs = dict(sodium=0.088,
                                  model_kwargs=model_kwargs)
            
            # Actually run
            val_result_df_dict[dataset_name] = modeling.make_model_validation_df(
                val_df,
                model='linear_regression', 
                **val_kwargs
            )
        
        
    elif config['use_model_from'] == 'json':
        ### NUPACK model ###
        assert isinstance(json_file, str)
        
        for dataset_name, val_df in val_df_dict.items():
            # Special validation settings for duplexes
            if dataset_name in {'lit_uv', 'ov'}:
                val_kwargs = dict(
                    sodium = 'varied',
                    model_kwargs={'DNA_conc': val_df['DNA_conc'].values}
                )
            else:
                # val_kwargs = dict(model_kwargs={'ensemble':True})
                val_kwargs = dict()
            
            # Actually run
            val_result_df_dict[dataset_name] = modeling.make_model_validation_df(
                val_df,
                model='nupack', 
                model_param_file=json_file,
                **val_kwargs
            )
            
    val_result_df = pd.concat(
        val_result_df_dict,
        axis=0,
        join='outer',
        keys=val_result_df_dict.keys()
    )
        
    ### Evaluate val_result_df ###
    metric_dict = dict()
    
    # Only Tm is available for all 3 datasets
    metric_dict['all'] = dict(Tm=modeling.get_metric_dict(val_result_df, 'Tm'))
    # Tm by dataset, plus dH dG for arr
    for k,v in val_result_df_dict.items():
        metric_dict[k] = dict(Tm=modeling.get_metric_dict(v, 'Tm'))
        
        if k == 'arr':
            metric_dict[k].update(dict(
                dH=modeling.get_metric_dict(v, 'dH'),
                dG=modeling.get_metric_dict(v, 'dG_37'),
            ))

        
    if save_val_result_df:
        try:
            val_result_fn = wandb.run.name + '_val_result_df.csv'
        except:
            if json_file is not None:
                val_result_fn = json_file.replace('.json', '_val_result_df.csv')
            else:
                val_result_fn = 'insert_name_here_val_result_df.csv'

        val_result_fn = os.path.join('./models/', val_result_fn)    
        try:
            val_result_df.to_csv(val_result_fn)
            print('val_result_df saved to %s' % val_result_fn)
        except:
            return val_result_df
    
    if save_metric_json:
        try:
            metric_fn = wandb.run.name + '_metrics.json'
        except:
            metric_fn = 'some_test_run_metrics.json'
        metric_fn = os.path.join('./models/', metric_fn)
            
        fileio.write_json(
            dict(
                config=dict(config),
                metrics=metric_dict), 
            metric_fn)
        
    ### Log ###
    if log_wandb:
        try:
            wandb.log(flatten_metric_dict(metric_dict))
        except:
            print('Must call `wandb.init()` first')
            return metric_dict
    else:
        return metric_dict
    

def flatten_metric_dict(metric_dict):
    """Helper function for wandb logging"""
    flat_dict = dict()
    for dataset_name in metric_dict.keys():
        flat_dict.update(
            {f'val_Tm_{m}-{dataset_name}' : v for m,v in metric_dict[dataset_name]['Tm'].items()}
        )
        if dataset_name == 'arr':
            for p in ['dH', 'dG']:
                flat_dict.update(
                    {f'val_{p}_{m}-{dataset_name}' : v for m,v in metric_dict[dataset_name][p].items()}
                )
    return flat_dict
    