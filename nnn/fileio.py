import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json
import pickle

from . import util

def to_csv(df, path):
    """
    https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv
    """
    # Prepend dtypes to the top of df
    df2 = df.copy()
    df2.loc[-1] = df2.dtypes
    df2.index = df2.index + 1
    df2.sort_index(inplace=True)
    # Then save it to a csv
    df2.to_csv(path, index=False)

def read_csv(path):
    # Read types first line of csv
    dtypes = {key:value for (key,value) in pd.read_csv(path,    
              nrows=1).iloc[0].to_dict().items() if 'date' not in value}

    parse_dates = [key for (key,value) in pd.read_csv(path, 
                   nrows=1).iloc[0].to_dict().items() if 'date' in value]
    # Read the rest of the lines with the types from above
    return pd.read_csv(path, dtype=dtypes, parse_dates=parse_dates, skiprows=[1])


def read_santalucia_df(santalucia_file):
    santa_lucia = pd.read_csv(santalucia_file, sep='\t')
    santa_lucia['motif'] = santa_lucia['motif_paper'].apply(util.convert_santalucia_motif_representation)
    santa_lucia.drop(columns='motif_paper', inplace=True)
    
    return santa_lucia

def read_fitted_variant(filename, filter=True, annotation=None,
                        add_chisq_test=True, sodium=0.063, verbose=True):
    """
    Overwrites salt correction in the annotation df with sodium conc
    Args:
        annotation - df, if given, merge onto fitted variant file
        filter - Bool
    """
    df = pd.read_csv(filename, sep='\t').set_index('SEQID')
    if 'chisquared_all_clusters' in df.columns:
        df.rename(columns={'chisquared_all_clusters': 'chisq'}, inplace=True)
    df.rename(columns={s: s.replace('_final', '') for s in df.columns if s.endswith('_final')}, inplace=True)

    # Add dG and chi2 for old versions
    if not 'dG_37' in df.columns:
        df = util.add_dG_37(df)
    if add_chisq_test:
        pass
        # df = util.add_chisq_test(df)

    # Change all temperatures into celsius to avoid headaches later
    for col in ['Tm', 'Tm_lb', 'Tm_ub']:
        df[col] -= 273.15

    # Disambiguate standard error columns for params
    for c in df.columns:
        if (c.endswith('_std') and (not c.endswith('_norm_std'))):
            df.rename(columns={c: c.replace('_std', '_se')}, inplace=True)

    # Filter variants
    if filter:
        variant_filter = "n_clusters > 5 & dG_37_se < 2 & Tm_se < 25 & dH_se < 25 & RMSE < 0.5"
        df = util.filter_variant_table(df, variant_filter, verbose=verbose)

    # Optionally join annotation
    if annotation is not None:
        df = df.join(annotation, how='left')
        df['GC'] = df.RefSeq.apply(util.get_GC_content)
        df['Tm_NUPACK_salt_corrected'] = df.apply(lambda row: util.get_Na_adjusted_Tm(Tm=row.Tm_NUPACK, dH=row.dH_NUPACK, GC=row.GC, Na=sodium), axis=1)
        df['dG_37_NUPACK_salt_corrected'] = df.apply(lambda row: util.get_dG(dH=row.dH_NUPACK, Tm=row.Tm_NUPACK_salt_corrected, celsius=37), axis=1)
        
        for param in ['dH', 'dS']:
            df[param+'_NUPACK_salt_corrected'] = df[param+'_NUPACK']

    return df


def read_annotation(annotation_file, mastertable_file=None, sodium=None):
    """
    Older version required giving mastertable and merging. 
    Latest version simply reads in the annotation file.
    Returns:
        annotation - df indexed on 'SEQID' with construct class information
    """
    annotation = pd.read_csv(annotation_file, sep='\t').set_index('SEQID')

    # Rename for deprecated versions, does not affect new version
    annotation.rename(columns={c: c+'_NUPACK' for c in ['dH', 'dS', 'Tm', 'dG_37C']}, inplace=True)
    annotation.rename(columns={'dG_37C_NUPACK': 'dG_37_NUPACK'}, inplace=True)

    if mastertable_file is not None:
        mastertable = pd.read_csv(mastertable_file, sep='\t')
        annotation = annotation.reset_index().merge(mastertable[['Series', 'ConstructClass']], on='Series', how='left').set_index('SEQID')

    if sodium is not None:
        annotation['GC'] = annotation.RefSeq.apply(util.get_GC_content)
        annotation['Tm_NUPACK_salt_corrected'] = annotation.apply(lambda row: util.get_Na_adjusted_Tm(Tm=row.Tm_NUPACK, dH=row.dH_NUPACK, GC=row.GC, Na=sodium), axis=1)
        annotation['dG_37_NUPACK_salt_corrected'] = annotation.apply(lambda row: util.get_dG(dH=row.dH_NUPACK, Tm=row.Tm_NUPACK_salt_corrected, celsius=37), axis=1)
        
        for param in ['dH', 'dS']:
            annotation[param+'_NUPACK_salt_corrected'] = annotation[param+'_NUPACK']


    return annotation


def read_melt_file(melt_file):
    """
    Args:
        melt_file - str
    Returns:
        long-form dataframe, index not set
    """
    df = pd.read_csv(melt_file, header=1)
    melt = pd.DataFrame(data=df.values[:,:2], columns=['Temperature_C', 'Abs'])
    melt['ramp'] = 'melt'
    anneal = pd.DataFrame(data=df.values[:,2:4], columns=['Temperature_C', 'Abs'])
    anneal['ramp'] = 'anneal'
    
    return pd.concat((melt, anneal), axis=0)
    
    
def read_ml_data(datadir, append_2_arr=False):
    """
    read arr df and train val test split for ML 
    """
    arr = pd.read_csv(os.path.join(datadir, 'arr.csv'), index_col=0)
    with open(os.path.join(datadir, 'train_val_test_split.json'), 'r') as fh:
        data_split_dict = json.load(fh)
        
    if append_2_arr:
        arr['data_split'] = ''
        arr.loc[data_split_dict['train_ind'], 'data_split'] = 'train'
        arr.loc[data_split_dict['val_ind'], 'data_split'] = 'val'
        arr.loc[data_split_dict['test_ind'], 'data_split'] = 'test'
        
    return arr, data_split_dict
    
def read_json(fn):
    with open(fn, 'r') as fh:
        json_dict = json.load(fh)
    return json_dict

def write_json(object, fn):
    with open(fn, 'w') as fh:
        json.dump(object, fh, indent=4)

    
def read_pickle(fn):
    with open(fn, 'rb') as fh:
        pickle_dict = pickle.load(fh)
    return pickle_dict
    
def write_pickle(object, fn):
    with open(fn, 'wb') as fh:
        pickle.dump(object, fh)
        
        
def read_Oliveira_df(csv_file):
    """
    50mM NaCl, 10mM sodium phosphate, pH 7.4
        7.5 mM Na2 + 2.5 mM Na = 17.5 mM Na+
        17.5 mM + 50 mM = 67.5 mM Na+
    oligo concentration 1 uM 
    """
    def center_2_5p_seq(x):
        return f'CGACGTGC{x[:3]}ATGTGCTG'
        
    def center_2_3p_seq(x):
        return f'CAGCACAT{x[-3:][::-1]}GCACGTCG'
        
    def center_2_targetstruct(center):
        # Note seq2 is 3' to 5'
        seq1, seq2 = center.split('/')
        targetstruct = [['(']*3, [')']*3]
        for i in range(3):
            if seq1[i] != util.rcompliment(seq2[i]):
                targetstruct[0][i] = '.'
                targetstruct[1][-1-i] = '.'
        
        targetstruct = '('*8 + ''.join(targetstruct[0]) + '('*8 + '+' + ')'*8 + ''.join(targetstruct[1]) + ')'*8
        return targetstruct
        
    center_df = pd.read_csv(csv_file)
    center_df['a'] = center_df.center.apply(center_2_5p_seq)
    center_df['b'] = center_df.center.apply(center_2_3p_seq)
    center_df['RefSeq'] = center_df.apply(lambda row: [row.a, row.b], axis=1)
    center_df['TargetStruct'] = center_df.center.apply(center_2_targetstruct)
    center_df['SEQID'] = ['OV%d'%x for x in np.arange(len(center_df))]
    center_df['sodium'] = 0.0675
    center_df['DNA_conc'] = 1e-6
    
    return center_df.set_index('SEQID')[['a', 'b', 'center', 'sodium', 'DNA_conc', 'RefSeq', 'TargetStruct', 'Tm']]
    
def clean_uv_df(uv_df, ecl_oligo_df, annotation):
    def fill_hp_struct(row):
        if not isinstance(row['TargetStruct'], str):
            return util.get_symmetric_struct(len(row['RefSeq']), len_loop=4)
        else:
            return row['TargetStruct']

    uv_df = uv_df.join(ecl_oligo_df[['sequence']]).join(annotation[['RefSeq', 'TargetStruct']])

    uv_df['RefSeq'].fillna(uv_df['sequence'], inplace=True)
    uv_df.drop(columns=['sequence'], inplace=True)

    uv_df = uv_df[['Na_mM', 'conc_uM', 'RefSeq', 'TargetStruct', 'dH', 'Tm', 'dG_37', 'dS']]
    uv_df['TargetStruct'] = uv_df.apply(fill_hp_struct, axis=1)

    uv_agg_df = uv_df.reset_index(names=['SEQID']).groupby(['SEQID', 'Na_mM', 'conc_uM']).apply(np.mean)
    uv_agg_df = uv_agg_df.drop(columns=['Na_mM', 'conc_uM']).reset_index().set_index('SEQID').join(uv_df[['RefSeq', 'TargetStruct']])
    uv_agg_df = uv_agg_df.drop_duplicates()
    uv_agg_df = uv_agg_df.rename(columns=dict(Na_mM='sodium', conc_uM='DNA_conc'))
    uv_agg_df.sodium *= 1e-3
    uv_agg_df.DNA_conc *= 1e-6
    return uv_agg_df
    
def read_val_df(split='val', datadir='./data'):
    """
    Combine datasets. Use adjusted array data at low salt concentration.
    Params:
        split - str, {'train', 'val', 'test', 'all'}
    """
    join_path = lambda x: os.path.join(datadir, x)
    
    arr_df = pd.read_csv(join_path('models/processed/arr_v1_adjusted_n=27732.csv'), index_col=0)
    uv_df = pd.read_csv(join_path('models/raw/uv_n=19.csv'), index_col=0) # All validation no test
    center_df = read_Oliveira_df(join_path('literature/Oliveira_2020_mismatches.csv'))
    oligos348_df = pd.read_csv(join_path('literature/compiled_DNA_Tm_348oligos.csv'), index_col=0)
    
    arr_split_dict = read_json(join_path('models/raw/data_split.json'))
    uv_split_dict = read_json(join_path('models/raw/data_split_uv.json'))
    center_split_dict = read_json(join_path('models/raw/data_split_Oliveira.json'))
    oligo_split_dict = read_json(join_path('models/raw/data_split_348oligos.json'))
    
    arr_df['sodium'] = 0.063
    
    if split == 'all':
        val_df = pd.concat(
            (arr_df,
            uv_df,
            center_df,
            oligos348_df,
            ),
            keys=['arr', 'uv', 'ov', 'lit_uv'],
            names=['dataset', 'SEQID'],
            axis=0
        )
        split_dict_list = [arr_split_dict, uv_split_dict, center_split_dict, oligo_split_dict]
        combined_split_dict = {key+'_ind': sum([mydict[key+'_ind'] for mydict in split_dict_list], [])
                               for key in ('train', 'val', 'test')}
        return val_df[['RefSeq', 'TargetStruct', 'sodium', 'DNA_conc', 'dH', 'Tm', 'dG_37']], combined_split_dict
    else:
        val_df = pd.concat(
            (arr_df.loc[arr_split_dict[split+'_ind']],
            uv_df.loc[uv_split_dict[split+'_ind']],
            center_df.loc[center_split_dict[split+'_ind']],
            oligos348_df.loc[oligo_split_dict[split+'_ind']],
            ),
            keys=['arr', 'uv', 'ov', 'lit_uv'],
            names=['dataset', 'SEQID'],
            axis=0
        )
    
        return val_df[['RefSeq', 'TargetStruct', 'sodium', 'DNA_conc', 'dH', 'Tm', 'dG_37']]
        
def load_val_df(filename):
    def gnn_format_refseq(x):
        if isinstance(x, list):
            return ''.join(x)
        elif '[' in x:
            return ''.join(eval(x))
        else:
            return x
            
    df = pd.read_csv(filename).set_index('SEQID')
    df = df.sort_index()
    df.RefSeq = df.RefSeq.apply(gnn_format_refseq)
    return df