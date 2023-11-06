from inspect import stack
from itertools import count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder
from ipynb.draw import draw_struct
from nnn import util

##### Helper Functions #####
def sort_stack(stack):
    return '_'.join(sorted(stack.split('_')))
    
    
def sort_stack_full(stack, sep='+'):
    seq, struct = stack.split('_')
    return sep.join(sorted(seq.split(sep))) + '_' + struct
    
    
def plot_elements(feats, ax=None):
    """
    Args:
        feats - A list of features with struct
    """
    if ax is None:
        fig, ax = plt.subplots(1, len(feats), figsize=(1.8*len(feats),1.8))
        
    for i, feat in enumerate(feats):
        seq, struct = feat.split('_')
        seq = seq.replace('x', 'N').replace('y', 'N').replace('+', ' ')
        # print(seq, struct)
        draw_struct(seq, struct, ax=ax[i])
    
    
##### Feature Extractors #####

def get_stack_feature_list(row, stack_size=1):

    pad = min(stack_size - 1, 1)
    # print('pad:', pad)
    seq = 'x'*pad+row['RefSeq']+'y'*pad
    struct = '('*pad+row['TargetStruct']+')'*pad # has one more stack at the end

    loops = LoopExtruder(seq, struct, neighbor_bps=stack_size-1)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    return loops_cleaned+stacks_cleaned

def get_mismatch_stack_feature_list(row, stack_size=1, fit_intercept=False, symmetry=False):
    """
    Assumes the mistmatch pair to be paired when extracting features
    Stacks only
    """
    pad = min(stack_size - 1, 1)
    seq = 'x'*pad+row['RefSeq']+'y'*pad
    struct = '('*pad + util.get_symmetric_struct(len(row['RefSeq']), 4) + ')'*pad # has one more stack at the end
    # loops = LoopExtruder(seq, struct, neighbor_bps=stack_size-1)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)

    # loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks]
    if symmetry:
        stacks_cleaned = [sort_stack(x) for x in stacks_cleaned]

    # return loops_cleaned + stacks_cleaned
    # return stacks_cleaned
    if fit_intercept:
        return stacks_cleaned + ['intercept']
    else:
        return stacks_cleaned


def get_stack_feature_list_A(row, stack_size=1):
    """
    Considers the 'A's at the end of the construct.
    Not recommended.
    """
    seq = 'A'*(stack_size-1)+row['RefSeq']+'A'*(stack_size-1)
    struct = '('*(stack_size-1)+row['TargetStruct']+')'*(stack_size-1) # has one more stack at the end
    
    loops = LoopExtruder(seq, struct, neighbor_bps=stack_size-1)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    return loops_cleaned+stacks_cleaned


def get_stack_feature_list_simple_loop(row, stack_size=2, loop_base_size=0,
                                       fit_intercept=False, symmetry=False):
    """
    Args:
        loop_base_size - int, #stacks at the base of the loop to consider
        symmetry - bool, if set to True, view 2 symmetric motifs as the same
    """
        
    pad = stack_size - 1
    seq = 'x'*pad+row['RefSeq']+'y'*pad
    struct = '('*pad+row['TargetStruct']+')'*pad # has one more stack at the end

    loops = LoopExtruder(seq, struct, neighbor_bps=loop_base_size)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    if symmetry:
        stacks_cleaned = [sort_stack(x) for x in stacks_cleaned]
    
    if fit_intercept:
        return loops_cleaned + stacks_cleaned + ['intercept']
    else:
        return loops_cleaned + stacks_cleaned


def get_hairpin_loop_feature_list(row, stack_size=2, loop_base_size=0, 
                                  fit_intercept=False, symmetry=False,
                                  count_scaffold_list=None):
    """
    Extract features for hairpin loops.
    Use scaffold as a feature instead of stem WC stacks if `count_scaffold_list` is set
    Only count scaffold that is in `count_scaffold_list`
    """
    seq = row['RefSeq']
    struct = row['TargetStruct']
    scaffold = row['scaffold']
    
    loops = LoopExtruder(seq, struct, neighbor_bps=loop_base_size)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    loops_cleaned = [x.split(',')[0].replace(' ','_') for x in loops]
    stacks_cleaned = [x.split(',')[0].replace(' ','_') for x in stacks[:-1]]
    
    if symmetry:
        stacks_cleaned = [sort_stack(x) for x in stacks_cleaned]
    feature_list = loops_cleaned
    
    if count_scaffold_list is not None:
        if scaffold in count_scaffold_list:
            feature_list += [scaffold]
        
    if fit_intercept:
        feature_list += ['intercept']
        
    return feature_list
    
    
def get_feature_list(row, stack_size:int=2, sep_base_stack:bool=False, hairpin_mm:bool=False,
                     fit_intercept:bool=False, symmetry:bool=False, ignore_base_stack:bool=False):
    """
    Keep dot bracket in the feature to account for bulges and mismatches etc.
    Args:
        loop_base_size - int, #stacks at the base of the loop to consider. Fixed to 1
        sep_base_stack - bool, whether to separate base stack of hairpin loops to save parameters
        hairpin_mm - bool, if True, add hairpin mismatch parameters. only use when `sep_base_stack = True`
        symmetry - bool, if set to True, view 2 symmetric motifs as the same
    """
    def clean(x):
        cleaned = x.replace(' ','+').replace(',', '_')
        if cleaned[1] == 'y' and cleaned.split('_')[1] == '((+))':
            # Flipped terminal
            cleaned = f'x{cleaned[4]}+{cleaned[0]}y_((+))'
        return cleaned
        
    hp_pattern = re.compile(r'^\([.]+\)')
    loop_base_size = 1
    pad = stack_size - 1
    
    refseq = row['RefSeq']
    if isinstance(refseq, list):
        # Duplex
        # Placeholder 'b' for strandbreak because RGV is stupid and have trouble
        # handling multiple strands
        seq = 'x'*pad + refseq[0] + 'y'*pad + 'b' + 'x'*pad + refseq[1] + 'y'*pad
        struct = '('*pad + row['TargetStruct'].replace('+', '('*pad + '.' + ')'*pad) + ')'*pad
        
    elif isinstance(refseq, str):
        # Hairpin
        seq = 'x'*pad+refseq+'y'*pad
        struct = '('*pad+row['TargetStruct']+')'*pad # has one more stack at the end
    
    loops = LoopExtruder(seq, struct, neighbor_bps=loop_base_size)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    loops_cleaned = [clean(x) for x in loops if not ('b' in x)] # remove anything with strand break
    stacks_cleaned = [clean(x) for x in stacks]
    
    if sep_base_stack:
        for loop in loops_cleaned:
            seq, struct = loop.split('_')
            if hp_pattern.match(struct):
                # hairpin loop
                hairpin_loop = LoopExtruder(seq, struct, neighbor_bps=0)[0]
                hairpin_stack = StackExtruder(seq, struct, stack_size=1)[0]
                
                if len(seq) <= 6:
                    loops_cleaned.append(clean(hairpin_loop))
                else:
                    loops_cleaned.append('NNNNN_.....')
                                   
                if hairpin_mm:
                    loops_cleaned.append('%s%s+%s%s_(.+.)' % (seq[0], seq[1], seq[-2], seq[-1]))
                    ignore_base_stack = True
                    
                if not ignore_base_stack:
                    loops_cleaned.append(clean(hairpin_stack))
                    
                
                loops_cleaned.remove(loop)
                    
            elif struct == '(..(+)..)':
                # double mismatch
                mm = f'{seq[1:3]}+{seq[6:8]}_..+..'
                mm_stack = f'{seq[0]}+{seq[3]}+{seq[5]}+{seq[8]}_(+(+)+)'
                loops_cleaned.append(mm)
                loops_cleaned.append(mm_stack)
                loops_cleaned.remove(loop)
                
            elif struct == '(...(+)...)':
                # triple mismatch
                # very bad treatment right now, reduce to double mismatch lol
                mm = f'{seq[1]}{seq[3]}+{seq[7]}{seq[9]}_..+..'
                mm_stack = f'{seq[0]}+{seq[4]}+{seq[6]}+{seq[10]}_(+(+)+)'
                loops_cleaned.append(mm)
                loops_cleaned.append(mm_stack)
                loops_cleaned.remove(loop)
                
    if symmetry:
        stacks = [sort_stack_full(x, sep='+') for x in stacks_cleaned]
        
    try:
        if row['Series'] == 'VARloop':
            for loop in loops_cleaned:
                seq, struct = loop.split('_')
                if hp_pattern.match(struct):
                    loops_cleaned.remove(loop)
                    loops_cleaned.append('NNNNN_.....')
    except:
        pass # in case row doesn't have `Series`
    
    feature_list = loops_cleaned + stacks_cleaned
    # throw away features without sequence
    feature_list = [x for x in feature_list if '_' in x]   
    
    if fit_intercept:
        return feature_list + ['intercept']
    else:
        return feature_list


def get_nupack_feature_list(row, fit_intercept:bool=False,
                            directly_fit_3_4_hairpin_loop=True):
    """
    Parameterize the hairpins according to NUPACK parameters.
    Use NUPACK parameter terminologies in the parameter names.
    05/16/2023 Yuxi
    Args:
        directly_fit_3_4_hairpin_loop - Bool, if False, do not directly fit 
            hairpin_triloop or hairpin_tetraloop, but populate the field from 
            calculation while saving to json
    """        
    sep = '#'
    hairpin_pattern = re.compile(r'^\([.]+\)') # hairpin pattern
    loop_base_size = 1
    stack_size = 2
    pad = stack_size - 1
    seq = 'x'*pad+row['RefSeq']+'y'*pad
    struct = '('*pad+row['TargetStruct']+')'*pad # has one more terminal stack at the end
    feature_list = []

    loops = LoopExtruder(seq, struct, neighbor_bps=loop_base_size)
    stacks = StackExtruder(seq, struct, stack_size=stack_size)
    
    for loop in loops:
        seq, struct = loop.split(',')
        
        """ hairpin loops """
        if hairpin_pattern.match(struct):
            hairpin_size = len(seq) - 2
            feature_list.append('hairpin_size%s%d' % (sep, hairpin_size))
            
            if directly_fit_3_4_hairpin_loop:
                if hairpin_size == 3:
                    terminal_bp = seq[-1] + seq[0]
                    feature_list.append('hairpin_triloop%s%s' % (sep, seq))
                    feature_list.append('terminal_penalty%s%s' % (sep, terminal_bp))
                elif hairpin_size == 4:
                    feature_list.append('hairpin_tetraloop%s%s' % (sep, seq))
                elif hairpin_size > 4:
                    hairpin_mismatch = seq[-2:] + seq[:2]
                    feature_list.append('hairpin_mismatch%s%s' % (sep, hairpin_mismatch))
                else:
                    pass
            else:
                # do not fit triloop/ tetraloop parameters directly
                # but extract hairpin_mismatch parameter & hairpin loop_mid only
                # and populate the hairpin_triloop field by calculations
                hairpin_mismatch = seq[-2:] + seq[:2]
                feature_list.append('hairpin_mismatch%s%s' % (sep, hairpin_mismatch))
                if hairpin_size == 3 or hairpin_size == 4:
                    hairpin_loop_mid = seq[2:-2]
                    feature_list.append('hairpin_loop_mid%s%s' % (sep, hairpin_loop_mid))
                
                
                                
        else:
            """ internal loops """
            seq1, seq2 = seq.split(' ')
            n1 = len(seq1) - 2
            n2 = len(seq2) - 2
        
            if ((n1 == 0) or (n2 == 0)) and (n1 + n2 != 0):
                """ bulges """
                # bulges are very dumbly parameterized
                bulge_size = max(n1, n2)
                feature_list.append('bulge_size%s%d' % (sep, bulge_size))
                
                if bulge_size == 1:
                    stack = seq1[0] + seq1[-1] + seq2[0] + seq2[-1]
                    feature_list.append('stack%s%s' % (sep, stack))
                else:
                    pair1 = seq1[0] + seq2[-1]
                    pair2 = seq2[0] + seq1[-1]
                    feature_list.append('terminal_penalty%s%s' % (sep, pair1))
                    feature_list.append('terminal_penalty%s%s' % (sep, pair2))
                
            else:
                """ mismatches """
                interior_size = n1 + n2
                feature_list.append('interior_size%s%d' % (sep, interior_size))
                # no asymmetric constructs in the library
                # therefore ignoring the interior assymm parameter
                mm1 = seq2[-2:] + seq1[:2]
                mm2 = seq1[-2:] + seq2[:2]
                
                if interior_size == 2:
                    """ 1x1 mismatch """
                    # extra intermediate parameter for oppositing stack interaction
                    # only for 1x1 mismatch
                    mm_stacks = seq1[0] + seq1[-1] + seq2[0] + seq2[-1]
                    feature_list.append('interior_mismatch_stacks%s%s' % (sep, mm_stacks))
                    feature_list.append('interior_mismatch%s%s' % (sep, mm1))
                    feature_list.append('interior_mismatch%s%s' % (sep, mm2))
                elif interior_size > 2:
                    """ 2x2 mismatch """
                    feature_list.append('interior_mismatch%s%s' % (sep, mm1))
                    feature_list.append('interior_mismatch%s%s' % (sep, mm2))
                    # feature_list.append('terminal_mismatch%s%s' % (sep, mm1))
                    # feature_list.append('terminal_mismatch%s%s' % (sep, mm2))
                
            
    for stack in stacks:
        seq, struct = stack.split(',')
        if 'x' in seq or 'y' in seq:
            """ terminal stack """
            terminal_bp = seq[1] + seq[3]
            feature_list.append('terminal_penalty%s%s' % (sep, terminal_bp))
        else:
            """ stack """
            stack = seq.replace(' ', '')
            feature_list.append('stack%s%s' % (sep, stack))
                
    
    if fit_intercept:
        # make the format recognizable to reg expression
        return feature_list + ['intercept#0']
    else:
        return feature_list
        
        
def get_stem_nn_feature_list(row):
    dup_row = util.get_duplex_row(row)
    seq = dup_row.RefSeq
    # struct = dup_row.TargetStruct
    stem_len = int((len(seq) - 4.0) / 2.0)
    seq_pad = f'x{seq[:stem_len]}y+x{seq[-stem_len:]}y'
    nn_list = []
    for flag in range(stem_len + 1):
        if flag == 0:
            nn = seq_pad[:2] + '+' + seq_pad[-2:] + '_((+))'
        elif flag == stem_len:
            nn = seq_pad[-flag-2:-flag] + '+' + seq_pad[flag:flag+2] + '_((+))'
        else:
            nn = seq_pad[flag:flag+2] + '+' + seq_pad[-flag-2:-flag] + '_((+))'
        nn_list.append(nn)
        
    return nn_list
    
    
def get_partial_position_feature_list(row, seq_type:str='hairpin_loop', **kwargs):
    """
    To see the contribution of different sequence components in figure 3
    Args:
        seq_type - {'hairpin_loop', 'internal_loop', 'single_mm'}
        **kwargs - `seq_type` specific settings
            `hairpin_loop':
                loop_base_size - int
                pos_list - List[int or List[int]]. e.g. loop_base_size = 2, 
                    -2 is first base, 0 the first in the loop
            `single_mm`:

    """
    def clean(x):
        return x.replace(' ','+').replace(',', '_')

    assert seq_type in {'hairpin_loop', 'internal_loop'}
    feature_list = []
    seq, struct = row.RefSeq, row.TargetStruct
    
    if seq_type == 'hairpin_loop':
        loop_base_size = kwargs['loop_base_size']
        pos_list = kwargs['pos_list']
        
        loops = LoopExtruder(seq, struct, neighbor_bps=loop_base_size)
        loops = [clean(loop) for loop in loops]
        
        for loop in loops:
            for pos in pos_list:
                if isinstance(pos, int):
                    feat = loop[pos+loop_base_size] + str(pos)
                else:
                    feat = ''.join([loop[p+loop_base_size] for p in pos])
                    feat += ''.join([str(p) for p in pos])

                feature_list.append(feat)
    elif seq_type == 'single_mm':
        idxmm = (('('+struct).find('((.(') + 1, (struct+')').find(').))') + 1) # Tuple[int, int]
        pass 
        # TODO
        
    return feature_list
    
    
def get_rich_feature_list(row):
    """
    Rich feature list as in Zakov et al. 2011
    """
    feature_list = []
    
    return feature_list