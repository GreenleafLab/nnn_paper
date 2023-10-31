import pandas as pd
import os
import nupack
from nupack import Domain, TargetStrand, TargetComplex, TargetTube, tube_design, Diversity
from . import util

def design_nupack(target):
    """
    Args:
        target - Dict, e.g.
            target = dict(a='SN6NN6S',
                b='SN6NN6S',
                struct='(7.(7+)7.)7',
                Tm=40,
                sodium=0.088,
                ss_conc=7.2e-5,
                task_name='mismatch15',
                prefix='DMM',
                start_index=1,
                n=7)
    Returns:
        my_results
    """
    a = Domain(target['a'], name='a')
    b = Domain(target['b'], name='b')
    A = TargetStrand([a], name='strand A')
    B = TargetStrand([b], name='strand B')

    duplex = TargetComplex([A, B], target['struct'], name=target['task_name'])
    strandA = TargetComplex([A], A.nt() * '.', name='free strand A')
    strandB = TargetComplex([B], B.nt() * '.', name='free strand B')

    t1 = TargetTube(on_targets={duplex: 0.5 * target['ss_conc'],
                                strandA: 0.5 * target['ss_conc'],
                                strandB: 0.5 * target['ss_conc']}, 
                    name=target['task_name'] + '_tube')

    div1 = Diversity(word=4, types=2)
    # div2 = Diversity(word=6, types=3)

    my_model = nupack.Model(material='dna04', celsius=target['Tm'], sodium=target['sodium'], magnesium=0.0)
    my_design = tube_design(tubes=[t1],
                            soft_constraints=[],
                            defect_weights=None,
                            model=my_model,
                            hard_constraints=[div1])
    my_results = my_design.run(trials=target['n'])
    
    return my_results


def parse_nupack_result(my_result):
    """
    Hard-coded to return a List[str, str] of the designed 2 strands
    """
    if not os.path.isdir('./tmp'):
        os.mkdir('./tmp')
        
    with open('./tmp/tmp.txt', 'w+') as fh:
        fh.write(str(my_result))

    with open('./tmp/tmp.txt', 'r') as fh:
        lines = fh.readlines()
        
    strands = [line.strip()[9:-1].replace(' ','') for line in lines if line.strip().startswith('strand ')]
    return strands
    
    
def results_2_df(my_results, target):
    samples = [target['prefix']+'%03d' % i for i in 
           range(target['start_index'], target['start_index']+target['n'])]
    ind = []
    for s in samples:
        ind.append(s+'_5p')
        ind.append(s+'_3p')

    result_df = pd.DataFrame(index=ind,
                            columns=['sequence', 'struct', 'Tm', 'sodium', 'ss_conc', 'task_name'])
    
    append_col = ['struct', 'Tm', 'sodium', 'ss_conc', 'task_name']
    for i in range(target['n']):
        strands = parse_nupack_result(my_results[i])
        idnum = i + target['start_index']
        result_df.loc[target['prefix']+'%03d_5p'%(idnum), 'sequence'] = strands[0]
        result_df.loc[target['prefix']+'%03d_3p'%(idnum), 'sequence'] = strands[1]
        result_df[['dH_NUPACK', 'dS_NUPACK', 'Tm_NUPACK', 'dG_37_NUPACK']] = util.get_nupack_dH_dS_Tm_dG_37(strands, struct=target['struct'], sodium=target['sodium'])
        
    try:
        result_df['length'] = result_df.sequence.apply(len)
    except:
        print(result_df['sequence'])
    
    for c in append_col:
        result_df[c] = target[c]
        
    return result_df
    
    
def design_target(target):
    """
    Calls design_nupack etc.
    """
    my_results = design_nupack(target)
    result_df = results_2_df(my_results, target)
    return result_df
