#!/usr/bin/env python
# This script takes in a list of sequences (tab delimited) with the first row = sequence names, second row = sequences
import random
    
def read_file(RNAnamecheck, filename):
    seqfile = open(filename, 'r')
    RNAnamelist = []
    RNAseqlist = []
    RNAs = {}
    seqID = 0

    for line in seqfile:
        seqID += 1
        items = line.split()
        if len(items) == 2: 
            RNAname = str(items[0]) + str(seqID)
            index = 1   
        else: 
            RNAname = RNAnamecheck + str(seqID)
            index = 0
            
        RNAnamelist.append(RNAname)
        RNAseqlist.append(items[index])
        RNAs[RNAname] = items[index]         

    return [RNAnamelist, RNAs]


def RNAtoDNA(seq):
    seq = seq.replace("U", "T")
    return seq


def get_rc(seq):
    cseq = ''
    for c in seq[::-1].upper():
        if c == 'A':
            b = 'T'
        elif c == 'T':
            b = 'A'
        elif c == 'G':
            b = 'C'
        elif c == 'C':
            b = 'G'
        cseq += b
    return cseq

def rcompliment(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', '-':'-'}
    return "".join(complement.get(base, base) for base in reversed(seq))


def add_libfrag1(seq, forwardprimer):
    
    # makes library fragment with added optional forward primer 
    
    lenseq = len(seq)
    fillerseq = "AACAACAACAACATACTAACAACAACATAACAAATCAAAA"[0:-lenseq]
    newseq = forwardprimer + "GCTGTTGAAGGCTCGCGATGCACACGCTCTGGTACAAGGAA" + seq + \
        "AAGGCACTGGGCAATACGAGCTCAAGCCAGTCTCGCAGTCC" + \
        fillerseq + "AGATCGGAAGAGCGGTTCAGCAGGAA"

    return (newseq)



def extract_refseq(seq):
    """
    From order information, return RefSeq
    """
    lib5p = "GCTGTTGAAGGCTCGCGATGCACACGCTCTGGTACAAGGAA"
    lib3p = "AAGGCACTGGGCAATACGAGCTCAAGCCAGTCTCGCAGTCC"
    rev_seq = rcompliment(seq)
    start_ind = rev_seq.find(lib5p) + len(lib5p)
    end_ind = rev_seq.find(lib3p)

    return rev_seq[start_ind:end_ind]

def add_libfrag2(seq, forwardprimer):
    
    # makes library fragment with added optional forward primer
    lenseq = len(seq)
    fillerseq = "AACAACAACAACATACTAACAACAACATAACAAATCAAAA"[0:-lenseq]
    newseq = forwardprimer + "GCTGTTGAAGGCTCGCGATGCACACGCTCTGGTACAAGGAA" + seq + \
        "AAGGCGACTCCACTATAGTACCGTCGTCCGGTGGAGTCTGG" + \
        fillerseq + "AGATCGGAAGAGCGGTTCAGCAGGAA"

    return (newseq)


def make_oligolib(RNAdict):
    #adds the right fragments to the variants so they can be amplified. This is designed to add two copies of each construct to the order in case of synthesis errors. 
    
    newRNAdict = {}

    for name, RNAseq in RNAdict.items():
        DNA = RNAtoDNA(RNAseq)
        newDNA = add_libfrag1(DNA, '')
        newDNA = get_rc(newDNA)
        newRNAdict[name + '1'] = newDNA
        #newDNA = add_libfrag2(DNA, '')
        #newDNA = get_rc(newDNA)
        newRNAdict[name + '2'] = newDNA

    return newRNAdict


def printtofile(newseqdict, outfile):

    outfile = open(outfile, 'w')
    count = 0

    for seq in newseqdict.keys():
        count += 1
        line = seq + '\t' + newseqdict[seq] + '\n'
        outfile.write(line)
    
    print ('total number of sequences in library: ', count)


def sizechecker(DNAseqdict):
    # checks if the seqs are all the same length, if not print out error message

    lenseqcompare = len(random.choice(list(DNAseqdict.values())))
    countwronglen = 0
    print(lenseqcompare)
    totalcount = 0

    for seq in DNAseqdict:
        if lenseqcompare != len(DNAseqdict[seq]):
            countwronglen += 1
            print(seq)
            print(DNAseqdict[seq])
            print(len(seq))
        totalcount += 1
    print("Your library has " + str(countwronglen) +
          " constructs of the wrong length.")
    print (totalcount)


if __name__ == '__main__':

    # Read in the file of the sequences to be tested
    libinfo1 = read_file('', "../data/library/NNNlib2b.txt")
    #libinfo2 = read_file('', 'pk_point_mutations.txt')
    #libinfo3 = read_file('mismatch_3mer', '20210907_mismatch_sequences_yuxi/mismatch_1and2_3mer_scaffold-GCGC.txt')
    # libinfo1 = read_file('', 'NNN_ssfluor_ctrls.txt')
    
    # From the sequences in the file will convert any that are RNA to DNA
    # Will also add on the appropriate library compatible sequence (including filler sequence)
    DNAseqdict = make_oligolib(libinfo1[1])
    #DNAseqdictpk = make_oligolib(libinfo2[1])
    #DNAseqdictmismatches = make_oligolib(libinfo3[1])
    # Checks that everything is the same length
    sizechecker(DNAseqdict)
    #sizechecker(DNAseqdictpk)
    #sizechecker(DNAseqdictmismatches)
    
    #DNAseqdict.update(DNAseqdictpk)
    #DNAseqdict.update(DNAseqdictmismatches)
    
    # Prepares the final sequences for ordering
    printtofile(DNAseqdict, "../data/library/NNN_lib2b_order.txt")
    # printtofile(DNAseqdict, "NNN_ssfluor_order.txt")
