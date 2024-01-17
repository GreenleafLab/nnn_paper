# nnn_paper

Code for the paper "High-Throughput DNA melt measurements enable improved models of DNA folding thermodynamics" https://doi.org/10.1101/2024.01.08.574731.
"NNN" stands for "Not-Nearest-Neighbor"

## Figures

Jupyter notebooks 01.1 to 01.5 correspond to the 5 main figures.
Notebook `01.0_DataPrep.ipynb` performs data cleaning and train-val-test split from the output of the preprocessing pipeline.

## `nnn`

Functions used for generation of figures are defined in `nnn/`. 

## Setting up environments

Three major conda environments were used:

    - `nnn.yml` most analysis in the repository
    
    - `torch.yml` for training and running graph neural networks
    
    - `nn_train.yml` for fitting and running linear regression models.
      Also available as a singularity container as defined in `nn_train.def`.
      
Packages `draw_rna` and `RiboGraphViz` were installed from file as directed on Das lab github repositories https://github.com/DasLab/draw_rna and https://github.com/DasLab/RiboGraphViz.

NUPACK4 was also manually installed from file as it requires a free liscence for download (https://docs.nupack.org/).
    
## Scripts for Library design

Python scripts in `scripts/` generates the sequences in the variant library and are helpful to understand design logics.

## Graph Neural Networks

Run `gnn_run.py` in `torch` envoronment, pointing to the path of the saved model state dict file.

For any questions, contact
Yuxi Ke (kyx@stanford.edu)

Jan. 2024