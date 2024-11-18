# nnn_paper

Code for the paper "High-Throughput DNA melt measurements enable improved models of DNA folding thermodynamics" https://doi.org/10.1101/2024.01.08.574731.
"NNN" stands for "Not-Nearest-Neighbor"

## Figures

Jupyter notebooks 01.1 to 01.5 correspond to the 5 main figures.
Notebook `01.0_DataPrep.ipynb` performs data cleaning and train-val-test split from the output of the preprocessing pipeline.

## `nnn`

Functions used for generation of figures are defined in `nnn/`. 

## Setting up environments

### Conda environments

Three major conda environments were used:

    - `nnn.yml` most analysis in the repository
    
    - `torch.yml` for training and running graph neural networks
    
    - `nn_train.yml` for fitting and running linear regression models.
      Also available as a singularity container as defined in `nn_train.def`.
      
To install, make sure `conda` is already installed, then run `conda create -f {path/to/yml/file/name}`. For example, `conda create -f envs/nnn.yml`.

The local conda environment `nnn` is also directly exported to `envs/nnn_environment.yml`. It is provide for record keeping, and `envs/nnn.yml` is still recommended for installing from scratch.

### RiboGraphViz

Package `RiboGraphViz` needs to be installed from cloned github repository, as directed at https://github.com/DasLab/RiboGraphViz. 

### Installing NUPACK4

NUPACK4 (v4.0.0.27) was manually installed from file as it requires a free liscence for download (https://docs.nupack.org/). As of Oct. 2024, NUPACK is temporarily free for academic personal users but may need paid subscription in the future.

1. Register a new account to get an academic liscence. Verify your email.
2. After logging in and acknowledging the liscence, you will find download links at https://www.nupack.org/download/software. Click `NUPACK 4.0.0.28` to download the zip file.
3. Unzip the zip file. Go to the directory `nupack-4.0.0.28/package`.
4. Choose one of the four `cp38` wheel files depending on your operating system. For example, `nupack-4.0.0.28-cp38-cp38-macosx_10_13_x86_64.whl` on macos.
5. Run `conda activate nnn` to activate the `nnn` environment.
6. Run `pip install {path/to/your/nupack/whl/file}` to install the NUPACK python module.
    
## Scripts for Library design

Python scripts in `scripts/` generates the sequences in the variant library and are helpful to understand library design logics.

## Graph Neural Networks

Run `gnn_run.py` in `torch` envoronment, pointing to the path of the saved model state dict file.

For any questions, contact
Yuxi Ke (kyx@stanford.edu).

Jan. 2024, updated Oct. 2024