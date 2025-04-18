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
      
To install, make sure `conda` is already installed, then run `conda env create -f {path/to/yml/file/name}`. For example, `conda env create -f envs/nnn.yml`.

The local conda environment `nnn` is also directly exported to `envs/nnn_environment.yml`. It is provide for record keeping, and `envs/nnn.yml` is still recommended for installing from scratch.

> Note for container users: If you're running this in a minimal linux container, you may need to manually install system build tools beforehand. For example, `sudo apt install build-essential`.

### RiboGraphViz

The `RiboGraphViz` package is required for some visualization tasks.

By default, it is not directly available on PyPI or conda. To install, you can either:

**Option 1 (recommended by original authors):**  
Clone and install manually, following the instructions at  
https://github.com/DasLab/RiboGraphViz

```bash
git clone https://github.com/DasLab/RiboGraphViz.git
cd RiboGraphViz
pip install .
```

**Option 2 (alternative):**
Install directly via pip with dependencies:

```bash
pip install networkx matplotlib seaborn git+https://github.com/DasLab/RiboGraphViz
```

Note: While the RiboGraphViz developers have expressed interest in adding the package to PyPI, this has not yet been done as of this writing.

### Installing NUPACK4

NUPACK4 (v4.0.0.27) was manually installed from file as it requires a free licence for download (https://docs.nupack.org/). As of Oct. 2024, NUPACK is temporarily free for academic personal users but may need paid subscription in the future.

1. Register a new account to get an academic licence. Verify your email.
2. After logging in and acknowledging the licence, you will find download links at https://www.nupack.org/download/software. Click `NUPACK 4.0.0.28` to download the zip file.
3. Unzip the zip file. Go to the directory `nupack-4.0.0.28/package`.
4. Choose one of the four `cp38` wheel files depending on your operating system. For example, `nupack-4.0.0.28-cp38-cp38-macosx_10_13_x86_64.whl` on macos.
5. Run `conda activate nnn` to activate the `nnn` environment.
6. Run `pip install {path/to/your/nupack/whl/file}` to install the NUPACK python module.

## Running linear regression models

The parameter estimation process could be replicated by following these steps:

1.	**Prepare the environment.** Install the conda environment specified in `envs/nn_train.yml` with `conda env create -f envs/nn_train.yml`. This yaml file specifies the required packages and their versions.

    a.	Alternatively, use a singularity container. The build file of this singularity container is `envs/nn_train.def`.
2.	**Activate the environment.** `conda activate nn_train`

3.	**Run the script.** Enter `python run_nn_train.py` in the command line. You may also submit it as a job, using `run_nn_train.sh` as a template (you will need to modify the slurm settings). 
In `run_nn_train.py`, edit the `config` dictionary to run models with different settings; edit the `myrange` list to change the percentage of training data used for the plots.

    a.	Alternatively, run the notebook interactively. Launch `jupyter lab` and run the notebook `03.2_TrainNN.ipynb` in the `nnn_paper` repository.
    
    b.	Note that in either script or notebook settings, you will be prompted to login to `wandb` to log the model training runs. This helps to keep track of models trained with different settings.

    
## Scripts for Library design

Python scripts in `scripts/` generates the sequences in the variant library and are helpful to understand library design logics.

## Graph Neural Networks

Run `gnn_run.py` in `torch` envoronment, pointing to the path of the saved model state dict file.

For any questions, contact
Yuxi Ke (kyx@stanford.edu).

Jan. 2024, updated Oct. 2024