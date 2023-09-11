#!/bin/env bash
#SBATCH -A NAISS2023-22-752          # find your project with the "projinfo" command
#SBATCH -p alvis                    # what partition to use (usually not needed)
#SBATCH -t 0-03:00:00                # how long time it will take to run
#SBATCH --gpus-per-node=A40:1        # choosing no. GPUs and their type
#SBATCH -J parameter_sweep             # the jobname (not needed)
# #SBATCH -o SOME_FILENAME.out        # name of the output file

# Load modules
ml purge
ml load PyTorch/1.9.0-fosscuda-2020b
ml load SciPy-bundle/2020.11-fosscuda-2020b
ml load scikit-learn/0.23.2-fosscuda-2020b
ml load matplotlib/3.3.3-fosscuda-2020b
ml load h5py/3.1.0-fosscuda-2020b
ml load IPython/7.18.1-GCCcore-10.2.0
ml load JupyterLab/2.2.8-GCCcore-10.2.0
ml load Pillow/8.0.1-GCCcore-10.2.0
ml load plotly.py/4.14.3-GCCcore-10.2.0

# OPTIONAL: Activate your custom virtual environment
source /mimer/NOBACKUP/groups/snic2022-22-552/skall/transformers/bin/activate

# Interactive
#jupyter lab                                # uncomment if you desire jupyter lab

# or you can instead use
#jupyter notebook                            # launches jupyter notebook

# Non-interactive
#ipython -c "%run /cephyr/users/skall/Alvis/Ecotoxformer/scripts/k-fold-crossvalidation-01.ipynb"      # if you do not want an interactive notebook

# or you can instead use
jupyter nbconvert --to python /cephyr/users/skall/Alvis/TRIDENT/development/kfold_crossvalidation_sweep_combined_model.ipynb &&
python /cephyr/users/skall/Alvis/TRIDENT/development/kfold_crossvalidation_sweep_combined_model.py                    # if you do not want an interactive
