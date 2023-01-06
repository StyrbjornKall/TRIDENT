# `development`

Contains all the scripts used to generate all results and fine-tuned models presented in the publication. The notebooks used for model development depend on weights and biases (wandb) keeping track of which parameters are used for each model. This includes which data to use for training, how to split data into training and validation, which hyperparameters to use during training, modifications to the model architecture etc.. In other words, for these scripts to function properly, a wandb repository is required where the user is required to build a file tree similar to what was used during development. For a discription on this, refer to *wandb_configurations* section below.

## `development_utils`
Contains utility scripts for data preprocessing and model building and training.

## `figures`
Contains script for generating figures presented in the publication.

## `wandb_configurations`
As mentioned the development scripts depend on wandb. This folder contains all weights and biases configurations necessary for running each hyperparameter/parameter sweep presented in the publication. Note that this works only using weights and biases and requires already set up directories in wandb in order to be fully functional. 

In order to reproduce the sweeps, sweeps must be configured as per the configuration files attached in this directory.

For reference, see [this](https://wandb.ai/ecotoxformer) wandb repository, used during model development of the publication. Below are instructions on how to replicate that repository.

### Create a wandb account
Create an Weights and biases account [here](https://wandb.ai/)

This will be the basic repo in which all different development folders will be stored.

### Create wandb *Projects*
A wandb Project is basically a folder containing something specific to that project. In this publication a project is specific to a type of run (e.g. 10x10 K-fold CV for all species groups). Here are the projects according to the presented study:

1. ***chemberta_version_and_loss_function_sweep*** Will host sweeps establishing correct ChemBERTa version and loss function
2. ***K_fold_HP_sweep_EC50_fish*** Will host basyesian sweeps establishing hyperparameter and locking M<sub>50</sub> hyperparameter/parameter configuration
3. ***K_fold_HP_sweep_EC10_fish*** Will host basyesian sweeps establishing hyperparameter and locking M<sub>10</sub> hyperparameter/parameter configuration
4. ***K_fold_HP_sweep_EC50EC10_fish*** Will host basyesian sweeps establishing hyperparameter and locking M<sub>50/10</sub> hyperparameter/parameter configuration

5. ***100Fold_CV_fish*** Will host grid sweeps for 10x10 K-fold cross-validation used for model accuracy calculations for M<sub>50</sub>, M<sub>10</sub>, M<sub>50/10</sub>
6. ***100Fold_CV_invertebrates*** Will host grid sweeps for 10x10 K-fold cross-validation used for model accuracy calculations for M<sub>50</sub>, M<sub>10</sub>, M<sub>50/10</sub>
7. ***100Fold_CV_algae*** Will host grid sweeps for 10x10 K-fold cross-validation used for model accuracy calculations for M<sub>50</sub>, M<sub>10</sub>, M<sub>50/10</sub>

8. ***Final_model*** Will host runs for training the final models

### Create sweeps inside Projects using the configuration .yml files
With the wandb Projects (directories) set up, the next step is to create sweeps inside these projects using the configuration files given in `wandb_configurations` (folders are named according to the Projects they are subposed to be in). A sweep is created by clicking the broom icon inside a project and then "Create Sweep". Paste the contents of the .yaml into the prompt and click "Create Sweep".

Finally the wandb "file"-tree should follow this structure:

```
.
└── wandb_account/
    ├── chemberta_version_and_loss_function_sweep/
    │   └── Sweeps/
    │       ├── BPEtok10M.yaml
    │       └── SMILEStok1M.yaml
    ├── K_fold_HP_sweep_EC50_fish/
    │   └── Sweeps/
    │       ├── batch_size.yaml
    │       ├── 2_hidden_layers_hpsweep.yaml
    │       ├── 3_hidden_layers_hpsweep.yaml
    │       └── 4_hidden_layers_hpsweep.yaml
    ├── K_fold_HP_sweep_EC10_fish/
    │   └── Sweeps/
    │       ├── 2_hidden_layers_hpsweep.yaml
    │       ├── 3_hidden_layers_hpsweep.yaml
    │       └── 4_hidden_layers_hpsweep.yaml
    ├── K_fold_HP_sweep_EC50EC10_fish/
    │   └── Sweeps/
    │       ├── 2_hidden_layers_hpsweep.yaml
    │       ├── 3_hidden_layers_hpsweep.yaml
    │       └── 4_hidden_layers_hpsweep.yaml
    ├── 100Fold_CV_fish/
    │   └── Sweeps/
    │       ├── 100fold_CV_EC50_fish.yaml
    │       ├── 100fold_CV_EC10_fish.yaml
    │       └── 100fold_CV_EC50EC10_fish.yaml
    ├── 100Fold_CV_invertebrates/
    │   └── Sweeps/
    │       ├── 100fold_CV_EC50_invertebrates.yaml
    │       ├── 100fold_CV_EC10_invertebrates.yaml
    │       └── 100fold_CV_EC50EC10_invertebrates.yaml
    ├── 100Fold_CV_algae/
    │   └── Sweeps/
    │       ├── 100fold_CV_EC50_algae.yaml
    │       ├── 100fold_CV_EC10_algae.yaml
    │       └── 100fold_CV_EC50EC10_algae.yaml
    └── Final_model
```