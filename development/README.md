# `development`

Contains all the scripts used to generate the fine-tuned models presented in the publication.

## `development_utils`
Contains utility scripts for data preprocessing and model building and training.

## `figures`

Contains script for generating figures presented in the publication.

## `wandb_configurations`

Contains all weights and biases configurations necessary for running each hyperparameter/parameter sweep presented in the publication. Note that this works only using weights and biases and requires already set up directories in wandb in order to be fully functional. 

In order to reproduce the sweeps, sweeps must be configured as per the configuration files attached in this directory.

For access to the runs presented in the publication, refer to [this](https://wandb.ai/ecotoxformer) wandb repository.