# `development`

Contains all the scripts used to generate the fine-tuned models presented in the publication.

## `figures`

Contains script for generating figures presented in the publication.

## `sweeps`

Contains all code necessary for running each hyperparameter/parameter sweep presented in the publication. Note that this works only using weights and biases and requires already set up directories in wandb in order to be fully functional. 

In order to reproduce these sweeps, sweeps must be configured as per the configuration files attached in this directory.

For access to the runs presented in the publication, refer to [this](https://wandb.ai/ecotoxformer) wandb repository.