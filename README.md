![TRIDENT](trident-logo.svg)
# 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains all code and data necessary to replicate the results presented in the publication [Transformers enable accurate prediction of acute and chronic chemical toxicity in aquatic organisms](https://www.biorxiv.org/content/10.1101/2023.04.17.537138v1) (*preprint*).

## How to Use
To replicate the study, refer to the documentation under the `development` section.

For very extensive predictions (>100 MB) consider cloning this repo and follow the tutorials under `tutorials` (requires basic python understanding).

Clone this repository:
```bash 
git clone https://github.com/StyrbjornKall/TRIDENT
```

## Dependencies
**Replicate entire study**
Contains all packages required to reproduce this study:
```bash
conda env create -f trident_development_environment.yml
```

## Layout
`data` contains all preprocessed data used for training our nine fine-tuned EC50, EC10 and combined models. Also contains QSAR comparison data. 

`development` contains all code needed to replicate the findings presented in the publication.

`TRIDENT` contains the nine fine-tuned Deep Neural Network modules for the models. For the fine-tuned transformer (RoBERTa) modules, refer to [Huggingface model-hub](https://huggingface.co/StyrbjornKall).

`tutorials` contains very simple tutorial notebooks for running inference using the fine-tuned models. Written in order to minimize programmatic interference so that very basic python knowledge suffice. 

*Refer to each sections README for further descriptions.*

## Architecture
![TRIDENT model architecture](final_model.svg)
