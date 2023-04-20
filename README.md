![ecoCAIT](ecocait-logo.svg)
# 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains all code and data necessary to replicate the results presented in the publication L!.

## How to Use
To replicate the study, refer to the documentation under the `development` section.

For very extensive predictions (>100 MB) consider cloning this repo and follow the tutorials under `tutorials` (requires basic python understanding).

Clone this repository:
```bash 
git clone https://github.com/StyrbjornKall/ecoCAIT
```

## Dependencies
**Replicate entire study**
Contains all packages required to reproduce this study:
```bash
conda env create -f ecocait_development_environment.yml
```

## Layout
`data` contains all preprocessed data used for training our nine fine-tuned EC50, EC10 and combined models. Also contains QSAR comparison data. 

`development` contains all code needed to replicate the findings presented in the publication.

`ecoCAIT` contains the nine fine-tuned Deep Neural Network modules for the models. For the fine-tuned transformer (RoBERTa) modules, refer to [Huggingface model-hub](https://huggingface.co/StyrbjornKall).

`tutorials` contains very simple tutorial notebooks for running inference using the fine-tuned models. Written in order to minimize programmatic interference so that very basic python knowledge suffice. 

*Refer to each sections README for further descriptions.*

## Architecture
![ecoCAIT model architecture](final_model.svg)
