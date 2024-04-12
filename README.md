![TRIDENT](trident-logo.svg)
# 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains all code and data necessary to replicate the results presented in the publication [Transformers enable accurate prediction of acute and chronic chemical toxicity in aquatic organisms](https://doi.org/10.1126/sciadv.adk6669).

## How to Use
To use the best trained models, refer to: [TRIDENT](https://trident.serve.scilifelab.se/)

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

## Cite our models
When using any of our models, please cite us!
@article{
doi:10.1126/sciadv.adk6669,
author = {Mikael Gustavsson  and Styrbjörn Käll  and Patrik Svedberg  and Juan S. Inda-Diaz  and Sverker Molander  and Jessica Coria  and Thomas Backhaus  and Erik Kristiansson },
title = {Transformers enable accurate prediction of acute and chronic chemical toxicity in aquatic organisms},
journal = {Science Advances},
volume = {10},
number = {10},
pages = {eadk6669},
year = {2024},
doi = {10.1126/sciadv.adk6669},
URL = {https://www.science.org/doi/abs/10.1126/sciadv.adk6669}}
