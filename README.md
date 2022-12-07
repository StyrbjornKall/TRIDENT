# fishbAIT - fish based AI Toxicity
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![fishbAITe model architecture](final_model.svg)
## Overview
This repository contains all code and data necessary to replicate the results presented in the publication L!.

## How to Use
The models in this repository can also be used directly through our online web application:
[fishbAIT](https://fishbait.streamlit.app/)

For very extensive predictions (>100 MB) consider cloning this repo and follow the tutorials under `tutorials`.

Clone this repository:
```bash 
git clone https://github.com/StyrbjornKall/fishbAIT
```

## Dependencies

`fishbait_development.yml` and `fishbait_development.txt`
Contains all packages required to reproduce this study:
```bash
conda env create -f fishbait_development_environment.yml
```

`fishbait_inference.yml` and `fishbait_inference.txt`
Contains only the packages required for making predictions using the fine-tuned models and skips development-only packages:
```bash
conda env create -f fishbait_inference_environment.yml
```

## Layout
`data` contains all preprocessed data used for training our fine-tuned EC50, EC10 and combined model.

`development` contains all code needed to replicate the findings presented in the publication.

`fishbAIT` houses the fine-tuned model weigths.

`tutorials` contains tutorial notebooks for running inference using the fine-tuned models.
