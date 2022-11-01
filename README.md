# fishbAITe - fish based AI Toxicity engine
![fishbAITe model architecture](final_model.svg)
## Overview
This repository contains all code and data necessary to replicate the results presented in the publication L!.

The models in this repository can also be used directly through our online web application:
[fishbAITe](https://new-fishbait-app.herokuapp.com/)

For very extensive predictions (>100 MB) consider following the tutorial under `tutorials`.

## How to Use
To do

## Layout
`data` contains all preprocessed data used for training our fine-tuned EC50, EC10 and combined model.

`development` contains all code used to replicate the findings presented in the publication.

`fishbAIT` houses the fine-tuned model parameters

`tutorials` contains tutorial notebooks for running inference using the fine-tuned models.

`utils` contains necessary utility scripts in order to run both the scripts in `development` and `tutorials`.

# License Information

<p xmlns:dct="http://purl.org/dc/terms/" xmlns:vcard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <a rel="license"
     href="http://creativecommons.org/publicdomain/zero/1.0/">
    <img src="http://i.creativecommons.org/p/zero/1.0/88x31.png" style="border-style: none;" alt="CC0" />
  </a>
  <br />
  To the extent possible under law,
  <a rel="dct:publisher"
     href="github.com/gchure/reproducible_research">
    <span property="dct:title">Griffin Chure</span></a>
  has waived all copyright and related or neighboring rights to
  <span property="dct:title">A template for using git as a platform for reproducible scientific research</span>.
This work is published from:
<span property="vcard:Country" datatype="dct:ISO3166"
      content="US" about="github.com/gchure/reproducible_research">
  United States</span>.
</p>