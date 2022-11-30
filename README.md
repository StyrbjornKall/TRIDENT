# fishbAIT - fish based AI Toxicity
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

## Layout
`data` contains all preprocessed data used for training our fine-tuned EC50, EC10 and combined model.

`development` contains all code used to replicate the findings presented in the publication.

`fishbAIT` houses the fine-tuned model parameters

`tutorials` contains tutorial notebooks for running inference using the fine-tuned models.

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