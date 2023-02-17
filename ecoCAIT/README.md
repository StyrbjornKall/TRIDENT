## `fishbAIT`

Contains the final EC50, EC10 and Combined (M<sub>50</sub>, M<sub>10</sub> & M<sub>50/10</sub>) models generated in the publication. Each model comes trained on either fish, invertebrate or algae data. A model is loaded in two parts:

1. From the .pt files in this directory containing the weights and biases for the DNN-module of the model
2. From [Huggingface](https://huggingface.co/StyrbjornKall) for the respective model's RoBERTa (transformer)-module

For a tutorial on how to use these models, refer to `tutorials`.