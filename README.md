# Stochastic Gradient Descent Methods and Uncertainty Quantification in Extended CLSNA Models

This repository supports the paper "Stochastic Gradient Descent Methods and Uncertainty Quantification in Extended CLSNA Models".

## Code Overview

- `simulation/`: This directory includes `main.ipynb` and `simulate_congress.ipynb`. The `main.ipynb` notebook serves as the entry point for simulations with synthetic data. It simulates a dynamic network from the CLSNA model with a fixed number of members and fits the extended CLSNA model to the simulated data for inference. The `simulate_congress.ipynb` simulates a dynamic network with changing membership and fit the extended CLSNA model to the simulated data for inference.

- `X/207/`: Contains the `207.ipynb` file, which is the primary script for analyzing Twitter congressional hashtag networks with 207 nodes. These nodes represent members who were consistently present throughout the study period.

- `X/505/`: Contains the `505.ipynb` file, analyzing the same Twitter congressional hashtag networks, but includes all nodes recorded during the study.

- `X/yearly/`: This directory contains raw network data files named `aggregated_network_hashtag_intersection_year***.csv`. Additionally, the `network_features.csv` file provides metadata about the actors in the network.

- `X/compare/`: The entry point here is `compare.ipynb`, which plots the trajectory of the mean latent positions of the members of each party, comparing the model fitting result from the reduced dataset in `207.ipynb` and the full dataset in `505.ipynb`.

## Implementation Details

Each notebook (`main.ipynb`, `simulate_congress.ipynb`, `207.ipynb`, and `505.ipynb`) can be executed from top to bottom. They call helper functions and classes from `utils.py` and `congress_utils.py`. Each notebook follows a unified structure with three primary steps:

1. **Initial CLSNA Model Fit**: Start with a CLSNA model with a higher-dimensional space than the intended dimension. The model is then fitted, utilizing the first `p` principal components of the fitted latent position as the initial values for the next step.

2. **Point Estimation**: A model with the targeted dimension is fitted. The outputs are point estimations for model parameters.

3. **Variance/Covariance Estimation**: Perform variance/covariance estimation for the parameters of interest.

Each step is unified under an SGD approach and utilizes a unified `nn.module` class from the PyTorch autodifferentiation library, defined in `utils.py` and `congress_utils.py`.

## Reporting Bugs

To report bugs encountered while running the code, please contact Hancong Pan at hcpan@bu.edu.
