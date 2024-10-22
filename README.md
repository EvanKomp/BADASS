# BADASS - Designing diverse and high-performance proteins with a large language model in the loop

[![CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Introduction

BADASS package introduces a novel approach to protein engineering through directed evolution with machine learning, integrating two key components: **Seq2Fitness**, a semi-supervised neural network model for predicting protein fitness, and **BADASS** (Biphasic Annealing for Diverse Adaptive Sequence Sampling), an optimization algorithm for efficient sequence exploration.

**Seq2Fitness** combines evolutionary data with experimental labels to predict fitness landscapes, providing more accurate fitness predictions than traditional models. It enhances the ability to design proteins with multiple mutations, helping discover high-fitness sequences where evolutionary selection might not directly apply.

**BADASS** improves the exploration of vast sequence spaces by dynamically adjusting mutation energies and temperature parameters, maintaining diversity and preventing premature convergence. This approach outperforms traditional methods like Markov Chain Monte Carlo, offering a more efficient way to discover high-performance proteins while requiring fewer computational resources.

Together, these tools enable rapid, accurate, and diverse protein design, with potential applications extending beyond proteins to other biomolecular sequences like DNA and RNA. BADASS requires no gradient computations, allowing for faster and more scalable optimization.

## Installation

To install the package, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SoluLearn/BADASS.git
2. Navigate to the project directory:
   ```bash
   cd BADASS
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 

## Usage

This package comes with example Jupyter notebooks, located in the `notebooks` folder, to train Seq2Fitness models and run the BADASS optimizer with both ESM2 and Seq2Fitness, using the **[Alpha Amylase](https://doi.org/10.1016/j.csbj.2024.09.007)** dataset. You can modify these notebooks to meet your needs accordingly. Additionally, code to train the model and run optimization for the **[NucB](https://doi.org/10.1101/2024.03.21.585615)** dataset is provided. The main notebooks to use the code include:

1. **[Train_Seq2Fitness_Models.ipynb](notebooks/Train_Seq2Fitness_Models.ipynb)**: This notebook demonstrates how to train the Seq2Fitness models by leveraging evolutionary and experimental data to build fitness prediction models, using the alpha-amylase dataset.
   
2. **[Run_BADASS_Alpha_Amylase.ipynb](notebooks/Run_BADASS_Alpha_Amylase.ipynb)**: This notebook shows how to use BADASS for protein sequence optimization for alpha amylase, either using a Seq2Fitness or a an **ESM2** model.





