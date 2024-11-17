# Bayes-CATSI
[Seminar YouTube Link](https://youtu.be/roWfvmFN9Qc)

This repository contains the implementation of the Bayes-CATSI and Partial Bayes-CATSI models, designed for efficient medical time-series imputation with uncertainty quantification using variational inference. The project is built using PyTorch, ensuring modularity, scalability, and ease of integration into existing workflows

## Requirements
Requires Python 3.9 or later with PyTorch and related libraries. Please refer to requirements.txt for details of python packages required.

## Usage
The folders `/BayesCATSI` and `/partialBayesCATSI` contain the code for Bayes-CATSI and partial Bayes-CATSI models. Place the dataset in a new folder and train the model as follows.

`python main.py -i /path/to/training/data -t /path/to/test/data`



