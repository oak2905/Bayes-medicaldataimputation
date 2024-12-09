# Bayes-CATSI
[Seminar YouTube Link](https://youtu.be/roWfvmFN9Qc)

This repository contains the implementation of the Bayes-CATSI and Partial Bayes-CATSI models, designed for efficient medical time-series imputation with uncertainty quantification using variational inference. The project is built using PyTorch, ensuring modularity, scalability, and ease of integration into existing workflows

## Requirements
Requires Python 3.10 or later with PyTorch and related libraries. Please refer to requirements.txt for details of python packages required.

## Usage
The folders `/BayesCATSI` and `/partialBayesCATSI` contain the code for Bayes-CATSI and partial Bayes-CATSI models. Place the dataset in a new folder and train the model as follows.
```{bash}
python main.py --input /path/to/training/data --testing /path/to/test/data
optional arguments:
--output                Folder to save the results
--epochs                number of epochs
--batch_size            batch size for the model
--eval_batch_size       evaluation batch size for the model
```

Data used in this project can be found at: [Data](https://physionet.org/content/challenge-2018/1.0.0/) <br>
Dummy data has been uploaded in the `\data` folder to provide an idea of the data structure. <br>
Computational Resources: In this project, we leverage Google Colab’s free tier, which provides a CPU-based execution environment for running our code.

## Citing the work
```{bash}
@misc{kulkarni2024bayescatsivariationalbayesiandeep,
      title={Bayes-CATSI: A variational Bayesian deep learning framework for medical time series data imputation}, 
      author={Omkar Kulkarni and Rohitash Chandra},
      year={2024},
      eprint={2410.01847},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.01847}, 
}
```



