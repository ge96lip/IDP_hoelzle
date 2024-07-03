# README

## Project Overview

This project contains the codebase used for the comparison of PCN Toolkit BLR vs. GAM/GAMLSS modelling. The code has been adapted from the following sources:

- **PCN Toolkit**: Adapted from the [PCN Braincharts GitHub repository](https://github.com/predictive-clinical-neuroscience/braincharts).
- **GAM/GAMLSS**: Adapted from the work by Wachinger, C., Hedderich, D., and Bongratz, F. "Stochastic Cortical Self-Reconstruction." arXiv preprint arXiv:2403.06837, 2024.

## Running the PCN Toolkit Code

### 1. Clone the Repository

```
git clone git@github.com:ge96lip/IDP_hoelzle.git
```

### 2. Install the Environment

#### Using Anaconda:

1. Install Anaconda3.
2. Create an environment using the provided environment.yaml:

    ```
    conda env create --file myenv.yml
    ```

3. Activate the environment:

    ```
    conda activate myenv.yml
    ```

## Training and Evaluating the Model

### To Train a New Model:

```
bash script.sh train "model_path" "data_path"
```

### To Evaluate a Trained Model:

```
bash script.sh eval "model_path" "data_path" "file_name_for_sites"
```

#### Optional Arguments:

The following are optional arguments for the `run.py` script:
- `--map_roi`: If the ROI regions from the trained models are named different than in the dataset used, only for eval run, in training the column names are used for naming the models.
- `--idp_ids`: The ROI regions a model should be trained for, default: ['L_entorhinal','L_inferiortemporal','L_middletemporal','L_inferiorparietal','L_fusiform'].
- `--cols_cov`: Which covariates should be used for training/testing, default: AGE, GENDER.
- `--drop_columns`: If the dataset has columns which should not be included, default: True (specific for the dataset used during training). 

## Running the GAM/GAMLSS Code

### 1. Import R File into R Environment

Load the provided R script into your R environment.

### 2. Install Required Packages

Ensure all necessary packages are installed in your R environment.

### 3. Run the Code

Execute the provided R script within the R environment to perform the modelling.

---

For further details on usage, please refer to the respective documentation and source code comments within the project.
