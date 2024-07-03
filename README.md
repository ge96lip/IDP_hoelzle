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

4. Install the required conda packages.

### 3. Install PCN Toolkit and Dependencies

```
pip install pcntoolkit
```

## Training and Evaluating the Model

### To Train a New Model:

```
python run.py train "data_path" "model_path"
```

### To Evaluate a Trained Model:

```
python run.py test "data_path" "model_path" "site_names"
```

#### Optional Arguments:

The following are optional arguments for the `run.py` script:
- `--arg1`: Description of arg1.
- `--arg2`: Description of arg2.
- `--arg3`: Description of arg3.

## Running the GAM/GAMLSS Code

### 1. Import R File into R Environment

Load the provided R script into your R environment.

### 2. Install Required Packages

Ensure all necessary packages are installed in your R environment.

### 3. Run the Code

Execute the provided R script within the R environment to perform the modelling.

---

For further details on usage, please refer to the respective documentation and source code comments within the project.
