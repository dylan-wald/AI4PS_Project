# AI4PS_Project

This is the repository for the AI4PS group project
Authors: Dylan Wald, Andrew Rouze, Ghazaleh Sarfi

## Code structure for training and testing the PINN:
- main.py (for hyperparameter tuning, run "main_optuna.py" instead)
- src/
    - utils.py
        - CunstomDataset() -> to create a torch dataset (not used here, we do it manually)
        - Data() -> manual utility functions (loading/importing data, normalization, train-test split)
        - PINNLoss() -> this is the physics loss used in the training of the PINN
    - NN.py
        - FNN() -> fully connected neural network (not used)
        - LSTM() -> long short-term memory network class
- data/
    - PAPER_normal.csv -> the data used to train the PINN under normal conditions (also used in the optuna hyperparameter tuning process)
    - PAPER_fault.csv -> the data used to train the PINN under fault conditions
    - PAPER_bypass.csv -> the data used to trian the PINN under bypass conditons
    - there are many other .csv files in here that were used in various iterations of this project (dont need to worry about them)


## To run the code: 
python main.py

## To run the Optuna problem: 
python main_optuna.py