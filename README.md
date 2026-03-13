# Battery_Capacity_Project

This project creates an MLPRegressor that predicts battery capacity
based on voltage, temperature, and cycle number.  The project shows the 
behavior of this model with different solvers and analyzes the fit of 
the final model (AdaM solver) vs the actual data. I also compare the 
performance of the MLPRegressor against a simpler linear model.

Predicting capacity can significantly reduce the time and cost required for 
battery testing. Running batteries to the end of life experimentally can take
years and gigawatts of power. A predictive model like this can accelerate 
research.


## Dataset

The model uses the NASA Battery Degradation Dataset (cycle-level).

Dataset source:
https://www.kaggle.com/datasets/yashxss/nasa-battery-cycle-level-dataset

The dataset contains experimental measurements from lithium-ion batteries cycled under different conditions.


Features (X)

```
- voltage
- temperature
- cycle
```

Target (Y)

```
- capacity
```

Total samples used: 1415

## Installation

1. Download the NASA dataset: 
```
https://www.kaggle.com/datasets/yashxss/nasa-battery-cycle-level-dataset
```
2. Clone the repo:
```
git clone https://github.com/MMTovar2/Battery_Capacity_Project.git
```
3. Create a virtual environment: 
```
python -m venv .venv
```
4. Activate the virtual environment:

- Mac/Linux:
 ```
.venv/bin/activate 
 ```
- Windows(command prompt): 
```
.venv\Scripts\activate.bat
``` 
- Windows (PowerShell):
```
.venv\Scripts\Activate.ps1
```
5. Install the dependencies: 
```
pip install -r requirements.txt
```
6. Replace ``` path/to/battery_cycle_level_dataset_CLEAN_FINAL.csv```
with the actual path to your dataset
7. You can use this command to run the project: 
```
python Battery_Capacity_Predictor.py
```
8. When done, deactivate your virtual environment: 
```
deactivate
```

## Visualizations
The script will print out the train, valid, and test MSE and R<sup>
2</sup> for most of the models, along with the training time each model 
took. 

The script also generates several plots:

1. **Predictions vs Residuals**  
   Scatter plot of predicted values against residuals

2. **Residual Histogram**  
   Distribution of residuals

3. **Loss Curve**  
   MLPRegressor number of iterations against Magnitude of Error (loss)

These plots help analyze model performance and assess overfitting or instability.