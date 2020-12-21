# House_price_prediction_model

This repository provides a Pythonic implementation of House Prices by utilizing Python concept of OOP.
This repository is is refactored to fit any prediction based problems and consists of the Machine Learning
Pipelines (Data Exploratory, Feature Engineering, Feature Selection, Cross Validation, Hyperparameter Optimization
and Model training and prediction)

#Installation

This implementation is written with Python version 3.6 with the listed packages in the requirements.txt file

1) Clone this repository with git clone https://github.com/Victoloporsche/House_price_prediction_model.git
2) With Virtual Environent, use : 
    a) pip install virtualenv
    b) cd path-to-the-cloned-repository
    c) virtualenv myenv
    d) source myenv/bin/activate
    e) pip install -r requirements.txt
3) With Conda Environment, use:
  a) cd path-to-the-cloned-repository
  b) conda create --name myenv
  c) source activate myenv
  d) pip install -r requirements.txt

# Running the Implementation:
-- Input folder consists of the training and testing data
-- Model folder consists of the trained model
-- Output folder consists of the encoded categorical features
-- src folder consists of the python and jupyter files

The oder of running this repository is:

1) data_exploration.py : This classs provides a detailed information about the dataset
2) feature_engineering.py: This class performs feature engineering techniques on the data 
3) feature_selection.py: This class selects the best features for the model
4) preprocessed_data.py: This class combines step 2 and 3
5) model_optimizer.py: This class performs cross validation, hyperparameter optimization and model training as well as prediction
6) Main.py: This class combines steps 1-5 
6) example_house_price_predictor.ipynb: This provides documentation of the model.

Next Step: Deploying this model as a web based application for automated house price prediction

More modifications and commits would be made to this repository from time to time. Kindly reach out if you have any questions or improvements.
