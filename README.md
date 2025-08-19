## Overview
ML system using spatio-temporal patterns to forecast Uber/Lyft ride demand for Chicagos different community areas. The data is downloaded from https://data.cityofchicago.org/ and contains 150 million samples. To processes this enourmous amount of data,
Apache Spark was initially used for initial exploration on Databricks community edition while majority of exploration was done locally using duckdb. Models were tested using statsmodels Poisson regression, sktime autoarmima, Meta/Facebook Prophet, and PyTorch Geometric Temporal. A Spatio-Temporal Graph Convolutional Network model architecture was chosen due to significantly lower validation mean squared error.
Feature transforms and seasonal variables The models and performance were tracked using MLflow, with the final PyTorch Geometric Temporal model being wrapped in an MLflow PythonModel for easy model serving. With the model, the rest of the system design was layed out.

## System Architecture 
<img src="images/systemarchitechure.png" width="800"/>
