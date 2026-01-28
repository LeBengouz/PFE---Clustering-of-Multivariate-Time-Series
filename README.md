# Clustering of Multivariate Time-Series Data 

**author** : Benjamin Olive


## Overview

This project aims to explore and becnhmark various clustering techniques applied to multivariate time-series. We look for different solutions to compare and group multivariate time series by comparing the multiple dimensions of the same trace, with the goal of improving the accuracy and relevance of the identified clusters. 

This projects then explores different methods, with the end-goal of constructing a graph that represents the connections between the main subsequences of our time-series data.

The "/Sources" folder contains pdf files of the articles related to algorithms compared in this project.

## Requierements

- marimo==0.19.2
- plotly==6.5.1
- numpy==2.3.3
- pandas==2.3.3
- regex==2025.11.3
- matplotlib==3.10.8
- tslearn==0.7.0
- tqdm==4.67.1

## Dataset

First dataset used is cabspottingdata. It contains mobility traces of taxi cabs in San Francisco, USA. This dataset includes GPS coordinates of approximately 500 cabs. This dataset includes GPS coordinates (latitude and longitude) for approximately 500 taxis, providing two-dimensional time-series data.

**Why this dataset ?**
- Interpretability : it is easy to visualize GPS traces on a map, making it easier to evaluate by hand the performance of the clustering produced.
- Benchmarking : hence, this dataset is a baseline to test and compare different clustering techniques before applying the best-suited algorithms to other multivariate datasets.

**How to use ?**
In order to use the San Francisco's dataset, user will need a folder named *"cabspottingdata"* in your porject directory. This folder should contain all the data files from this dataset.