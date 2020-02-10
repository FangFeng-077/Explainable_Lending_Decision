# Explainable_Lending_Decision
A practical and effective approach to explaian the lending decisions machine made

## 1. Overview

### 1.1. Problem Statement

More businesses and governments are adopting artificial intelligence (AI) technologies such as advanced machine learning to make faster and better decisions. But, the GDPR (General Data Protection Regulation) requires financial firms adopting AI to meet a number of requirements like providing transparency and demonstrating compliance, which means Lenders must give applicants detailed disclosure about the use of automation to make credit decisions. And many of the artificial intelligence (AI) technologies are hard to explainable as Machine learning models are complex “black boxes.”

### 1.2. Solution

In Explainable_Lending_Decison, I have developed a practical and explainable solution through a novel application of Shapley Values.  Adopting Kiva’s datasets and a studied case, I constructed A series of high performance financial models and selected the most reasonable model by comparing their performances. And Individual profile could be explained for lending consideration to Help businesses lower financial risk and increase benefits.

## 2. Resource list

- [**Presentation slides**](bit.ly/explainable_ff_slides) explaining the problem, solution approach and results in 5 mins are available here
- **Streamlit reports:**
  - [**Project Demo**](https://share.streamlit.io/0.36.0-2Qf24/index.html?id=JDjgoPh55HrSxbKvpthCj2M) 

## 3. Running the code on your machine

### 3.1. Requisites

- anaconda
- Python 3.6 
- SHAP
- [Streamlit](https://streamlit.io/secret/docs/index.html)

This repo uses conda's virtual environment for Python 3.

#### Install (mini)conda if not yet installed:

For MacOS:
```shell
$ wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
$ chmod +x miniconda.sh
$ ./miniconda.sh -b
```

#### Create the virtual environment:

cd into the directory and create the ```debial-ml``` conda virtual environment from environment.yml
```shell
$ conda env create -f environment.yml
```

#### Activate the virtual environment:

```shell
$ source activate debias-ml
```

### 3.2. Running the code

As described above, there is script that can be run to produce Streamlit reports.
- ```analysis_data.py```

These can be all be run from the command line. To do this cd into the ```source``` directory and call,
```shell
$ python analysis_data.py
```
The scripts are listed in order of running time.

## 4. Data

Testing of this methodology was performed using census income data ([Kiva dataset](https://bigml.com/user/ashikiar/gallery/dataset/52290c30035d0729c1004566)):
- Around 4 million data points
- Target feature: Status( paid or defaulted)
- 10 features (in addition to the target feature) including borrowers’ information


### 4.1. File structure / data flow in the code

- The raw data files are saved in ```data/raw```
- The raw data is converted to csv format and saved as ```data/preprocessed/adult-data.csv```
- The input parameters are set manually in ```config/params.ini```
- After processing, the code saves a new csv file containing the processed data in ```data/processed/adult-data.csv```
- Parameters which are calculated in data processing and required for later calculations are written to the config file ```config/new_params.ini```

### 4.2. Running the code on a new data set

1. Save the csv file in the folder ```data/preprocessed/```
2. Edit the parameter values in the config file, ```config/params.ini```
3. Don't worry about overwriting the parameters for ```adult-data.csv```, a copy of the config file is saved as ```adult-data_params.ini```
4. Follow the instructions above for running the code

#### Notes:

- While efforts have been made to generalise, this code has not been tested on other datasets
- Credits to [Michail Tzoufras](https://github.com/michail-tzoufras/LendingAtlas)
