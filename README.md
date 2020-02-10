# Explainable_Lending_Decision

## 1. Overview

### 1.1. Problem Statement

More businesses and governments are adopting artificial intelligence (AI) technologies such as advanced machine learning to make faster and better decisions. But, the GDPR (General Data Protection Regulation) requires financial firms adopting AI to meet a number of requirements like providing transparency and demonstrating compliance, which means Lenders must give applicants detailed disclosure about the use of automation to make credit decisions. And many of the artificial intelligence (AI) technologies are hard to explainable as Machine learning models are complex “black boxes.”

### 1.2. Solution

In Explainable_Lending_Decison, I have developed a practical and explainable solution through a novel application of Shapley Values.  Adopting Kiva’s datasets and a studied case, I constructed A series of high performance financial models and selected the most reasonable model by comparing their performances. And Individual profile could be explained for lending consideration to Help businesses lower financial risk and increase benefits.

## 2. Resource list

- [**Presentation slides**](bit.ly/explainable_ff_slides) explaining the problem, solution approach and results in 5 mins are available here
- **Streamlit reports:**
  - [**Project Demo**](https://share.streamlit.io/0.36.0-2Qf24/index.html?id=JDjgoPh55HrSxbKvpthCj2M) # need to add new link

## 3. Running the code on your machine

### 3.1. Requisites
The code was developed on Python 3.7 and requires the following libraries:

- scikit-learn
- keras
- argparse
- numpy
- [Streamlit](https://streamlit.io/secret/docs/index.html)

After cloning the repository you can recreate the environment and install the package dependencies using:

```
conda env create -f build/envlending.yml
conda activate envlending
```

### User Interface
- model selection: Random Forest, Xgboost and  ...
- feature selection

### Running the code
```
cd /src
streamlit run main.py
```
## 4. Data

Testing of this methodology was performed using census income data ([Kiva dataset](https://bigml.com/user/ashikiar/gallery/dataset/52290c30035d0729c1004566)):
- Around 4 million data points
- Target feature: Status( paid or defaulted)
- 10 features (in addition to the target feature) including borrowers’ information


### 4.1. File structure / data flow in the code

- The raw data files are saved in ```data/raw```
- After processing, the code saves a new csv file containing the processed data in ```data/processed/cleaned_labeled.csv```
- Follow the instructions above for running the code

#### Notes:

- While efforts have been made to generalise, this code has not been tested on other datasets
- Credits to [Michail Tzoufras](https://github.com/michail-tzoufras/LendingAtlas)
