# Heart disease Prediction: A Comprehensive Machine Learning Project

**Overview**
This project presents a systematic investigation into heart disease prediction, using machine learning techniques, which focuses mainly on balancing techniques for imbalanced datasets, and transfer learning. The project compares multiple data balancing strategies on different tree ensemble based models and evaluates the effectiveness of it on this dataset, and testing an offered method for transfer learning, offered by the paper:
**"Heart failure survival prediction using novel transfer learning based probabilistic features"**
Published at **Peer J, computer science** 


## Table of Contents

- [Dataset](#dataset)
- [Data Preprocessing](#data_preprocessing)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Experiments Design](#experiments-design)
  - [Experiment 1: Data Balancing](#experiment-1-transfer-learning)
  - [Experiment 2: Data Balancing Techniques](#experiment-2-data-balancing-techniques)
- [Results](#results)
- [Key Findings](#key-findings)

## Dataset
**Name:** Heart Failure Clinical Records
**Source:** https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
**Size:** 299 
**Number of features:** 12
**Target:** Binary classification (1 = death class, 0 = survival class)
**Class distribution:** 32% positive group, 68% negative group

## Data Preprocessing 
The features are all numerical, no missing values. 
The dataset was explored for correlation among features, and feature importance. 
It showed that the highest correlation was of time and the target, -0.53. 
This is logical, as time gets shorter for the positive group patients. 

As we used tree based models for this project, scaling isn't required. We also chose to leave the outliers, as these type of models are robust for outlier noise. 

The researchers applied similar data exploration.

## Project Structure

project-name/
├── transfer_learning_project.ipynb
├── data_balancing_experiments.ipynb
├── heart_failure_clinical_records_dataset.csv
├── Heart failure survival prediction.pdf
└── README.md

## File Descriptions

**transfer_learning_project.ipynb:** focuses on conducting the transfer learning technique introduced by the paper - creating new probabilistic feature from a trained model, and retrain. The dataset was balanced using smote.
**data_balancing_experiments.ipynb:** focuses on conducting 3 different experiments: 
train unbalanced models, train smote balanced models and balance using class weights hyperparameter. 3 models were tested at each experiment: XGB, RF, catboost. 
**heart_failure_clinical_records_dataset.csv:** contain the dataset.
**Heart failure survival prediction.pdf:** contain the paper this project is based of. 
**README.md:** Documentation file

## Experiments Design

**Experiment 1: Transfer Learning** 
Investigation of the technique proposed in the paper. 
**Base model**: RF with SMOTE applied for balancing the dataset, with 10 folds cross-validation
**Strategy:** Probabilistic feature extraction and concatenation
**Process:** Train RF model using a pipeline with SMOTE, extract class probabilities from each CV test fold (prevent data leakage), concatenate the probabilities to the original train set, as a new feature and train a new model on the enhanced dataset. 
**Expected challenges:** The new feature would act as a summery of the other features its based on, rather then additional predictive feature.

**Experiment 2: Data Balancing Techniques** 
Our investigation focuses on 3 primary approaches: 
 1. **No Balancing** 
		 Training a baseline model with no speical balancing. 
 2. **SMOTE (Synthetic Minority Oversampling Technique)**
		1. Synthetic creation of samples from minority class, based on closest neighbors.
		2. Implementation using imbalanced learn library
 3. **Class Weights**
	1. Tunable Hyperparameter which Penalizes misclassification
	2. Built in model parameter in each model

**Model Selection & Evaluation**
3 Robust ensemble models were chosen for the comparison:

 1. **Random Forest (RF):** Bagging based ensemble model
 2. **Extreme Gradient Booting (XGB):** Gradient boosing with advanced regularization
 3. **CatBoost:** Gradient boosting optimized for categorical features

**Hyperparameter Optimization**
**Method:** Randomized Search Cross-Validation
**CV:** 10 fold cross-validation
**Overfitting monitoring:** CV Train vs. CV Test and validation scores comparison.
Optimized metric: F1-score
**Evaluation metrics:** Accuracy, Precision, Recall, F1-score

# Results
**Performace Comparison**

|Model|Balancing_Method|Accuracy|Precision|Recall|F1-Score|
|-----|----------------|--------|---------|------|--------|
|RF|No_Balance|0.87|0.87|0.68|0.76|
|RF|SMOTE|0.88|0.80|0.84|0.82|
|RF|Class_Weights|0.87|0.82|0.74|0.78|
|XGB|No_Balance|0.88|0.88|0.74|0.80|
|XGB|SMOTE|0.88|0.83|0.79|0.81|
|XGB|Class_Weights|0.88|0.88|0.74|0.80|
|CatBoost|No_Balance|0.88|0.88|0.74|0.80|
|CatBoost|SMOTE|0.87|0.79|0.79|0.79|
|CatBoost|Class_Weights|0.87|0.82|0.74|0.78|

**Tranfer Learning Results**
For minority group:

|Model|Approach|Accuracy|Precision|Recall|F1-Score|Feature_Count|
|-----|--------|--------|---------|------|--------|-------------|
|RF|Baseline with SMOTE|0.93|0.90|0.90|0.90|12|
|RF|Enhanced_with_Probabilities|0.97|1.00|0.89|0.94|14|
|XGB|Baseline with SMOTE|0.90|0.80|0.89|0.84|12|
|XGB|Enhanced_with_Probabilities|0.90|0.80|0.89|0.84|14|

# Key Findings
1. **Transfer Learning:** 
The new probabilistic feature engineering didn't add a new predictive information to the model, and the results remained the same.

3. **Data Balancing:**
No Balancing: Boosting models naturally handled imbalanced data better then RF, and showed good initial results.
SMOTE: improved minority class recall in all models, with expected tradeoff of reduced precision.
Class Weights: RF benefited from class weights, but the boosting models already focused on correcting minority errors, benefited less if any.  


