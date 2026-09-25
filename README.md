# SoarDDDS8555-6
Build and Evaluate Tree Models
README.md

# Assignment 6: Build and Evaluate Tree Models

**Course:** DDS 8555  
**Assignment:** 6 — Build and Evaluate Tree Models  
**Author:** Soar, Donald  
**Date:** September 13, 2026

## Overview

This project evaluates tree-based and Bayesian ensemble methods for
multiclass obesity-risk prediction using the Kaggle
**Multi-Class Prediction of Obesity Risk** dataset.

The analysis addresses the required ISLR conceptual and applied
questions and develops classification models using:

1. Decision Tree
2. Bagging
3. Random Forest
4. Boosting
5. BART (Bayesian Additive Regression Trees)

Logistic regression is also included as a baseline statistical
classification method.

The primary response variable is `NObeyesdad`, which contains seven
obesity-risk categories.

## Data

The training data contain 20,758 observations and 18 variables.

The test data contain 13,840 observations and 17 predictor variables.

The response variable is:

`NObeyesdad`

The seven response classes are:

- Insufficient_Weight
- Normal_Weight
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III
- Overweight_Level_I
- Overweight_Level_II

The training data contain no missing values and no duplicate rows.

## Modeling Strategy

Categorical predictors are converted to numerical representations using
scikit-learn preprocessing pipelines. Numerical predictors are retained
as continuous variables.

The data are divided into training and validation portions using a
stratified split so that the seven outcome classes remain represented
in approximately their original proportions.

Model performance is evaluated using:

- Accuracy
- Macro F1
- Weighted F1
- Class-specific precision
- Class-specific recall
- Confusion matrices

Macro F1 is emphasized because this is a multiclass problem and the
class frequencies are not perfectly equal.

## Tree-Based Models

### Decision Tree

A decision tree provides an interpretable recursive binary partition of
the predictor space.

### Bagging

Bagging fits multiple bootstrap samples of decision trees and combines
their predictions to reduce variance.

### Random Forest

Random forests extend bagging by randomly selecting a subset of
predictors at each split. This decorrelates the individual trees and
can improve predictive performance.

### Boosting

Boosting sequentially builds weak learners, with subsequent learners
emphasizing observations that were not adequately modeled by earlier
learners.

### BART

Bayesian Additive Regression Trees represent the response function as a
sum of many regularized trees. BART is implemented using PyMC-BART.

Because `NObeyesdad` contains seven classes, the BART analysis uses a
one-versus-rest construction. A separate Bernoulli-BART model is fitted
for each class, and the resulting class probabilities are normalized
before selecting the final multiclass prediction.

## Current Validation Results

From Kaggle Evidence of Submission, the completed tree-ensemble models currently produce the following
validation results:


|**Submission**	   | **Private Score**	 | **Public Score**	 |  
|:---              |---:                 |---:                   |
|
|  Bart            |   0.73256	         |      0.73338	         |  
|
|  Random Forest   |   0.89216	         |      0.89161	         |   
|
|  Decision Tree   |   0.83607	         |      0.83995 	 |   
|
|  Boosting        |   0.90408	         |     0.91004  	 |
|
|  Bagging         |   0.89387	         |     0.89848	         |
____________________________________________________________________________


Boosting is currently the strongest completed model.

## Feature Importance

The random forest analysis identifies `Weight` as the most important
predictor, followed by `Age`, `Height`, and `FCVC`.


## Repository Structure

```text
SoarDDDS8555-6/
??? SoarDDDS8555-6.Rproj
??? SoarDDDS8555-6.Rmd
??? SoarDDDS8555-6-Report.Rmd
??? README.md
??? .gitignore
??? data/
??? scripts/
?? submissions/
    ??? tables/
