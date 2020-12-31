# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model using hyperdrive configuration vs  Azure AutoML method. 

## Summary

Data set used for this project https://archive.ics.uci.edu/ml/datasets/Bank+Marketing contains the information from Portuguese Bank Marketing. The informations is based on phone calls and We want to predict if the client has suscribed a term deposit. Based to Age, type of job, marital, education, has credit in defualt, housing, loan, type of contact , last contact, day of the week of the contact, 
duration in seconds, campaigns, and other variables  

The best performance model was the algorithm Voting Ensemble found using AutoMl.The accuracy was of 0.9164 .
Regarding the method using python with Hyperdrive gives an accuracy of 0.9109


## Scikit-learn Pipeline

**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

First we configurate the Compute Target Cluster with a vm_size STANDARD_D2_V2 and max_nodes =4 . This give us the resources for creating the infraestructure to run the projects. We used the Bank Marketing database explained before and we create a estimator Sklearn where the train.py is running. The train.py open and create the dataset and convert to data frame , clean the information where convert the categorical values to dummies, split in train test data and  apply the logistic regression model. 

We apply the hyperdriveconfig using  RandomParameterSampling  '--C':choice(0.01,0.05, 0.1, 0.5,1) and --max_iter':choice(5, 10, 20, 50, 100) This choice discrete values over a parameter search space.
We also used a Banditpolicy with the following parameters

Evaluation_interval =2  the frequency for applying the policy

delay_evaluation = 0.1  delays the first policy evaluation for a specified number of intervals

**Parameter sampler**
Random Sampling use random combinations of the hyperparameters to find best solution for the built model. 
Random Sampling Compared has better results over Grid Search method.

**Stopping policy**

Bandit Policy help us to stop based in the slack factor and evaluation interval . 
The policy early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run , so we can avoid to spend much hardware resources.

## AutoML

Auto Machine learning give us the possibility to run diferent and multiple models with training jobs to find the right model. This is very fast and helpful beca use we would have to spend a lot of time going into different models and different featurization pipelines for each of the algorithms.


## Pipeline comparison

**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The performance using AutoML 0.9164 is slighly better than Hyperdrive 0.9109 but AutoML is faster than hyperdrive also gives the opportunity to implement and compare different models and the steps are very easy with code and with the azure platform. We can reduce implementation times and deployments


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

We can improve the accuracy by trying different parameters in the hyperdrive configuration.

Also using the option to run Deeplearning in AutoML configuration could improve the accuracy of the model and GPU option trying other compute target.
## Proof of cluster clean up
![Cleanup](imgs/Clustercleanup.png)
