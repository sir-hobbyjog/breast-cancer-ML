# Breast Cancer Tumor Classification ML

   ### CONTENTS
**[ML Team](#ml-team)**<br>
**[Malignant or Benign?](#malignant-or-benign)**<br>
**[Data Source](#data-source)**<br>
**[Methodology](#methodology)**<br>
**[Repository Assets](#repository-assets)**<br>
**[Findings/Conclusion](#findingsconclusion)**<br>

## ML Team

This demonstration of machine learning (ML) is the collective work of the following team members as part of University of Utah's AI Bootcamp by edX:
- Bab Jan, *CEO*
- Jake Lee, *Fearless Driver*
- Adam Millington, *He tried and succeeded*
- Ryan Mosher, *Mortgage and bond Oracle*
- Nathan Tyler, *The documenter*

## Malignant or Benign?

### Background

Approximately 240,000 women in the United States are diagnosed with breast cancer each year. Believe it or not, this also affects about 2,100 men each year as well.[^1] Whether through diagnosis or relation this affects many families.

[^1]: https://www.cdc.gov/cancer/breast/basic_info/index.htm

### Our Model

Our project revolves around utilizing machine learning techniques to tackle the challenge of tumor classification. Specifically, we're working with a dataset containing features extracted from cell nuclei images obtained from fine needle aspirates of suspicious nodes. These features encompass various characteristics such as size, shape, and texture of the nuclei.

Through supervised learning, we're training a machine learning model to discern patterns within these features that distinguish between malignant (cancerous) and benign (non-cancerous) tumors. By feeding the model labeled data – where each tumor's malignancy status is known – we enable it to learn how to predict the likelihood of malignancy for new, unseen tumors.

Ultimately, our aim is to develop a reliable tool that can assist healthcare professionals in making more accurate and timely diagnoses. This has the potential to significantly enhance patient care by facilitating early detection and appropriate treatment of cancerous tumors.

## Data Source

The data used for this ML model was sourced from [Kaggle](https://www.kaggle.com/datasets/nancyalaswad90/breast-cancer-dataset) uploaded by Ms. Nancy Al Aswad with the purpose of utilizing it for training ML models to be able to more rapidly classify tumors.

## Methodology

Utilizing modern ML tools found in Scikit-learn and the Python* programming language, a Jupyter Notebook (.ipynb) will be used to handle pre-processing of the original data file, which will output cleaned training and testing datasets as CSV files. Other notebooks contain the code to train and evaluate the several ML models. The Jupyter Notebooks include description of the process we go through to setup an ML model with a final python script implementing the model of our choice. Generally the notebooks will be saved with output. The Python script can be run from a terminal.

*The Python environment should be setup to include:
- Python 
- Scikit-learn
- Pandas
- Plotly
- Jupyter & Jupyter Lab

An analysis of the model will be summarized below in the **[Findings](#findings)** section and a final presentation will be put together using Microsoft PowerPoint.

### Data Exploration/Preprocessing

The initial review of the data showed a very clean dataset with an ID column (which we dropped as unneeded), a diagnosis column used as the target labeled data (which we converted to 1s for malignant and 0s for benign), and 30 features that were numerical data. Unsure what each of the features meant we did not attempt any feature engineering. With only 569 observations we knew we would need to upscale to have a minimum of 1000 data points. We obtained some help with the upscaling code from our teacher Matt. Prior to upscaling the data we found the data was slightly unbalanced, so with our upscaling we balanced the data to 500 malignant and 500 benign data points. With such a clean dataset we determined there was no need for any further pre-processing. Therefore, we performed a training (75%) and test (25%) split of the data.

### Principal Component Analysis

A Principal Component Analysis (PCA) was performed to determine if using PCA would be make a better model. We found that it took 17 components for a linear regression model to get nearly identical results as training the model on all of the data. Therefore we decided that PCA was not necessary with this dataset. We will use the linear regression as a base model and work for improved accuracy with other models.

### Model Exploration

We decided to run multiple ML models on the data to determine which gets the best results. The following seven models were of interest to us:

1. Linear Regression (our base model)
2. Logistic Regression
3. K Nearest Neighbors
4. Support Vector Classifier
5. Decision Trees
6. Random Forest
7. Extra Trees

The results from this step displayed marked improvement from our base model across all other models:

![Model Scores](https://github.com/NeifyT/breast-cancer-ml/blob/main/Plots/classifier_scores.png)

### Model Optimization

Upon finding the Random Forest model was the best we set out to optimize the model to see if we could improve the model.

We explored the importance of individual features of the data using Gini scores and permutations.

![Comparison of feature importance by Gini score and permutation](https://github.com/NeifyT/breast-cancer-ml/blob/main/Plots/feature_importance.png)

However, not being experts in the field and having such a small set of features we decided to continue to train the model with all of the data.

We then explored hyperparameter tuning. First we iterated through 100 random states and taking the mean accuracy score at each level for number of trees we arrive at greater than 99% accuracy around 21 decision trees (the default is 100). The greater number of trees the higher the accuracy score will be, but also the greater chance of overfitting the data, and the small amount of gains diminishes the higher the value.

![Average Accuracy for Number of Trees across different random states](https://github.com/NeifyT/breast-cancer-ml/blob/main/Plots/number_of_trees.png)

We also tried to fine tune the maximum decision tree depth. Choosing a single random state for consistency we compared the effects of numbers of trees with maximum depth to the accuracy score and graphed that in 3D space. [Two such interactive plots](https://github.com/NeifyT/breast-cancer-ml/blob/main/Plots/) are available in this repository, one with more iteration, and another zoomed in around closer to the 21 decision trees. Alas, these do not show within a GitHub repository but could be downloaded and viewed from a local file.

Finally we tried to ensemble models to see if we could optimize by pushing the incorrect predictions to eliminate false negatives (as that can be deadly with breast cancer) while not overburdening patients with false positives. We generated confusion matrices to demonstrate this effort.

![Comparison of confusion matrices](https://github.com/NeifyT/breast-cancer-ml/blob/main/Plots/confusion_matrix.png)

For all the model optimization we found that the default settings for Random Forest would produce the best and most consistent results. Our final model then just kept to the defaults with the Random Forest and trained with all the data features.

## Repository Assets

### Source Code

- **data_handling.ipynb**: Pre-processes and splits data into training and testing sets
- **model_exploration.ipynb**: Trains and tests data on seven distinct models
- **importance.ipynb**: Explores which “features” of the data have the most significance for random forest decision trees.
- **model_optimization**: Uses a logical approach to find the optimal hyperparameters for the model including the number of trees and the maximum depth.
- **classifier_test.py**: A utility file with predefined function used to train and test different classifier models and print basic results.
- **main.py**: Our final model in a ready to run script which makes a call to classifier_test.py for output.

### Resources

- original-data.csv[^2]
- training_data.csv
- testing_data.csv

[^2]: Obtained from [Kaggle](https://www.kaggle.com/datasets/nancyalaswad90/breast-cancer-dataset)

## Findings/Conclusion

We found that Machine Learning models are well suited for use in the medical field for diagnosis of cancer tumors provided the data collected is accurate. We are pleased to see news articles indicating their use and have great hope in the future for all the women in our lives.
