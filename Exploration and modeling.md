# Data Exploration Approach

I tried to correlate other independant variables to the target `MULTIPLE_OFFENSE` and found only some features had a little correlation with the same, I noticed some missing values in the column `X_12` and next I explored the target variable `MULTIPLE_OFFENSE`'s distribution, which is heavily skewed as one can see in the plot EDA notebook(**Novarits_data_competition_eda.ipynb**). Over `95%` of the cases where of class `0`.

I also built a **simple** Decision Tree model, to get a better understanding of the independant variables. The simple model set a baseline of `85` recall on the test set, I also plotted the decision tree's leaves which is present in the pdf along with the attached files, from that we can see the important features `X_10`, `X_11`, `X_12` and `X_15`.

# Modeling Approach

With the assumptions in the notebook regarding the classes and what they would mean, the decision tree model did a decent job, without any feature engineering, I chose Random Forests to model the data, I also applied tried to balance the classes with `balanced_subsample` to counter the skewness of the data, I also did not use a **Scaler** since random forests do not need it. I increased the `max_depth` to `5` when compared to `4` when building the baseline model, I used a simple median imputing strategy for dealing with missing variables, I built the model training pipeline, with

    1. MedianImputer
    2. RandomForestClassifier

This model caught all the cases where the minority class was present, it also predicted some false positives, Please refer the training notebook evaluation section for more details. The decision on the best model for this task needs to consider the cost of false alarms and cost of wrong predictions. I have attached the Scikit-learn pipeline objects for both decision tree and the random forest along with other files.