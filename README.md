# Credit Score Classification

## Project Outline

The goal of this project is to classify credit scores using machine learning. The process involves several key steps, including data cleaning, preprocessing, feature engineering, and model training. A brief overview of the project is provided below.

## Data Cleaning and Preprocessing Steps

- Dropped features that are not useful for model training: `ID`, `Customer_ID`, `Name`, `SSN`.
- A significant number of null values are present, and many numeric-type columns seem to be encoded in object format. The data is preprocessed appropriately as needed for these features.
- Converted the `Credit_History_Age` feature from a string format representing years and months to a single numerical value in years.
- Dropped the `Type_of_Loan` column due to its high cardinality and inconsistent entries.

### Handling Inconsistent Entries
- Erroneous entries in each feature were replaced with NaN values.
- Replaced negative values in certain columns (where the presence of negative values is illogical) with zero or NaN.
- The NaN values were later imputed with appropriate imputation methods.

### Outlier Detection and Handling
- Used box plots to visualize outliers.
- Replaced outliers with NaN values to avoid data loss and prepared the data for imputation.

### Imputation
- Implemented Iterative Imputer to handle missing values in numeric columns using RandomForestRegressor as the estimator.
- For categorical columns (`Occupation`, `Credit_Mix`, `Payment_Behaviour`) RandomForestClassifier was used to impute missing values.

### Encoding Categorical Data
- `Payment_Behaviour` and `Credit_Mix` columns were encoded using Ordinal Encoder since they contained ordinal data.
- One-Hot Encoding was applied for `Month`, `Occupation` and `Payment_of_Min_Amount` columns since they contained nominal data.
- The above two transformations and standardization on numerical columns were applied to the dataset using ColumnTransformer and Pipelines.

## Model Training
- The data was somewhat imbalanced with the majority class being `Standard`. The percentage of the class `Good` was only about 17%.
- Since the dataset is imbalanced, the F1 Score (macro) was considered the main evaluation metric. The F1 Score (macro) provides the average of F1 Scores computed for each class individually.
- Evaluated several machine learning models, including Logistic Regression, RandomForestClassifier, CatBoostClassifier, XGBoost, KNeighborsClassifier and DecisionTreeClassifier

### Hyperparameter Tuning
- The two best-performing models were RandomForestClassifier and XGBoostClassifier, and they were chosen for hyperparameter tuning.
- HalvingGridSearchCV was used for hyperparameter tuning in this project.
- Significant improvements in F1 Score were achieved for both models with hyperparameter tuning.

### Using Ensemble Model
- The dataset was trained again using the VotingClassifier algorithm with the above two tuned models of RandomForestClassifier and XGBoostClassifier as the base estimators, and the final credit score classification model was further improved.
- The VotingClassifier is a meta-estimator that combines the predictions from multiple base estimators for final classification. Different training algorithms may best capture certain parts of the data better than the other. Combining these models using the VotingClassifier in that case can improve the performance metrics of the final model.
