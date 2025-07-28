from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target,name='target')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the RndomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# # Run without MLflow tracking
# grid_search.fit(X_train, y_train)

# # Display the best parameters and score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print(f"Best Parameters: {best_params}")
# print(f"Best Cross-Validation Score: {best_score:.4f}")

# Start MLflow run
mlflow.set_experiment("Breast-cancer-rf-hp")

with mlflow.start_run():
    # Fit the model with GridSearchCV
    grid_search.fit(X_train, y_train)

    # Log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('accuracy',grid_search.cv_results_['mean_test_score'][i])

    # Log the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Log parameters and metrics to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cross_validation_score", best_score)
     
    # Log training data
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, 'training')

    # Log test data
    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, 'testing') 

    # Log source code
    mlflow.log_artifact(__file__)

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

    # Set tags
    mlflow.set_tag("author", "Sk Mahiduzzaman")
    

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Score: {best_score:.4f}")