import numpy as np
import pandas as pd
import pickle
import sys
import os
import glob
import pickle
import csv
import timeit

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import plots



class Model:
    
    def __init__(self, datafolder=None, param_grid=None):

        datafolder = sys.argv[1] #data/prepared
        self.df = pickle.load(open(os.path.join(datafolder, 'data.pkl'), 'rb'))
        
        self.standard_ml_model = LogisticRegression()
        
        os.makedirs('model', exist_ok=True)
        self.model_filename = sys.argv[2] #model.pkl

        os.makedirs('reports', exist_ok=True)
        os.makedirs('reports/metrics', exist_ok=True)
        self.scores_filename = os.path.join('reports', 'metrics', sys.argv[3]) #scores.csv

        os.makedirs('predicted', exist_ok=True)
        self.predictions_filename = os.path.join('predicted', sys.argv[4]) #predictions.csv


    def prepocess(self):
        """
        Preprocess the data through normalization of numeric variables and categorical transformations.
        """

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
                                transformers=[
                                    ('num', numeric_transformer, selector(dtype_exclude=object)),#self.numeric_features),
                                    ('cat', categorical_transformer, selector(dtype_include=object))#self.categorical_features)
                                ],
                                remainder='passthrough'
                            )


    def split(self, test_size):
        """
        Split the dataset into a training and test set for model building.
        """

        X = self.df[self.df.columns[:-1]]
        y = self.df['Star type']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=1)
       

    def fit(self):
        """
        Fit a Grid Search on multiple classification models. Save each model.
        """

        # Define hyperparameters per classification model
        names = [
                "Logistic Regression",
                "Random Forest",
                "Support Vector Machine",
                "KNeighbors"
                ]

        param_grid = [
                    {'classifier': [LogisticRegression()],
                    'classifier__C': [0.1, 1.0, 10]},
                    {'classifier': [RandomForestClassifier()],
                    'classifier__n_estimators': [50, 100],
                    'classifier__max_features' :['sqrt', 'log2'],
                    'classifier__max_depth': [4,6,8]},
                    {'classifier': [SVC()],
                    'classifier__C': [0.1, 1.0, 10],
                    'classifier__kernel': ['linear'],
                    'classifier__probability': [True]},
                    {'classifier': [KNeighborsClassifier()],
                    'classifier__n_neighbors': [3, 5],
                    'classifier__weights': ['uniform', 'distance']}
                    ]

        classifier_param = list(zip(names, param_grid))

        # Loop over each classifier to fit the Grid Search
        for name, params in classifier_param:
            
            # Collect run time for each model and add to filename
            start = timeit.default_timer()

            # Create classifier
            clf = params['classifier'][0]
        
            # Build pipeline
            steps = [('preprocessor', self.preprocessor),
                    ('classifier', OneVsRestClassifier(self.standard_ml_model))
                    ]

            # Fit Grid Search using cross validation
            grid_search = GridSearchCV(Pipeline(steps), param_grid=params, cv=5)
            model = grid_search.fit(self.X_train, self.y_train)

            stop = timeit.default_timer()
            runtime = round(stop - start, 3)

            # Save each model
            pickle.dump(model, open(os.path.join('model', '{}_{}_{}'.format(name, runtime, self.model_filename)), 'wb')) #model.pkl
            

    def evaluate(self):
        """
        Evaluate the performance of each model in terms of the accuracy score.
        """

        # Create output dataframe 
        output_cols = ["Classifier", "Accuracy", "Best parameters", "Run time"]
        output = pd.DataFrame(columns=output_cols)

        # load each model
        with os.scandir('model') as entries:
            for entry in entries:
                loaded_model = pickle.load(open(os.path.join(entry), 'rb'))
                
                # Get score of each model and write output to file
                score = accuracy_score(self.y_test, loaded_model.predict(self.X_test)) #loaded_model.score(self.X_test, self.y_test)

                # Extract the run time from the file name
                runtime = entry.name.split('_')[1] 
                # Extract the classifier from the file name
                classifier = entry.name.split('_')[0]
                # Write results to dataframe
                output_entry = pd.DataFrame([[classifier, score, loaded_model.best_params_, runtime]], columns=output_cols)
                output = output.append(output_entry, ignore_index=True)

        output.sort_values(by=['Accuracy', 'Run time'], inplace=True)
        output.to_csv(self.scores_filename, index=False)
    

    def predict(self):
        """
        Predict the target variable for the best performing model.
        """

        # Find best model
        with open(os.path.join(self.scores_filename), 'rb') as fd:
            df = pd.read_csv(fd)

        # Load best model (order by first row, column 'Classifier')
        with os.scandir('model') as entries:
            for entry in entries:
                if entry.name.startswith(df.iloc[0,0]):
                    best_model = pickle.load(open(os.path.join(entry), 'rb'))
        
        # Predict target variable and write to csv file
        predictions = best_model.predict(self.X_test)
        pd.DataFrame(predictions, columns=['predicted']).to_csv(self.predictions_filename, index=False)


if __name__ == '__main__':
    model_instance = Model()
    model_instance.prepocess()
    model_instance.split(0.2)
    model_instance.fit()
    model_instance.evaluate()
    model_instance.predict()

# python src/star_type_predictions.py data/prepared model.pkl scores.csv predictions.csv