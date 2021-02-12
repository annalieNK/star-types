import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
import json
import sys
import os
import pickle
import csv


class Model:
    
    def __init__(self, datafolder=None, model_type=None):
        datafolder = sys.argv[1] #data/prepared
        self.df = pickle.load(open(os.path.join(datafolder, 'data.pkl'), 'rb'))
        
        model_type = sys.argv[2] #logistic_regression
        self.model_type = model_type

        if model_type == 'random_forest':
            self.ml_model = RandomForestClassifier()
        elif model_type == 'logistic_regression': 
            self.ml_model = LogisticRegression()

        os.makedirs('model', exist_ok=True)
        self.model_filename = os.path.join('model', sys.argv[3]) #final_model.pkl

        os.makedirs('predicted', exist_ok=True)
        self.predictions_filename = os.path.join('predicted', 'predictions.csv')
    

    def prepocess(self):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
                                transformers=[
                                    ('num', numeric_transformer, selector(dtype_exclude=object)),#self.numeric_features),
                                    ('cat', categorical_transformer, selector(dtype_include=object))#self.categorical_features)
                                ],
                                remainder='passthrough'
                            )
   

    def pipeline(self):
        self.clf = Pipeline(
                steps=[
                    ('preparation', self.preprocessor),
                    ('classifier', self.ml_model) 
                ]
        )


    def split(self, test_size):
        X = self.df[self.df.columns[:-1]]
        y = self.df['Spectral Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=1)
       

    def gridsearch(self):
        
        if self.model_type == 'random_forest':
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__max_features' :['sqrt', 'log2'],
                'classifier__max_depth' : [4,6,8],
            }
        elif self.model_type == 'logistic_regression':    
            param_grid = {
                'classifier__C': [0.1, 1.0, 10],
            }

        self.grid_search = GridSearchCV(self.clf, param_grid, cv=10)
    

    def fit(self):
        self.model = self.grid_search.fit(self.X_train, self.y_train)


    def save_model(self):
        self.model_filename
        pickle.dump(self.model, open(self.model_filename, 'wb'))


    def predict(self):
        loaded_model = pickle.load(open(self.model_filename, 'rb'))
        predictions = loaded_model.predict(self.X_test)
        pd.DataFrame(predictions, columns=['predicted']).to_csv(self.predictions_filename, index=False)


if __name__ == '__main__':
    model_instance = Model()
    model_instance.prepocess()
    model_instance.pipeline()
    model_instance.split(0.2)
    model_instance.gridsearch()
    model_instance.fit()
    model_instance.save_model()
    model_instance.predict()
