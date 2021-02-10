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
from sklearn.metrics import plot_confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pickle
import json
import sys


class Model:
    
    def __init__(self, datafile="../data/data.csv", model_type=None):
        self.df = pd.read_csv(datafile)
        self.df['Star color'] = self.df['Star color'].apply(lambda x: x.strip())
        
        model_type = sys.argv[1]
        self.model_type = model_type

        if model_type == 'rf':
            self.ml_model = RandomForestClassifier()
        else: 
            self.ml_model = LogisticRegression()

    
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
        
        if self.model_type == 'rf':
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__max_features' :['sqrt', 'log2'],
                'classifier__max_depth' : [4,6,8],
                }
        else:    
            param_grid = {
                'classifier__C': [0.1, 1.0, 10],
                }

        self.grid_search = GridSearchCV(self.clf, param_grid, cv=10)
    

    def fit(self):
        self.model = self.grid_search.fit(self.X_train, self.y_train)


    def save_model(self):
        self.filename = '../model/saved_model.sav'
        pickle.dump(self.model, open(self.filename, 'wb'))


    def predict(self):
        loaded_model = pickle.load(open(self.filename, 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)

        with open("../output/accuracy_score.json", 'w') as fd:
            json.dump({
                'parameters': loaded_model.best_params_,
                'accuracy': result
                },
                fd)

        if self.model_type == 'lr':
            confusion_matrix = plot_confusion_matrix(
                        self.grid_search, 
                        self.X_test, self.y_test,
                        cmap=plt.cm.Blues
                        )
            plt.savefig("../figures/confusion_matrix.png")
        return result


if __name__ == '__main__':
    model_instance = Model()
    model_instance.prepocess()
    model_instance.pipeline()
    model_instance.split(0.2)
    model_instance.gridsearch()
    model_instance.fit()
    model_instance.save_model()
    print(model_instance.predict())
