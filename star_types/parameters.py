def parameters(model_type):

    if model_type == 'random forest':
        param_grid = {
            'classifier__estimator__n_estimators': [50, 100],
            'classifier__estimator__max_features' :['sqrt', 'log2'],
            'classifier__estimator__max_depth' : [4,6,8]
        }
    elif model_type == 'logistic regression':    
        param_grid = {
            'classifier__estimator__C': [0.1, 1.0, 10]
        }
    elif model_type == 'support vector machine':
        param_grid = {
            'classifier__estimator__C': [0.1, 1.0, 10],
            'classifier__estimator__kernel': ['linear'],
            'classifier__estimator__probability': [True]
        }
    elif model_type == 'kneighbors':
        param_grid = {
            'classifier__estimator__n_neighbors': [1, 3, 5],
            'classifier__estimator__weights': ['uniform', 'distance']
        }
    
    return param_grid