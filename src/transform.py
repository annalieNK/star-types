from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector


def preprocess():
    numeric_transformer = StandardScaler(with_mean=True, with_std=True)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', numeric_transformer, selector(dtype_exclude=object)),#self.numeric_features),
                                ('cat', categorical_transformer, selector(dtype_include=object))#self.categorical_features)
                            ],
                            remainder='passthrough'
                        )
    return preprocessor


preprocess()