import io
import sys
import os
import numpy as np
import pandas as pd
import pickle


input = sys.argv[1] #data/data.csv
output = os.path.join('data', 'prepared', 'data.pkl')


def text_cleaning(input):
    df = pd.read_csv(input)

    # text cleaning
    df['Star color'] = df['Star color'].apply(lambda x: x.lower()) # lower case
    df['Star color'] = df['Star color'].apply(lambda x: x.strip()) # strip white spaces
    df['Star color'] = df['Star color'].str.replace('-', ' ') # remove '-'
    df['Star color'] = df['Star color'].replace({
                                                'yellowish white': 'white yellow', 
                                                'yellow white': 'white yellow',
                                                'yellowish': 'orange'
                                                }
                                                ) # replace string values
    return df


os.makedirs(os.path.join('data', 'prepared'), exist_ok=True)

with io.open(input, encoding='utf8') as fd_in:
    with io.open(output, 'wb') as fd_out:
        pickle.dump(text_cleaning(fd_in), fd_out)


# python src/prepare.py data/data.csv