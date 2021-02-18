# Star Type Prections

## Description

This repository shows an example of how to run experiments with the data version control package DVC. 
The objective is to predict the type of a star using only a handful of variables. 
From exploratory analysis we will see that stars follow a certain graph in the celestial space by researching only this handful of variables. A representation of this graph is called the Herzsprung-Russell Diagram, or HR-diagram. Consequently, we can classify stars by plotting its features based on that graph.


## Getting Started

This repository is organized as follows: 

In the `src/` folder you find a file to prepare the data for training the model, a file to train and save the model, and a file to evaluate the results of the model. These results are written to the folder metrics/ which contains both scores and model output as well as a confusion matrix and ROC AUC curve. 
The `src/` folder also contains the full code in the file star_type_predictions.py. This file writes the predictions to the corresponding file in the predicted/ folder.

The exploratory analysis is also located in the `src/` folder. 

To run experiments with various classification models change the model parameter in the params.yaml file to the model of your interest (all lower case). The classification models that are tested for this project are: kneighbors, logistic regression, support vector machine, and random forest.   


### Prerequisites

This project makes use of DVC data version control. The raw data is stored in a personal AWS S3 bucket. To replicate this project first download the raw data from the Kaggle project website and store it in the `data/` folder.


### Installation

To get started first download this repository and create your virtual environment. 
When using poetry to create your virtual environment install the dependencies with:

```poetry install```

Otherwise install the dependencies with:

```env/bin/pip install -r requirements.txt```


To run experiments initialize the directory as a DVC folder inside a Git project. 

```dvc init``` 

To replicate an experiment run the below line of code. This code will run the pipeline 'prepare - train - evaluate' as described in dvc.yaml.

```dvc repro --no-commit```


To predict the type of a star run the makefile in the root folder. This file runs two stages: the data preparation phase and the run phase. For this step it is not necessary to have the dependencies already installed, this is included in the makefile.

```make```

	
# Usage

An example of the Herzsprung-Russell Diagram can be found in the below figure. Where the yellow dot denotes our sun as a reference point. We clearly see that star types follow a sphere and are grouped together in terms of their temperature and absolute magnitude. 

![H-R diagram](https://github.com/annalieNK/star-types/blob/main/figures/hr_diagram.png?raw=true) 

Below figure shows a correlation matrix of the numerical variables used in the dataset. From this matrix we find that the absolute magnitude is higly correlated with the type of the star, where on a scale of 1-6 1 denotes a dwarf star abd 6 denotes a hyper giant star. 

![Correlation Matrix](https://github.com/annalieNK/star-types/blob/main/figures/correlation_matrix.png?raw=true) 


# Authors and acknowledgment

Annalie Kruseman

While I'm not schooled in astronomy, astrophyscis caught my interest while recovering after a recent surgery. During this project I learned a lot more about the features of stars that are used in this dataset. In the meantime I really enjoyed reading the book 'Reality is not wat it seems' by Carlo Rovelli. With the 5 years of physics and chemistry I have had during high school it was really fun to rehearse this knowledge and understand more about the concepts of general relativity, quantum theory, and quantum gravity. 

Dataset can be found at [Kaggle](https://www.kaggle.com/deepu1109/star-dataset).

Feel free to contact me for any questions on annaliakruseman@gmail.com