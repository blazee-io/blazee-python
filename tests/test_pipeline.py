from __future__ import print_function

import logging
from pprint import pprint
from time import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .conftest import client


def test_pipeline(client):
    data, pipeline = create_test_pipeline()
    y = pipeline.predict(data.data)

    model = client.deploy_model(pipeline)

    p = model.predict(data.data[0])
    assert p.prediction == y[0]
    assert p.probas == None


def create_test_pipeline():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # #############################################################################
    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
    ]
    # Uncomment the following to do the analysis on all the categories
    #categories = None

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    data = fetch_20newsgroups(subset='train', categories=categories)
    print("%d documents" % len(data.filenames))
    print("%d categories" % len(data.target_names))
    print()

    # #############################################################################
    # Define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__max_iter': (5,),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__max_iter': (10, 50, 80),
    }
    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return data, grid_search.best_estimator_
