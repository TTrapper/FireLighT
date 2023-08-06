import argparse

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
import torch
import umap

def load_data(fpath, labelcol=None):
    df = pd.read_csv(fpath)
    df['ftl'] = df['f'] + ' ' + df['t'] + ' ' + df['l']
    if labelcol:
        df['labels'] = df[labelcol]
    return df

def plot_projections(embeddings, df):
    reducer = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer.fit_transform(embeddings)

    df['embeddings_2d_c1'] = embeddings_3d[:, 0]
    df['embeddings_2d_c2'] = embeddings_3d[:, 1]
    df['embeddings_2d_c3'] = embeddings_3d[:, 2]

    fig = px.scatter_3d(df,
            x='embeddings_2d_c1',
            y='embeddings_2d_c2',
            z='embeddings_2d_c3',
            text='ftl', color='mike+blair')

    fig.update_traces(marker=dict(size=12))
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trainfile',
        default='ftl-train.csv',
        type=str,
        help='a csv file with columns "f", "t", "l", and "labels"'
    )
    parser.add_argument('--labelcol', default='labels', type=str, help='name of column containing labels')
    parser.add_argument(
        '--testfile',
        default='ftl-test.csv',
        type=str,
        help='a csv file with columns "f", "t", "l"'
    )
    parser.add_argument('--do_project', action='store_true', help='flag to plot embedding projections')
    args = parser.parse_args()

    df = load_data(args.trainfile, args.labelcol)
    df_test = load_data(args.testfile)

    sentences = list(df['ftl'])
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    embeddings = model.encode(sentences)
    if args.do_project:
        plot_projections(embeddings, df)

    # Set up training data
    x_train = embeddings
    y_train = df['labels'].values
    y_train = np.reshape(y_train, [-1, 1])
    y_train = np.where(y_train == 0.5, 0.0, y_train)
    # Train SVM classifier
    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(x_train, y_train)

    # Make predictions on the test set
    test_sentences = list(df_test['ftl'])
    test_embeddings = model.encode(test_sentences)
    y_pred = svm_classifier.predict_proba(test_embeddings)
    y_pred = y_pred[:, 1]

    # Sort and print test data
    sortidx = np.argsort(y_pred)
    for i in sortidx:
        print(test_sentences[i], '\t\t', y_pred[i])

