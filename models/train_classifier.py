import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['stopwords', 'punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    category_names = [x for x in df.columns.values if x not in ['id', 'message', 'original', 'genre']]
    Y = df[category_names]

    return X, Y, category_names


"""
First, the text is cleaned by removes symbols, using lowercase, and removing double spaces.
Then, the text is tokenized, stopwords are removed, and finally, the tokens are lemmatized.
"""
def tokenize(text):
    normalized = re.sub(r"[^a-zA-Z0-9]", " ", text).lower().strip()
    tokenized = word_tokenize(normalized)
    without_stop_words = [
        w for w in tokenized if w not in stopwords.words('english')
    ]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in without_stop_words:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


"""
Describes the model used on the data, consisting of NLP transformers and
an individual classifier of each category.
"""
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'clf__estimator__weights': ['uniform', 'distance'],
        'clf__estimator__n_neighbors': [2, 5, 8],
        'clf__estimator__leaf_size': [10, 20],
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


"""
Shows the accuracy, precision, and recall of the model on each category.
"""
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)

    for column in Y_test.columns:
        print(column)
        print(classification_report(Y_test[column], Y_pred_df[column]))

    print(model.best_params_)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()