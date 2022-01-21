import sys
from sqlalchemy import create_engine
import nltk
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import re
import pickle
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    
     '''
    This function loads the dataset from given database
    
    Input:
    database_filepath - Filepath of the database where the data is saved
    
    Output:
    X - Features for the classification model to be built
    y - Target Variable for the classification model to be built
    '''
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster', con=engine)
    X = df['message']
    y = df.iloc[:,4:]
    return X,y


def tokenize(text):
    
    '''
    This function performs a series of text proceesing on the given piece of text
    
    Input:
    text - The text which is to be processed
    
    Output:
    clean_words - Tokens of given text after processing and cleaning
    '''
    
    words = word_tokenize(text) # To tokenize the text
    lemmatizer = WordNetLemmatizer() # Initialization of Lemmatizer object
    clean_words = []
    for word in words:
        clean_words.append(lemmatizer.lemmatize(word).lower().strip()) # Lemmatizes, case normalizes and strips each word token.
    return clean_words

def build_model():
    
     '''
    This function builds a pipeline which entails the classification model. The pipeline contains a CountVectorizer,TfidfTransformer, RandomForestClassifier. 
    The pipeline also undergoes a grid search for hyper tuning and is returned with its best parameters.
   
    Input:
    None
    
    Output:
    cv - The classifier model post Grid search.
    '''
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer = tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100]
        }


    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    
    '''
    This function evaluates a given model.
    
    Input:
    model : The model to be evaluated
    X_test : The features in test dataset.
    Y_test : Target variable values in the dataset
    
    Output:
    None : Prints the accuracy of the model
    '''
    
    y_pred = model.predict(X_test)
    for i, column in enumerate(y_test):
        print('Category:',column,classification_report(y_test[column],y_pred[:,i]))
    accuracy = (y_pred == y_test).mean()
    print('Accuracy of the model: ',accuracy)

def save_model(model, model_filepath):
    '''
    This function save the model in a pickle file
    
    Input:
    model - The model to be saved
    model_filepath - The filepath of the pickle file where the data is to be stored
    
    Output:
    None
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
