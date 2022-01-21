import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np


def load_data(messages_filepath, categories_filepath):
    '''
    This function loads the dataset from given filepath
    
    Input:
    messages_filepath - Filepath where the messages dataset is stored
    categories_filepath - Filepath where the dataset containing categories is stored
    
    Output:
    df - Dataframe containing both messages and categories
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on = 'id')
    return df


def clean_data(df):
     '''
    This function cleans the given dataframe by performing string operations and data type conversions to create dummies for the 'category' values.
    
    Input:
    df - Dataframe containing both messages and categories
    
    Output: 
    df - Dataframe after cleaning
    '''
    
    categories = df['categories'].str.split(';',expand = True)
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype('str').str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    df = df.drop(['categories'],axis = 1)
    df = pd.concat([df,categories],axis = 1)
    df = df.drop_duplicates()
     
    return df


def save_data(df, database_filename):
    '''
    This function saves the given data in and sql table.
    
    Input:
    df - Dataframe containing both messages and categories
    database_filename - The name of the data base where the dataframe is to be saved
    
    Output: 
    None
    '''
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
