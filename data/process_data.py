import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the message and category data and returns the merged data."""

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    return df


def clean_data(df):
    """This function takes the pandas data frame and applies the following:
    1. Turns the category column into separate columns for each category with appropriate titles
    2. Turns the data values into binary numbers, dropping the non-binary ones
    3. Drops duplicate rows"""

    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = list(map(lambda col: col[:-2], row.values))
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype('int32')

    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)

    non_binary_indices = set()
    for colname in category_colnames:
        series = df[colname]
        indices = series[series > 1].index.values
        non_binary_indices.update(set(indices))
    df.drop(non_binary_indices, inplace=True)

    return df


def save_data(df, database_filename):
    """Saves the data as a database file"""

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')


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