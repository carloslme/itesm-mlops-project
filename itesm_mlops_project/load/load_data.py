import pandas as pd
import numpy as np
import re
import os

class DataRetriever:
    """
    A class for retrieving data from a given URL and processing it for further analysis.

    Parameters:
        url (str): The URL from which the data will be loaded.

    Attributes:
        url (str): The URL from which the data will be loaded.

    Example usage:
    ```
    URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
    data_retriever = DataRetriever(URL, DATASETS_DIR)
    result = data_retriever.retrieve_data()
    print(result)
    ```
    """

    DROP_COLS = ['name', 'ticket', 'boat', 'body', 'home.dest']
    # DATASETS_DIR = './data/'  # Directory where data will be saved.
    RETRIEVED_DATA = 'retrieved_data.csv'  # File name for the retrieved data.

    def __init__(self, url, data_path):
        self.url = url
        self.DATASETS_DIR = data_path

    def _get_first_cabin(self, row):
        """
        Helper function to extract the first cabin from a row.

        Parameters:
            row (str): The row containing cabin information.

        Returns:
            str: The first cabin from the row, or np.nan if not found.
        """
        try:
            return row.split()[0]
        except Exception:
            return np.nan

    def _get_title(self, passenger):
        """
        Helper function to extract the title from a passenger's name.

        Parameters:
            passenger (str): The name of the passenger.

        Returns:
            str: The title extracted from the passenger's name.
        """
        line = passenger
        if re.search('Mrs', line):
            return 'Mrs'
        elif re.search('Mr', line):
            return 'Mr'
        elif re.search('Miss', line):
            return 'Miss'
        elif re.search('Master', line):
            return 'Master'
        else:
            return 'Other'

    def retrieve_data(self):
        """
        Retrieves data from the specified URL, processes it, and stores it in a CSV file.

        Returns:
            str: A message indicating the location of the stored data.
        """
        # Loading data from specific URL
        data = pd.read_csv(self.url)

        # Uncovering missing data
        data.replace('?', np.nan, inplace=True)
        data['age'] = data['age'].astype('float')
        data['fare'] = data['fare'].astype('float')

        # Extract the first cabin | Extract the title from 'name'
        data['cabin'] = data['cabin'].apply(self._get_first_cabin)
        data['title'] = data['name'].apply(self._get_title)

        # Drop irrelevant columns
        data.drop(self.DROP_COLS, axis=1, inplace=True)

        # Create directory if it does not exist
        if not os.path.exists(self.DATASETS_DIR):
            os.makedirs(self.DATASETS_DIR)
            print(f"Directory '{self.DATASETS_DIR}' created successfully.")
        else:
            print(f"Directory '{self.DATASETS_DIR}' already exists.")

        # Save data to CSV file
        data.to_csv(self.DATASETS_DIR + self.RETRIEVED_DATA, index=False)

        return f'Data stored in {self.DATASETS_DIR + self.RETRIEVED_DATA}'

# Usage Example:
# URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
# data_retriever = DataRetriever(URL, "./itesm_mlops_project/data/")
# result = data_retriever.retrieve_data()
# print(result)
