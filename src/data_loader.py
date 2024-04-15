import pandas as pd
from abc import ABC, abstractmethod


class DataLoaderStrategy(ABC):
    """Data Loading strategy. The idea is
    we should be able to use different methods for loading
    our data. Hence, its like a data loading template that can
    be implemented in multiple ways"""

    @abstractmethod
    def load_df(self, data_path: str) -> pd.DataFrame:
        """Load the data as a pandas dataframe

        Args:
            data_path (str): Path to the dataset

        Returns:
            pd.DataFrame: data as pandas dataframe
        """
        ...


class CsvLoader(DataLoaderStrategy):
    """This implements the strategy for loading csv files

    Args:
        DataLoaderStrategy (ABC): Inherited loading strategy
    """

    def load_df(self, data_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(data_path)
            print("Data successfully loaded as a dataframe")
            return data
        except Exception as e:
            print(f"Error while loading data {e}")
            raise e
