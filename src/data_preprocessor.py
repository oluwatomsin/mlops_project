from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """This serves as the template for any data cleaning
    class.
    """
    @abstractmethod
    def handle_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """The abstract function for cleaning the data"""
        ...


class DataCleaningStrategy(DataStrategy):
    """This class will handle cleaning of our dataset"""

    def handle_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """This function performs the data cleaning process

        Returns:
            data: as a pandas dataframe
        """

        try:
            data['satisfaction'] = data['satisfaction'].map({
                "satisfied": 1,
                "dissatisfied": 0
            })
            data["Gender"] = data["Gender"].map({
                "Female": 0,
                "Male": 1
            })
            data["Customer Type"] = data["Customer Type"].map({
                "Loyal Customer": 1,
                "disloyal Customer": 0
            })
            data["Type of Travel"] = data["Type of Travel"].map({
                "Personal Travel": 0,
                "Business travel": 1
            })
            data["Class"] = data["Class"].map({
                "Eco": 0,
                "Eco Plus": 1,
                "Business": 2
            })

            data = data.dropna()
            print("Data cleaning task completed successfully")
            return data
        except Exception as e:
            print(f"Error while cleaning dataset {e}")
            raise e


class DataSplittingStrategy(DataStrategy):
    """This implements the data splitting"""

    def handle_df(self, data: pd.DataFrame) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"]
    ]:
        try:
            X = data.drop("satisfaction", axis=1)
            y = data['satisfaction']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=True, random_state=100
            )
            print("Data splitting executed successful")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"Error while splitting data {e}")
            raise e
