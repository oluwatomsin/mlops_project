from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from abc import ABC, abstractmethod
import pandas as pd


class ModelDevelopmentTemplate(ABC):
    """This serves as a template for building different machine learning
    models"""

    @abstractmethod
    def train(X_train, X_test, y_train, y_test) -> ClassifierMixin:
        ...


class LogisticRegressor(ModelDevelopmentTemplate):
    """This class implement's the logistic regression model
    from scikit-learn"""

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> ClassifierMixin:
        """The model is instantiated and fitted to the data.

        Args:
            X_train (pd.DataFrame): Training independent variable
            y_train (pd.DataFrame): Training dependent variable


        Returns:
            ClassifierMixin: _description_
        """
        try:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            print(f"Error training model {e}")
            raise e


class RandomForestRegressor(ModelDevelopmentTemplate):
    """This implements the Random forest classification model by
    scikit-learn."""

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> ClassifierMixin:
        """The model is instantiated and fitted to the data.

        Args:
            X_train (_type_): Training independent variable
            y_train (_type_): Training dependent variable

        Returns:
            ClassifierMixin: _description_
        """

        try:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            print(f"Error training model {e}")
            raise e
