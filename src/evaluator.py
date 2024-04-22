from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Abstract class for evaluating models.
class Evaluator(ABC):
    """Abstract class for evaluating models."""

    @abstractmethod
    def evaluate(self, model, X_test, y_test):
        """Evaluate a model on a test set."""
        ...


class AccuracyEvaluator(Evaluator):
    """impementing the accuracy evaluation of the model."""

    def evaluate(self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate a model on a test set."""
        try:
            y_pred = model.predict(X_test)
            print(y_pred)
            return accuracy_score(y_test, y_pred)
        except Exception as e:
            print(f"Error while evaluating accuracy: {e}")
            raise e
    

class PrecisionEvaluator(Evaluator):
    """Implementing the precision evaluation of the model"""

    def evaluate(self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate a model on a test set."""
        try:
            y_pred = model.predict(X_test)
            return precision_score(y_test, y_pred)
        except Exception as e:
            print(f"Error while evaluating precision: {e}")
            raise e

# Implementing the recall evaluation of the model


class RecallEvaluator(Evaluator):
    """Implementing the recall evaluation of the model"""
    
    def evaluate(self, model: ClassifierMixin , X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate a model on a test set."""
        try:
            y_pred = model.predict(X_test)
            return recall_score(y_test, y_pred)
        except Exception as e:
            print(f"Error while evaluating recall: {e}")
            raise e


class FscoreEvaluator(Evaluator):
    """Implementing the F1 score evaluation of the model"""

    def evaluate(self, model: ClassifierMixin , X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate a model on a test set."""
        try:
            y_pred = model.predict(X_test)
            return f1_score(y_test, y_pred)
        except Exception as e:
            print(f"Error while evaluating F1 score: {e}")
            raise e
