import pandas as pd
from typing import Tuple
from sklearn.base import ClassifierMixin
from prefect import task, get_run_logger
from src.evaluator import AccuracyEvaluator, PrecisionEvaluator, RecallEvaluator, FscoreEvaluator



@task(name="Model Evaluation", log_prints=True)
def evaluate_model(model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float]:
    """Creating a task that performs the evaluation of the models based
    on the 4 criterias"""
    
    logger = get_run_logger()
    
    logger.info("Evaluating model...")
    
    try:
        accuracy = AccuracyEvaluator().evaluate(model, X_test, y_test)
        logger.info("Accuracy: {}".format(accuracy))
        
        f1_score = FscoreEvaluator().evaluate(model, X_test, y_test)
        logger.info("F1 score: {}".format(f1_score))
        
        precision = PrecisionEvaluator().evaluate(model, X_test, y_test)
        logger.info("Precision: {}".format(precision))
        
        recall = RecallEvaluator().evaluate(model, X_test, y_test)
        logger.info("Recall: {}".format(recall))
        
        logger.info("Evaluation complete.")
        
    except Exception as e:
        print(f"Error while evaluating model performance: {e}")
        raise e
    
    
    return accuracy, precision, recall, f1_score
