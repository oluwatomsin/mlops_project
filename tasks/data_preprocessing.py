from prefect import task, get_run_logger
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from src.data_preprocessor import DataCleaningStrategy, DataSplittingStrategy


@task(name = "Data Preprocessing", log_prints=True)
def preprocess_df(data: pd.DataFrame) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"]]:
    log = get_run_logger()
    try:
        cleaner = DataCleaningStrategy()
        data = cleaner.handle_df(data=data)

        splitter = DataSplittingStrategy()
        X_train, X_test, y_train, y_test = splitter.handle_df(data=data)
        log.info("Data preprocessing completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        log.error("Error while implementing task: Data Preprocessing")
        raise e
