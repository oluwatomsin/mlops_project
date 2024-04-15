from src.data_loader import CsvLoader
from prefect import task, get_run_logger
import pandas as pd


@task(name = "Data Loading", log_prints=True)
def load_df(data_path: str) -> pd.DataFrame:
    """This loads a csv file as a pandas dataframe

    Args:
        data_path (str): path to the data to be loaded.

    Returns:
        pd.DataFrame: data as a pandas dataframe.
    """
    log = get_run_logger()
    try:
        loader = CsvLoader()
        log.info("Data Loading Successful.")
        return loader.load_df(data_path=data_path)
    except Exception as e:
        log.error(f"Error while loading the data {e}")
        log.debug("Please ensure that the data path was correctly specified")
        raise e
