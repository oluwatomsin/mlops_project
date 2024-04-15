from prefect import flow, get_run_logger
import pandas as pd

# The example above is for educational purposes.
# In general, it is better to use Prefect artifacts for
# storing metrics and output.
# Logs are best for tracking progress and debugging errors.


# @flow(log_prints=True)
@flow(retries=3, retry_delay_seconds=5)
def load_data(data_path: str) -> pd.DataFrame:

    """This function help us load the dataset
    as a pandas dataframe.
    :params:
        data_path: This is the path of the dataset to be read.
    :return:
        df: The loaded data as a pandas dataframe.
    """
    try:
        df = pd.read_csv(data_path)
        logger = get_run_logger()
        logger.info(f"This dataset has a shape of {df.shape}")
        logger.info("Here is a brief information about the dataset: ")
        logger.info(f"Here is is \n {df.info()}")
        return df
    except Exception as e:
        logger.error(
            "An error occurred while loading the dataset {}".format(e)
            )
        raise e


if __name__ == "__main__":
    load_data("./data/Invistico_Airline.csv")
