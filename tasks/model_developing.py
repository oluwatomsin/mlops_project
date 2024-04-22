# from src.model_developer import LogisticRegressor
from src.model_optimization import RandomForestTrainer
from prefect import task, get_run_logger


# @task(name="Model Training", log_prints=True)
# def train_model(X_train, y_train):
#     """This function create a task that trains a classification model

#     Args:
#         X_train (pd.DataFrame): Training independent variable
#         y_train (pd.DataFrame): Training dependent variable
#     """
#     log = get_run_logger()
#     try:
#         reg = LogisticRegressor().train(X_train, y_train)
#         log.info("Classification model trained successfully.")
#         return reg
#     except Exception as e:
#         log.error(f"Error while training model: {e}")
#         log.debug("Ensure that the training data are in the correct format"
#                   "(i.e.pandas dataframes and series or numpy arrays.)")


@task(name="Model Training", log_prints=True)
def train_model(
        X_train,
        y_train,
        artifact_store_path: str):
    """This function create a task that trains a classification model

    Args:
        X_train (pd.DataFrame): Training independent variable
        y_train (pd.DataFrame): Training dependent variable
    """
    log = get_run_logger()
    log.info(
        "Please note that this will take a while becuase of hyperparameter "
        "tuning.")
    try:
        reg = RandomForestTrainer(
                artifact_store_path
                )
        model = reg.optimizer(
            X_train,
            y_train,)
        log.info("Classification model trained and optimized successfully.")
        log.info("Returning best performing model")
        return model
    except Exception as e:
        log.error(f"Error while training model: {e}")
        log.debug("Ensure that the training data are in the correct format"
                  "(i.e.pandas dataframes and series or numpy arrays.)")
