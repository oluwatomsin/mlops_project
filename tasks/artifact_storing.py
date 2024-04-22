from prefect import task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# This task is being used to store the performance artifact.
# and also data used to train the model


@task(name="Store Performance Artifact", log_prints=True)
def store_train_artifact(
    model_name: str,
    accuracy: float,
    f1score: float,
    precision: float,
    recall: float,
):
    """This task is being used to store the performance artifact."""

    log = get_run_logger()

    try:
        log.info(
            f"Attempting to store the model {model_name} performance artifact."
            )

        create_markdown_artifact(
            key="performance-report",
            markdown=f"""
        ## Summary

        Below is a table that shows the performance of model: {model_name}.

        ## Metrics

        | Metric Name   | score |
        |:--------------|-------:|
        | Accuracy      | {round(accuracy * 100, 2)}% |
        | Accuracy      | {round(accuracy * 100, 2)}% |
        | Precision     | {round(precision * 100, 2)}% |
        | recall        | {round(recall * 100, 2)}% |
        | F1 score      | {round(f1score * 100, 2)}% |
        """,
            description="Model Performance Report")
        log.info(
            f"Successfully stored the model {model_name}"
            "performance artifact.")
    except Exception as e:
        log.exception(
            f"Failed to store the model {model_name} performance artifact."
            )
        log.info(f"Exception: {e}")
        raise e
