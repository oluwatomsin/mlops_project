from prefect import flow
from tasks.data_loading import load_df
from tasks.data_preprocessing import preprocess_df
from tasks.model_developing import train_model
from tasks.evaluation import evaluate_model
from tasks.artifact_storing import store_train_artifact


@flow(name="Mlops Training Pipeline")
def ml_training_pipeline(data_path: str):
    """This contain the flow of all the MLOps process
    from data loading, cleaning and model training.
    """

    # The data loading step
    data = load_df(data_path=data_path)

    # Data cleaning and splitting step
    X_train, X_test, y_train, y_test = preprocess_df(data)

    # Model training step
    trained_model = train_model(X_train, y_train)
    
    # model evaluation step
    accuracy, f1_score, precision, recall = evaluate_model(trained_model, X_test, y_test)
    
    # Storing the performance artifacts on the cloud
    store_train_artifact(
    accuracy=accuracy,
    f1score=f1_score,
    precision=precision,
    recall=recall,
    model_name=trained_model.__class__.__name__)
