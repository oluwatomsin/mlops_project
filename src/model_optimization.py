import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
import matplotlib.pyplot as plt
import pandas as pd
import os
from optuna.visualization import (
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
    plot_optimization_history)


class RandomForestTrainer:
    """This class will house the random forest classifier and
    perform hyper-parameter tuning using optuna"""

    def __init__(
            self,
            artifact_store_path: str) -> None:

        self.artifact_store_path = artifact_store_path

    def optimizer(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series) -> ClassifierMixin:
        """This method helps to implement hyperparameter search to
        optimize the machine learning model."""

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=1000))

        # Define the objective function to be maximized

        def objective(trial: int):
            """This function is used by optuna to tune hyperparameter of our
            model."""
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            max_depth = trial.suggest_int("max_depth", 10, 50)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 32)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 32)

            # Create a model with the suggested hyperparameters

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf)

            # Evaluate the model on the training set and return the accuracy
            # of the model on the training set
            score = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="accuracy").mean()
            return score

        # Run the optimization process

        study.optimize(
            objective,
            n_trials=3,
            n_jobs=2,
            show_progress_bar=True)

        # Get the best hyperparameters
        best_params = study.best_params
        print(f"The best hyperparameters are: {best_params}")
        print(f"\nThe best accuracy is: {study.best_value}")

        # Configure the best hyperparameter for the model
        n_model = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'])

        n_model.fit(X_train, y_train)

        try:
            # Generate and save optimization history plot
            optimization_history_plot = plot_optimization_history(study)
            optimization_history_plot.write_image(os.path.join(self.artifact_store_path, "optim_hist.png"))

            # Generate and save parallel coordinate plot
            parallel_coordinate_plot = plot_parallel_coordinate(study)
            parallel_coordinate_plot.write_image(os.path.join(self.artifact_store_path, "parallel_coordinate.png"))

            # Generate and save slice plot
            slice_plot = plot_slice(study, params=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'])
            slice_plot.write_image(os.path.join(self.artifact_store_path, "slice_plot.png"))

            # Generate and save parameter importances plot
            param_importance_plot = plot_param_importances(study)
            param_importance_plot.write_image(os.path.join(self.artifact_store_path, "param_importance.png"))

            # Close all figures after saving
            plt.close('all')

        except Exception as e:
            print(f"Issue storing artifact: {e}")
            pass

        return n_model
