import xgboost as xgb
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def train_xgb_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, params: dict, evals_result: dict, model_save_path: str, num_boost_rounds: int, early_stopping_rounds: int, plot_loss_path: str, inverse_transform) -> tuple:
    """Train an XGBoost model and evaluate its performance.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        params (dict): XGBoost model parameters.
        evals_result (dict): Dictionary to store evaluation results.
        model_save_path (str): Path to save the trained model.
        num_boost_rounds (int): Number of boosting rounds.
        early_stopping_rounds (int): Early stopping rounds.
        plot_loss_path (str): Path to save the loss plot.
        inverse_transform: Function to inverse transform the predictions.

    Returns:
        tuple: Evaluation results, R-squared scores, and mean squared errors.
    """
    # Convert the data into DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train the model with evals_result to store the evaluation results
    model_reg = xgb.train(params=params,
                          dtrain=dtrain,
                          num_boost_round=num_boost_rounds,
                          evals=[(dtrain, 'train'), (dtest, 'test')],
                          early_stopping_rounds=early_stopping_rounds,
                          evals_result=evals_result)

    # Save the model
    joblib.dump(model_reg, model_save_path)

    # Make predictions
    y_pred_train = model_reg.predict(dtrain)
    y_pred_test = model_reg.predict(dtest)

    # Clip the predictions between 0 and 1
    y_pred_train = np.clip(y_pred_train, 0, 1)
    y_pred_test = np.clip(y_pred_test, 0, 1)

    # Compute R-squared
    r2_train = r2_score(inverse_transform(y_train), inverse_transform(y_pred_train))
    r2_test = r2_score(inverse_transform(y_test), inverse_transform(y_pred_test))

    print(f"R-squared (Train): {r2_train:.2f}")
    print(f"R-squared (Test): {r2_test:.2f}")

    # Compute the mean squared error
    mse_train = mean_squared_error(inverse_transform(y_train), inverse_transform(y_pred_train))
    mse_test = mean_squared_error(inverse_transform(y_test), inverse_transform(y_pred_test))

    print(f"Mean Squared Error (Train): {mse_train:.2e}")
    print(f"Mean Squared Error (Test): {mse_test:.2e}")

    # Plot the training and validation metrics
    if plot_loss_path:
        plt.figure(figsize=(8, 4))
        plt.plot(evals_result['train']['rmse'], label=f'Train R2 score: {r2_train:.2f}')
        plt.plot(evals_result['test']['rmse'], label=f'Test R2 score: {r2_test:.2f}')
        plt.xlabel('Number of boost rounds', fontsize=15)
        plt.ylabel('RMSE', fontsize=15)
        plt.title('XGBoost RMSE', fontsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig(plot_loss_path)

    return model_reg, r2_train, r2_test, mse_train, mse_test


# Example usage (optional, part of main.py usually)
if __name__ == '__main__':
    # This part is typically run from main.py
    print("This script contains the training function.")
    print("To run training, execute main.py.")