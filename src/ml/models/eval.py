from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_regression(model, X_train, y_train, X_val, y_val):
    """
    Fit and score model, returning metrics and predictions.
    Returns metrics dict and (y_pred_train, y_pred_val).
    """
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    metrics = {
        "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "mae_val": float(mean_absolute_error(y_val, y_pred_val)),
        "rmse_val": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        "r2_val": float(r2_score(y_val, y_pred_val)),
    }
    return metrics, y_pred_train, y_pred_val
