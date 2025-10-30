from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

def get_model(name: str, **kwargs):
    """Return an untrained sklearn regressor by string name."""
    name = name.lower()
    if name in ["linreg", "linearregression"]:
        return LinearRegression(**kwargs)
    if name == "ridge":
        return Ridge(**kwargs)
    if name == "lasso":
        return Lasso(**kwargs)
    if name in ["rf", "randomforest", "randomforestregressor"]:
        return RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs)
    raise ValueError(f"Unknown model type: {name}")
