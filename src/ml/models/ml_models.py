from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor



def get_model(name: str, **kwargs):
    """
    Return an untrained regressor by string name. Accepts kwargs for model-specific params.
    Supported: linreg, ridge, lasso, rf, extratrees, dtr, xgb, lgbm
    """
    name = name.lower()
    if name in ["linreg", "linearregression"]:
        return LinearRegression(**kwargs)
    raise ValueError(f"Unknown model type: {name}")
