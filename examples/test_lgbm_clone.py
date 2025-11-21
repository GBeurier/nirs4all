
try:
    from lightgbm import LGBMRegressor
    from sklearn.base import clone

    model = LGBMRegressor(n_estimators=20, verbose=-1, verbosity=-1)
    print(f"Original params: {model.get_params()}")

    cloned = clone(model)
    print(f"Cloned params: {cloned.get_params()}")

    print(f"Original verbose: {model.verbose}")
    print(f"Cloned verbose: {cloned.verbose}")

except ImportError:
    print("LightGBM not installed")
