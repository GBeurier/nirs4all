from nirs4all.controllers.models.factory import ModelFactory

try:
    cls = ModelFactory.import_class('sklearn.cross_decomposition._pls.PLSRegression')
    print(f"Success: {cls}")
except Exception as e:
    print(f"Error: {e}")
