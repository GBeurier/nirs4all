from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
import json
from nirs4all.dataset.predictions import Predictions

# Load predictions
data = json.load(open('results/regression/regression_predictions.json'))
pred_obj = Predictions()
pred_obj._predictions = data
analyzer = PredictionAnalyzer(pred_obj)

# Get top 5 models
top_5 = analyzer.get_top_k(5, 'rmse')

print("Top 5 models:")
for i, model in enumerate(top_5):
    metadata = model.get('metadata', {})
    model_path = metadata.get('model_path', 'NO PATH')
    is_virtual = metadata.get('is_virtual_model', False)
    enhanced_name = model["enhanced_model_name"]
    rmse = model["metrics"]["rmse"]
    print(f'{i+1}. {enhanced_name} - RMSE: {rmse:.4f} - Virtual: {is_virtual}')
    print(f'   Path: {model_path}')
    print()