from nirs4all.dataset.predictions import Predictions
p = Predictions()
p.load_from_file('results/regression/predictions.json')
result = p.top_k(k=5, metric='rmse')
print(f'Top 5 RMSE scores on test: {[round(float(r["computed_score"]), 4) for r in result]}')
print(f'Models: {[r["model_name"] for r in result]}')