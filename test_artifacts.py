"""Test if artifacts are being saved correctly."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from nirs4all.pipeline import PipelineRunner
import yaml

# Create simple data
np.random.seed(42)
X = np.random.randn(50, 20)
y = np.random.randn(50)

# Train with save_files=True
print("Training with save_files=True...")
runner = PipelineRunner(save_files=True, verbose=1)
pipeline = [StandardScaler(), Ridge()]
predictions, _ = runner.run(pipeline, (X, y, {'train': 40}))

# Check if manifest was created
print("\nChecking for manifest files...")
runs = sorted((runner.workspace_path / 'runs').glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
print(f'Most recent run: {runs[0] if runs else "None"}')

if runs:
    manifests = sorted(runs[0].glob('*/manifest.yaml'), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f'Manifests found: {len(manifests)}')

    if manifests:
        print(f'\nReading most recent manifest: {manifests[0]}')
        with open(manifests[0]) as f:
            manifest = yaml.safe_load(f)

        artifacts = manifest.get('artifacts', [])
        print(f'Artifacts in manifest: {len(artifacts)}')

        for art in artifacts[:10]:
            skipped = art.get("skipped", False)
            reason = art.get("reason", "")
            print(f'  - {art.get("name")} (step {art.get("step")}, skipped={skipped}, reason="{reason}")')

        # Test prediction
        print("\n\nTesting prediction...")
        best = predictions.top(n=1)[0]
        print(f"Best model: {best['model_name']} (ID: {best['id']})")

        X_new = np.random.randn(5, 20)
        predictor = PipelineRunner(save_files=False, verbose=1)
        y_pred, _ = predictor.predict(best, X_new, dataset_name="new_data")
        print(f"\nPredictions shape: {y_pred.shape}")
        print(f"Predictions: {y_pred[:5]}")
        print("\n✅ Prediction successful!")

    else:
        print("No manifests found!")
else:
    print("No run directories found!")
