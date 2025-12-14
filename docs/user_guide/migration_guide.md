# Migration Guide for Prediction Format

This guide helps you migrate predictions from older nirs4all versions to the new format with enhanced tracking and retraining capabilities.

## What Changed

### New Fields in Predictions

The new prediction format includes additional fields for better tracking and retraining:

| Field | Description | When Added |
|-------|-------------|------------|
| `trace_id` | Unique identifier for the execution trace | v0.9+ |
| `model_artifact_id` | Reference to the saved model artifact | v0.9+ |
| `execution_hash` | Hash of the exact execution path | v0.9+ |
| `step_artifacts` | List of artifact IDs for each pipeline step | v0.9+ |

### Old Format (< v0.9)

```python
prediction = {
    'id': 'pred_12345',
    'dataset': 'wheat_2023',
    'pipeline_config': {...},
    'rmse': 0.234,
    'r2': 0.95,
    # No trace_id or artifact references
}
```

### New Format (≥ v0.9)

```python
prediction = {
    'id': 'pred_12345',
    'dataset': 'wheat_2023',
    'pipeline_config': {...},
    'rmse': 0.234,
    'r2': 0.95,
    # New fields
    'trace_id': 'trace_abc123def456',
    'model_artifact_id': 'artifact_789xyz',
    'execution_hash': 'hash_qwerty',
    'step_artifacts': [
        {'step': 0, 'artifact_id': 'art_001'},
        {'step': 1, 'artifact_id': 'art_002'},
        {'step': 2, 'artifact_id': 'art_003'},
    ]
}
```

## Impact of Changes

### Without Migration

Old predictions without the new fields will:
- ✅ Continue to work for basic operations (viewing results, loading from DB)
- ✅ Work with `predict()` if model folder still exists
- ⚠️ Not support `retrain()` with mode='transfer' or 'finetune'
- ⚠️ Not support step-level artifact control
- ⚠️ Not support bundle export with full artifact chain

### After Migration

Migrated predictions will have full access to:
- ✅ All retrain modes (full, transfer, finetune)
- ✅ Bundle export with complete artifact chain
- ✅ Step-level mode control during retrain
- ✅ Execution trace for exact reproduction

## Migration Methods

### Method 1: Automatic Migration (Recommended)

Run the migration utility on your predictions database:

```python
from nirs4all.database import PredictionsDB
from nirs4all.migration import migrate_predictions

# Load existing database
db = PredictionsDB('runs/')

# Migrate all predictions
results = migrate_predictions(
    db,
    dry_run=False,  # Set True to preview changes
    verbose=1
)

print(f"Migrated: {results['migrated']}")
print(f"Skipped (already current): {results['skipped']}")
print(f"Failed: {results['failed']}")
```

### Method 2: Individual Prediction Migration

Migrate a single prediction:

```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner(verbose=0)

# Load old prediction
old_pred = db.get('pred_12345')

# Check if migration is needed
if 'trace_id' not in old_pred:
    # Migrate by running predict
    migrated_pred = runner.migrate_prediction(old_pred)

    # Save back to database
    db.update(migrated_pred)
    print("Prediction migrated successfully")
```

### Method 3: Regenerate During Retrain

Old predictions are automatically migrated when used:

```python
from nirs4all.pipeline import PipelineRunner

runner = PipelineRunner(save_files=True, verbose=0)

# Using old prediction triggers migration
new_preds, _ = runner.retrain(
    source=old_prediction,  # Will be migrated automatically
    dataset=new_data,
    mode='full'
)
```

## Migration Script

Here's a complete migration script for batch processing:

```python
#!/usr/bin/env python
"""Migrate predictions database to new format."""

import argparse
import json
from pathlib import Path
from nirs4all.database import PredictionsDB

def migrate_prediction_record(pred: dict, runs_dir: Path) -> dict:
    """Add missing fields to a prediction record."""
    migrated = pred.copy()

    # Check if already migrated
    if 'trace_id' in pred and 'model_artifact_id' in pred:
        return None  # Already current

    # Generate trace_id from existing identifiers
    if 'trace_id' not in migrated:
        migrated['trace_id'] = f"migrated_{pred.get('id', 'unknown')}"

    # Find and link model artifact
    if 'model_artifact_id' not in migrated:
        pred_folder = runs_dir / pred.get('folder', '')
        model_file = pred_folder / 'model.joblib'
        if model_file.exists():
            migrated['model_artifact_id'] = f"artifact_{pred['id']}"
        else:
            migrated['model_artifact_id'] = None

    # Initialize step_artifacts list
    if 'step_artifacts' not in migrated:
        migrated['step_artifacts'] = []
        # Try to discover artifacts from folder
        pred_folder = runs_dir / pred.get('folder', '')
        if pred_folder.exists():
            for artifact_file in pred_folder.glob('*.joblib'):
                step_name = artifact_file.stem
                migrated['step_artifacts'].append({
                    'step': step_name,
                    'artifact_id': f"artifact_{pred['id']}_{step_name}"
                })

    return migrated


def migrate_database(db_path: str, dry_run: bool = True, verbose: int = 1):
    """Migrate all predictions in a database."""
    runs_dir = Path(db_path)
    results = {'migrated': 0, 'skipped': 0, 'failed': 0}

    # Find all prediction files
    for pred_file in runs_dir.rglob('prediction*.json'):
        try:
            with open(pred_file) as f:
                pred = json.load(f)

            migrated = migrate_prediction_record(pred, runs_dir)

            if migrated is None:
                results['skipped'] += 1
                if verbose >= 2:
                    print(f"Skipped (current): {pred_file}")
                continue

            if dry_run:
                if verbose >= 1:
                    print(f"Would migrate: {pred_file}")
                    print(f"  + trace_id: {migrated.get('trace_id')}")
                    print(f"  + model_artifact_id: {migrated.get('model_artifact_id')}")
            else:
                # Create backup
                backup_file = pred_file.with_suffix('.json.bak')
                with open(backup_file, 'w') as f:
                    json.dump(pred, f, indent=2)

                # Write migrated version
                with open(pred_file, 'w') as f:
                    json.dump(migrated, f, indent=2)

                if verbose >= 1:
                    print(f"Migrated: {pred_file}")

            results['migrated'] += 1

        except Exception as e:
            results['failed'] += 1
            if verbose >= 1:
                print(f"Failed: {pred_file} - {e}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate predictions database')
    parser.add_argument('path', help='Path to runs directory')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default is dry run)')
    parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    print(f"Migrating predictions in: {args.path}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print()

    results = migrate_database(args.path, dry_run=not args.apply, verbose=args.verbose)

    print()
    print("Summary:")
    print(f"  Migrated: {results['migrated']}")
    print(f"  Skipped:  {results['skipped']}")
    print(f"  Failed:   {results['failed']}")
```

## Checking Migration Status

Verify your predictions are migrated:

```python
from nirs4all.database import PredictionsDB

db = PredictionsDB('runs/')

# Count predictions by format
old_format = 0
new_format = 0

for pred in db.all():
    if 'trace_id' in pred and 'model_artifact_id' in pred:
        new_format += 1
    else:
        old_format += 1

print(f"Old format: {old_format}")
print(f"New format: {new_format}")

if old_format > 0:
    print(f"\n⚠️ {old_format} predictions need migration")
else:
    print("\n✅ All predictions are up to date")
```

## Backward Compatibility

The new system maintains backward compatibility:

| Feature | Old Predictions | New Predictions |
|---------|-----------------|-----------------|
| View results | ✅ Works | ✅ Works |
| Basic predict() | ✅ Works* | ✅ Works |
| retrain(mode='full') | ✅ Works* | ✅ Works |
| retrain(mode='transfer') | ⚠️ Limited | ✅ Works |
| retrain(mode='finetune') | ⚠️ Limited | ✅ Works |
| Bundle export | ⚠️ Limited | ✅ Works |
| Execution trace | ❌ Not available | ✅ Works |

*Requires model folder to still exist on disk

## Common Issues

### Missing Model Folder

```python
# Error: Model folder not found
# Solution: Old predictions without saved folders cannot be fully migrated

# Check if folder exists
from pathlib import Path
folder = Path(pred['folder'])
if not folder.exists():
    print("Original model folder missing - limited functionality")
```

### Artifact Chain Incomplete

```python
# Warning: Some step artifacts could not be located
# Solution: Run a new training with save_files=True for full functionality

runner = PipelineRunner(save_files=True, verbose=0)
new_preds, _ = runner.run(old_pred['pipeline_config'], dataset)
```

### Database Format Changed

If using the JSON predictions database:

```python
# Ensure database is using new schema
from nirs4all.database import PredictionsDB

db = PredictionsDB('runs/', schema_version='2.0')
db.upgrade_schema()  # Upgrades in place with backup
```

## Timeline

We recommend migrating predictions before these future versions:

| Version | Status | Migration Support |
|---------|--------|-------------------|
| v0.9.x | Current | Full support for old + new |
| v1.0.x | Planned | Warns on old format |
| v1.1.x | Future | Deprecates old format |
| v2.0.x | Future | Old format not supported |

## Best Practices

1. **Backup First**: Always backup your `runs/` directory before migration
2. **Dry Run**: Use `dry_run=True` to preview changes
3. **Migrate Gradually**: Migrate predictions as you use them
4. **Test After Migration**: Verify predictions still work correctly
5. **Update Workflows**: Use `save_files=True` for new runs

## Need Help?

If you encounter issues during migration:

1. Check the [troubleshooting section](#common-issues) above
2. Open an issue on [GitHub](https://github.com/GBeurier/nirs4all/issues)
3. Include your nirs4all version and error message

## See Also

- [Prediction and Model Reuse](prediction_model_reuse.md) - Prediction workflows
- [Retrain and Transfer](retrain_transfer.md) - Retrain capabilities
- [Export Bundles](export_bundles.md) - Bundle export features
- [Changelog](../../CHANGELOG.md) - Version history
