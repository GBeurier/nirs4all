# Predictions and Deployment

After training, nirs4all gives you a `RunResult` containing scores, ranked
configurations, and the ability to export trained pipelines. This page
explains how to interpret results, save models, make predictions on new data,
and adapt models to new conditions.

---

## RunResult

Every call to `nirs4all.run()` returns a `RunResult` object. It is your entry
point for everything that happened during training.

### Key properties

| Property               | What it returns                                    |
|------------------------|----------------------------------------------------|
| `result.best_score`    | Best pooled OOF score (RMSECV for regression)      |
| `result.best_rmse`     | Best RMSE (alias for regression tasks)             |
| `result.best_r2`       | Best R-squared                                     |
| `result.final`         | Refit entry (`fold_id="final"`)                    |
| `result.final_score`   | Refit model's test score (RMSEP)                   |
| `result.num_predictions` | Total number of prediction entries                |

### Ranking variants

When you use generators to produce multiple variants, `result.top(n)` gives
you the best N configurations:

```python
for entry in result.top(5):
    print(entry["pipeline_name"], entry["rmse"], entry["r2"])
```

By default, ranking uses the cross-validation score. You can switch to refit
scores:

```python
result.top(5, score_scope="final")
```

---

## Exporting a Model

The `.export()` method packages the best trained pipeline into a standalone
`.n4a` bundle file. This file contains the preprocessing chain, model
weights, and metadata -- everything needed to make predictions without the
original workspace.

```python
result.export("model.n4a")
```

The bundle is a self-contained ZIP archive. You can share it with colleagues,
deploy it to a server, or archive it for later use.

:::{tip}
The exported model includes the full preprocessing chain. When you load it for
prediction, data passes through the same transforms (scaling, SNV, etc.) that
were applied during training.
:::

See {doc}`/user_guide/deployment/export_bundles` for bundle formats and
compatibility details.

---

## Making Predictions

To apply a trained model to new data, use `nirs4all.predict()`:

```python
import nirs4all

predictions = nirs4all.predict("model.n4a", new_data)
```

`new_data` accepts the same formats as the `dataset` parameter in
`nirs4all.run()` -- a path to a folder, NumPy arrays, a dictionary, or a
SpectroDataset.

The predict function:

1. Loads the saved preprocessing chain from the bundle.
2. Applies each transform in order (using saved fitted parameters).
3. Passes the transformed data through the saved model.
4. Returns a `PredictResult` with predicted values.

---

## Explaining Predictions

SHAP-based feature importance is available through `nirs4all.explain()`:

```python
explanation = nirs4all.explain("model.n4a", data)
```

This returns an `ExplainResult` with per-feature importance values and
summary plots. It shows which wavelengths contribute most to the model's
predictions, which is valuable for understanding the underlying chemistry.

---

## Retraining

Models trained on one dataset can be adapted to new data using
`nirs4all.retrain()`. This is useful when instrument conditions change, a new
batch of samples arrives, or the model needs to be updated without starting
from scratch.

```python
new_result = nirs4all.retrain("model.n4a", new_data, mode="transfer")
```

Three retraining modes are available:

| Mode         | What it does                                         |
|--------------|------------------------------------------------------|
| `"full"`     | Retrain everything from scratch using the same pipeline definition. Preprocessing and model are both re-fitted on the new data. |
| `"transfer"` | Keep the saved preprocessing chain, retrain only the model on new data. Useful when the spectral characteristics remain similar. |
| `"finetune"` | Load existing model weights and continue training on new data. Available for models that support warm starting (deep learning, some sklearn models). |

See {doc}`/user_guide/deployment/retrain_transfer` for detailed retraining
patterns and mode selection guidance.

---

## The Session API

For workflows that go beyond a single `run()` call, the **Session API**
provides a stateful interface. A session holds a pipeline, a trained state,
and a workspace context, so you can train, predict, retrain, and save
without managing separate objects.

```python
import nirs4all

session = nirs4all.Session(
    pipeline=pipeline,
    name="WheatProtein",
)

# Train
result = session.run(dataset)

# Predict on new data
predictions = session.predict(new_data)

# Save the trained session
session.save("wheat_model.n4a")
```

A saved session can be loaded later:

```python
loaded = nirs4all.load_session("wheat_model.n4a")
predictions = loaded.predict(fresh_data)
```

Sessions are particularly useful for:

- **Interactive work** -- keep a trained model in memory and predict on demand.
- **Production pipelines** -- train, predict, and save in one coherent object.
- **Iterative refinement** -- retrain on new batches without rebuilding the
  pipeline.

See {doc}`/user_guide/predictions/session_api` for the complete Session API
reference.

---

## Workflow Summary

The typical deployment workflow looks like this:

```
1. Develop       nirs4all.run(pipeline, dataset)
                    |
                    v
2. Evaluate      result.best_score, result.top(5)
                    |
                    v
3. Export        result.export("model.n4a")
                    |
                    v
4. Deploy        nirs4all.predict("model.n4a", new_data)
                    |
                    v
5. Monitor       nirs4all.explain("model.n4a", new_data)
                    |
                    v
6. Update        nirs4all.retrain("model.n4a", new_batch, mode="transfer")
```

Each step in this workflow uses the same `.n4a` bundle as the link between
stages. The bundle carries enough information to reproduce preprocessing and
prediction without the original training environment.

---

## Next Steps

- {doc}`pipelines` -- understand pipeline definition and step types.
- {doc}`cross_validation` -- learn how `result.best` and `result.final`
  differ.
- {doc}`/user_guide/deployment/export_bundles` -- bundle format and
  compatibility.
- {doc}`/user_guide/predictions/session_api` -- stateful workflow reference.
- {doc}`/user_guide/deployment/retrain_transfer` -- retraining patterns.
