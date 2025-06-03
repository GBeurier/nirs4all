import logging
import hashlib

from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self):
        self.status = None

    def run_pipeline(self, pipeline, data, prefix=""):
        self.status = "running"
        print("Running pipeline")
        context = {
            "branch": 0,
            "group_filter": False,
            "processing": None,
            "augmentation": None,
        }
        for step in pipeline:
            self.run(step, data, context, prefix=prefix)
        self.status = "done"

    def _handle_sklearn_estimator(self, step, data, context, prefix=""):
        """Handle sklearn estimators - fit on train data."""
        def p(msg, end="\n"):
            print(prefix + msg, end=end)
        
        p(f"sklearn > {step.__class__.__name__}")
        X_train = data.X(set="train")
        y_train = data.y(set="train")
        
        if hasattr(step, 'fit'):
            step.fit(X_train, y_train)
            p(f"  Fitted on {len(X_train)} training samples")

    def _handle_transformer(self, step, data, context, prefix=""):
        """Handle sklearn transformers - fit on train, transform train and test."""
        def p(msg, end="\n"):
            print(prefix + msg, end=end)
        
        p(f"Transforming with {step.__class__.__name__}")
        
        # Get train and test data
        X_train = data.X(set="train", pad=True)
        X_test = data.X(set="test", pad=True) if len(data._select(set="test")) > 0 else None
        
        # Fit on train data
        step.fit(X_train)
        p(f"  Fitted transformer on {len(X_train)} training samples")
        
        # Transform train data
        X_train_transformed = step.transform(X_train)
        data.change_spectra(X_train_transformed.tolist(), set="train")
        p(f"  Transformed {len(X_train_transformed)} training samples")
        
        # Transform test data if exists
        if X_test is not None:
            X_test_transformed = step.transform(X_test)
            data.change_spectra(X_test_transformed.tolist(), set="test")
            p(f"  Transformed {len(X_test_transformed)} test samples")
        
        # Update processing hash
        import hashlib
        processing_hash = hashlib.md5(f"{step.__class__.__name__}_{context.get('processing', 'raw')}".encode()).hexdigest()[:8]
        data.set_tag("processing", processing_hash)

    def _handle_clustering(self, step, data, context, prefix=""):
        """Handle clustering - fit on data and add cluster labels."""
        def p(msg, end="\n"):
            print(prefix + msg, end=end)
        
        p(f"Clustering with {step.__class__.__name__}")
        X = data.X(pad=True)
        
        if hasattr(step, 'fit_predict'):
            labels = step.fit_predict(X)
        elif hasattr(step, 'fit') and hasattr(step, 'labels_'):
            step.fit(X)
            labels = step.labels_
        else:
            p(f"  Cannot cluster with {step.__class__.__name__}")
            return
        
        data.add_tag("cluster", labels.tolist())
        context["group"] = True
        p(f"  Created {len(set(labels))} clusters for {len(X)} samples")

    def _handle_sample_augmentation(self, transforms, data, context, prefix=""):
        """Handle sample augmentation - copy samples and apply transforms."""
        def p(msg, end="\n"):
            print(prefix + msg, end=end)
        
        for transform in transforms:
            if transform is None:
                p("  Identity augmentation")
                continue
            
            p(f"  Sample augmentation with {transform.__class__.__name__}")
            
            # Get original training samples
            train_data = data._select(set="train", augmentation="raw")
            if len(train_data) == 0:
                p("    No raw training samples to augment")
                continue
            
            X_orig = data.X(set="train", augmentation="raw", pad=True)
            
            # Apply augmentation
            if hasattr(transform, 'fit_transform'):
                X_aug = transform.fit_transform(X_orig)
            elif hasattr(transform, 'transform'):
                X_aug = transform.transform(X_orig)
            else:
                p(f"    Cannot augment with {transform.__class__.__name__}")
                continue
            
            # Add augmented samples back to dataset
            for i, spec in enumerate(X_aug):
                orig_row = train_data.row(i)
                data.add_spectra(
                    [spec.tolist()], 
                    target=data.y(source_id=orig_row['source_id'])[0],
                    origin=orig_row['origin'],
                    sample=orig_row['sample'],
                    type=orig_row['type'],
                    set="train",
                    processing=orig_row['processing'],
                    branch=orig_row['branch'],
                    augmentation=f"aug_{transform.__class__.__name__}"
                )
            
            p(f"    Added {len(X_aug)} augmented samples")

    def _handle_feature_augmentation(self, transforms, data, context, prefix=""):
        """Handle feature augmentation - fit on train, transform all."""
        def p(msg, end="\n"):
            print(prefix + msg, end=end)
        
        for transform in transforms:
            p(f"  Feature augmentation with {transform.__class__.__name__}")
            
            # Get all data for transformation
            X_train = data.X(set="train", pad=True)
            X_all = data.X(pad=True)
            
            # Fit on training data only
            if hasattr(transform, 'fit'):
                transform.fit(X_train)
                p(f"    Fitted on {len(X_train)} training samples")
            
            # Transform all data
            if hasattr(transform, 'transform'):
                X_transformed = transform.transform(X_all)
                
                # Update all spectra with transformed features
                all_rows = data._select()
                for i, spec in enumerate(X_transformed):
                    row = all_rows.row(i)
                    data.change_spectra([spec.tolist()], row_id=row['row_id'])
                
                p(f"    Transformed {len(X_transformed)} total samples")
            
            # Update processing hash
            import hashlib
            processing_hash = hashlib.md5(f"feat_{transform.__class__.__name__}_{context.get('processing', 'raw')}".encode()).hexdigest()[:8]
            data.set_tag("processing", processing_hash)

    def _handle_data_splitter(self, step, data, context, prefix=""):
        """Handle data splitting."""
        def p(msg, end="\n"):
            print(prefix + msg, end=end)
        
        p(f"Splitting with {step.__class__.__name__}")
        
        # Get unique samples (by source_id)
        if data.labels is None:
            p("  No labels to split")
            return
        
        source_ids = data.labels["source_id"].to_list()
        y = data.labels["target"].to_list()
        
        if hasattr(step, 'split'):
            splits = list(step.split(source_ids, y))
            for fold, (train_idx, test_idx) in enumerate(splits):
                train_sources = [source_ids[i] for i in train_idx]
                test_sources = [source_ids[i] for i in test_idx]
                
                # Update set labels
                data.set_tag("set", "train", source_id=train_sources)
                data.set_tag("set", "test", source_id=test_sources)
                data.add_tag("fold", fold)
                
                p(f"  Fold {fold}: {len(train_sources)} train, {len(test_sources)} test")
                break  # For now, just use first split
    def run(self, step, data, context, prefix=""):
        def p(msg, end="\n"):
            print(prefix + msg, end=end)

        if isinstance(step, str):
            if step == "uncluster":
                context["group_filter"] = False
                p(f"Control step: {step}")
            else:
                p(f"Unknown string step: {step}")
            return

        if isinstance(step, dict) and len(step) == 1:
            key, spec = next(iter(step.items()))
            p(f"Control step: {key}")

            if key == "cluster":
                self._handle_clustering(spec, data, context, prefix=prefix + "  ")
            elif key in ("sample_augmentation", "samples", "S"):
                self._handle_sample_augmentation(spec, data, context, prefix=prefix + "  ")
            elif key in ("feature_augmentation", "features", "F"):
                self._handle_feature_augmentation(spec, data, context, prefix=prefix + "  ")
            elif key == "branch":
                for branch in spec:
                    self.run(branch, data, context, prefix=prefix + "  ")
                    context["branch"] += 1
            elif key == "stack":
                self.run_model(spec, data, context, prefix=prefix + "  ")
                for model in spec["base_learners"]:
                    self.run_model(model, data, context, prefix=prefix + "  ")
            else:
                p(f"Unknown step: {key}")
            return

        if isinstance(step, dict) and len(step) > 1:
            if "model" in step:
                self.run_model(step, data, context, prefix=prefix)
            elif "cluster" in step:
                self._handle_clustering(step, data, context, prefix=prefix + "  ")
            else:
                p(f"Unknown step: {step}")
            return

        if isinstance(step, list):
            p("Sub-pipeline")
            for sub_step in step:
                self.run(sub_step, data, context, prefix=prefix + "  ")
            return

        # Handle individual components based on their type
        if hasattr(step, "transform") and issubclass(step.__class__, TransformerMixin):
            self._handle_transformer(step, data, context, prefix=prefix)
        elif hasattr(step, "split"):
            self._handle_data_splitter(step, data, context, prefix=prefix)
        elif hasattr(step, "fit") and issubclass(step.__class__, ClusterMixin):
            self._handle_clustering(step, data, context, prefix=prefix)
        elif issubclass(step.__class__, BaseEstimator):
            self._handle_sklearn_estimator(step, data, context, prefix=prefix)
        elif issubclass(step.__class__, tf.keras.Model):
            p(f"tensorflow > {step.__class__.__name__}")
            # TODO: Implement TensorFlow model handling
        else:
            p(f"Unknown step: {step}")    
    def run_model(self, step, data, context, prefix=""):
        def p(msg, end="\n"):
            print(prefix + msg, end=end)
        
        p("Model step:")
        
        # Handle model component
        model = step["model"]
        if hasattr(model, 'fit') and issubclass(model.__class__, BaseEstimator):
            self._handle_sklearn_estimator(model, data, context, prefix=prefix + "  ")
        else:
            self.run(model, data, context, prefix=prefix + "  ")
        
        # Handle y_pipeline (target preprocessing)
        if "y_pipeline" in step:
            p("  Target pipeline:")
            self.run(step["y_pipeline"], data, context, prefix=prefix + "    ")
        
        # Handle training parameters
        if "train_params" in step:
            p(f"  > Training: {step['train_params']}")
            
        if "finetune_params" in step:
            p(f"  > Finetuning: {step['finetune_params']}")
            
        # Perform actual training if model supports it
        if hasattr(model, 'fit'):
            X_train = data.X(set="train", pad=True)
            y_train = data.y(set="train")
            
            if len(X_train) > 0 and len(y_train) > 0:
                model.fit(X_train, y_train)
                p(f"  Model trained on {len(X_train)} samples")
                
                # Generate predictions for evaluation
                if hasattr(model, 'predict'):
                    # Predict on test set if available
                    X_test = data.X(set="test", pad=True)
                    if len(X_test) > 0:
                        y_pred = model.predict(X_test)
                        # Store predictions (simplified - in real case would need fold/seed info)
                        data.add_prediction(
                            model=model.__class__.__name__, 
                            fold=0, 
                            seed=42, 
                            preds=y_pred.tolist(),
                            set="test"
                        )
                        p(f"  Generated predictions for {len(y_pred)} test samples")
            else:
                p(f"  No training data available")

    
