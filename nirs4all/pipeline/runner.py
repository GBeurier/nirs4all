from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from nirs4all.data.predictions import Predictions

from nirs4all.controllers.registry import CONTROLLER_REGISTRY
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.binary_loader import BinaryLoader
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.pipeline.io import SimulationSaver
from nirs4all.pipeline.manifest_manager import ManifestManager
from nirs4all.pipeline.serialization import deserialize_component
from nirs4all.utils.emoji import ROCKET, TROPHY, MEDAL_GOLD, FLAG, CHECK, CROSS, DIAMOND, SEARCH, REFRESH, WARNING, PLAY, SMALL_DIAMOND
from nirs4all.visualization.reports import TabReportManager

def init_global_random_state(seed: Optional[int] = None):
    import numpy as np
    import random
    import os

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    # tensflow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed if seed is not None else 42)
    except ImportError:
        pass
    # sklearn
    try:
        from sklearn.utils import check_random_state
        _ = check_random_state(seed)
    except ImportError:
        pass


class PipelineRunner:
    """PipelineRunner - Executes a pipeline with enhanced context management and DatasetView support."""

    ##TODO handle the models defined as a class
    WORKFLOW_OPERATORS = ["sample_augmentation", "feature_augmentation", "branch", "dispatch", "model", "stack",
                          "scope", "cluster", "merge", "uncluster", "unscope", "chart_2d", "chart_3d", "fold_chart",
                          "model", "y_processing", "y_chart", "split", "preprocessing"]
    SERIALIZATION_OPERATORS = ["class", "function", "module", "object", "pipeline", "instance"]

    def __init__(self, ##TODO add resume / overwrite support / realtime viz
                 max_workers: Optional[int] = None,
                 continue_on_error: bool = False,
                 backend: str = 'threading',
                 verbose: int = 0,
                 parallel: bool = False,
                 workspace_path: Optional[Union[str, Path]] = None,
                 save_files: bool = True,
                 mode: str = "train",
                 load_existing_predictions: bool = True,
                 show_spinner: bool = True,
                 enable_tab_reports: bool = True,
                 random_state: Optional[int] = None,
                 plots_visible: bool = False,
                 keep_datasets: bool = True
                 ):

        if random_state is not None:
            init_global_random_state(random_state)
        self.plots_visible = plots_visible

        # Enable interactive mode for plots if visible
        self.max_workers = max_workers or -1  # -1 means use all available cores
        self.continue_on_error = continue_on_error
        self.backend = backend
        self.verbose = verbose
        self.parallel = parallel
        self.step_number = 0  # Initialize step number for tracking
        self.substep_number = -1  # Initialize sub-step number for tracking

        # Workspace configuration
        if workspace_path is None:
            workspace_path = Path.cwd() / "workspace"
        self.workspace_path = Path(workspace_path)
        self.runs_dir = self.workspace_path / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Create other workspace directories
        (self.workspace_path / "exports").mkdir(exist_ok=True)
        (self.workspace_path / "library").mkdir(exist_ok=True)

        # Will be set per-run
        self.current_run_dir = None
        self.saver = None
        self.manifest_manager = None
        self.pipeline_uid: Optional[str] = None
        self.operation_count = 0
        self.save_files = save_files
        self.mode = mode
        self.load_existing_predictions = load_existing_predictions
        self.step_binaries: Dict[str, List[str]] = {}  # Track step-to-binary mapping
        self.binary_loader: Optional[BinaryLoader] = None
        self.show_spinner = show_spinner
        self.enable_tab_reports = enable_tab_reports
        self.prediction_metadata: Optional[Dict[str, Any]] = None
        self.config_path: Optional[str] = None
        self.target_model: Optional[Dict[str, Any]] = None
        self.model_weights: Optional[List[float]] = None
        self._capture_model: bool = False  # Flag to capture model during prediction
        self._captured_model: Optional[Any] = None  # Captured model for SHAP analysis
        self.keep_datasets = keep_datasets
        if self.keep_datasets:
            self.raw_data: Dict[str, np.ndarray] = {}
            self.pp_data: Dict[str, Dict[str, np.ndarray]] = {}

        # Store figure references to prevent garbage collection
        self._figure_refs: List[Any] = []


    def _normalize_pipeline(
        self,
        pipeline: Union[PipelineConfigs, List[Any], Dict, str],
        name: str = "",
        max_generation_count: int = 10000
    ) -> PipelineConfigs:
        """
        Normalize pipeline input to PipelineConfigs.

        Args:
            pipeline: Can be:
                - PipelineConfigs instance (return as-is)
                - List[Any]: serialized steps (wrap as PipelineConfigs)
                - Dict/str: raw definition (parse via PipelineConfigs)
            name: Optional name for the pipeline
            max_generation_count: Max combinations to generate

        Returns:
            PipelineConfigs instance
        """
        if isinstance(pipeline, PipelineConfigs):
            return pipeline

        if isinstance(pipeline, list):
            # This is a list of serialized steps
            # Wrap it in a dict with "pipeline" key to match PipelineConfigs format
            pipeline_dict = {"pipeline": pipeline}
            return PipelineConfigs(pipeline_dict, name=name, max_generation_count=max_generation_count)

        # Otherwise, it's a raw definition (Dict or str)
        return PipelineConfigs(pipeline, name=name, max_generation_count=max_generation_count)


    def _normalize_dataset(
        self,
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "array_dataset"
    ) -> DatasetConfigs:
        """
        Normalize dataset input to DatasetConfigs.

        Args:
            dataset: Can be:
                - DatasetConfigs instance (return as-is)
                - SpectroDataset instance (wrap in DatasetConfigs)
                - np.ndarray: X data only (for prediction)
                - Tuple[np.ndarray, np.ndarray]: (X, y) for training/evaluation
                - Tuple[np.ndarray, np.ndarray, Dict]: (X, y, partition_info) with partition dict
                - Dict/List[Dict]/str/List[str]: raw configs (parse via DatasetConfigs)
            dataset_name: Name to use for array-based datasets

        Returns:
            DatasetConfigs instance
        """
        if isinstance(dataset, DatasetConfigs):
            return dataset

        if isinstance(dataset, SpectroDataset):
            # Wrap existing SpectroDataset in DatasetConfigs
            # Create a synthetic config that marks this as a preloaded dataset
            configs = DatasetConfigs.__new__(DatasetConfigs)
            configs.configs = [({"_preloaded_dataset": dataset}, dataset.name)]
            configs.cache = {dataset.name: self._extract_dataset_cache(dataset)}
            return configs

        if isinstance(dataset, np.ndarray):
            # Single array - assume X only (for prediction mode)
            spectro_dataset = SpectroDataset(name=dataset_name)
            spectro_dataset.add_samples(dataset, indexes={"partition": "test"})

            configs = DatasetConfigs.__new__(DatasetConfigs)
            configs.configs = [({"_preloaded_dataset": spectro_dataset}, dataset_name)]
            configs.cache = {dataset_name: self._extract_dataset_cache(spectro_dataset)}
            return configs

        if isinstance(dataset, tuple) and len(dataset) >= 2:
            # Tuple of arrays: (X, y) or (X, y, partition_info)
            X, y = dataset[0], dataset[1]

            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise ValueError("Tuple dataset must contain numpy arrays for X and y")

            spectro_dataset = SpectroDataset(name=dataset_name)

            # Check if there's partition information
            if len(dataset) >= 3 and isinstance(dataset[2], dict):
                partition_info = dataset[2]
                # Split data according to partition info
                # Expected format: {"train": slice/indices, "test": slice/indices}
                # or {"train": int} for train_size
                if "train" in partition_info and "test" in partition_info:
                    train_idx = partition_info["train"]
                    test_idx = partition_info["test"]

                    if isinstance(train_idx, int):
                        # Assume it's train size
                        train_idx = slice(0, train_idx)
                        test_idx = slice(train_idx.stop, None)

                    X_train = X[train_idx]
                    y_train = y[train_idx]
                    X_test = X[test_idx]
                    y_test = y[test_idx]

                    spectro_dataset.add_samples(X_train, indexes={"partition": "train"})
                    spectro_dataset.add_targets(y_train)
                    spectro_dataset.add_samples(X_test, indexes={"partition": "test"})
                    spectro_dataset.add_targets(y_test)
                elif "train" in partition_info:
                    # Only train size specified
                    train_size = partition_info["train"]
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]

                    spectro_dataset.add_samples(X_train, indexes={"partition": "train"})
                    spectro_dataset.add_targets(y_train)
                    if len(X_test) > 0:
                        spectro_dataset.add_samples(X_test, indexes={"partition": "test"})
                        spectro_dataset.add_targets(y_test)
                else:
                    # No valid partition info, add all as train
                    spectro_dataset.add_samples(X, indexes={"partition": "train"})
                    spectro_dataset.add_targets(y)
            else:
                # No partition info, add all as train
                spectro_dataset.add_samples(X, indexes={"partition": "train"})
                spectro_dataset.add_targets(y)

            configs = DatasetConfigs.__new__(DatasetConfigs)
            configs.configs = [({"_preloaded_dataset": spectro_dataset}, dataset_name)]
            configs.cache = {dataset_name: self._extract_dataset_cache(spectro_dataset)}
            return configs

        # Otherwise, it's a raw config
        return DatasetConfigs(dataset)


    def _extract_dataset_cache(self, dataset: SpectroDataset) -> Tuple:
        """Extract cache tuple from a SpectroDataset for DatasetConfigs cache."""
        # Try to extract train data
        try:
            x_train = dataset.x({"partition": "train"}, layout="2d")
            y_train = dataset.y({"partition": "train"})
            # Metadata and headers are optional
            try:
                m_train = dataset.metadata({"partition": "train"})
            except:
                m_train = None
            train_headers = None  # Not easily accessible from dataset
            m_train_headers = None
        except:
            x_train = y_train = m_train = train_headers = m_train_headers = None

        # Try to extract test data
        try:
            x_test = dataset.x({"partition": "test"}, layout="2d")
            y_test = dataset.y({"partition": "test"})
            try:
                m_test = dataset.metadata({"partition": "test"})
            except:
                m_test = None
            test_headers = None
            m_test_headers = None
        except:
            x_test = y_test = m_test = test_headers = m_test_headers = None

        return (x_train, y_train, m_train, train_headers, m_train_headers,
                x_test, y_test, m_test, test_headers, m_test_headers)


    def run(
        self,
        pipeline: Union[PipelineConfigs, List[Any], Dict, str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        pipeline_name: str = "",
        dataset_name: str = "dataset",
        max_generation_count: int = 10000
    ) -> Any:
        """
        Run pipeline configurations on dataset configurations.

        Args:
            pipeline: Pipeline definition (PipelineConfigs, List[steps], Dict, or file path)
            dataset: Dataset definition (DatasetConfigs, SpectroDataset, numpy arrays, Dict, or file path)
            pipeline_name: Optional name for the pipeline
            dataset_name: Optional name for array-based datasets
            max_generation_count: Maximum number of pipeline combinations to generate

        Returns:
            Tuple of (run_predictions, datasets_predictions)
        """
        # Normalize inputs
        pipeline_configs = self._normalize_pipeline(pipeline, name=pipeline_name, max_generation_count=max_generation_count)
        dataset_configs = self._normalize_dataset(dataset, dataset_name=dataset_name)

        # Clear previous figure references
        self._figure_refs.clear()

        nb_combinations = len(pipeline_configs.steps) * len(dataset_configs.configs)
        print("=" * 120)
        print(f"\033[94m{ROCKET}Starting Nirs4all run(s) with {len(pipeline_configs.steps)} pipeline on {len(dataset_configs.configs)} dataset ({nb_combinations} total runs).\033[0m")
        print("=" * 120)

        datasets_predictions = {}
        run_predictions = Predictions()

        # Get datasets from DatasetConfigs
        for config, name in dataset_configs.configs:
            # Create run directory: workspace/runs/YYYY-MM-DD_dataset/
            date_str = datetime.now().strftime("%Y-%m-%d")
            self.current_run_dir = self.runs_dir / f"{date_str}_{name}"
            self.current_run_dir.mkdir(parents=True, exist_ok=True)

            # Initialize saver and manifest manager with run directory
            self.saver = SimulationSaver(self.current_run_dir, save_files=self.save_files)
            self.manifest_manager = ManifestManager(self.current_run_dir)

            # Load global predictions from workspace root (dataset_name.meta.parquet)
            dataset_prediction_path = self.workspace_path / f"{name}.meta.parquet"
            global_dataset_predictions = Predictions.load_from_file_cls(dataset_prediction_path)
            run_dataset_predictions = Predictions()

            for i, (steps, config_name) in enumerate(zip(pipeline_configs.steps, pipeline_configs.names)):
                dataset = dataset_configs.get_dataset(config, name)
                dataset_name = name

                # Capture raw data BEFORE any preprocessing happens
                if self.keep_datasets and dataset_name not in self.raw_data:
                    self.raw_data[dataset_name] = dataset.x({}, layout="2d")

                if self.verbose > 0:
                    print(dataset)

                config_predictions = Predictions()
                self._run_single(steps, config_name, dataset, config_predictions)
                # Capture preprocessed data AFTER preprocessing
                if self.keep_datasets:
                    if dataset_name not in self.pp_data:
                        self.pp_data[dataset_name] = {}
                    self.pp_data[dataset_name][dataset.short_preprocessings_str()] = dataset.x({}, layout="2d")

                # Merge new predictions into stores
                if config_predictions.num_predictions > 0:
                    global_dataset_predictions.merge_predictions(config_predictions)
                    run_dataset_predictions.merge_predictions(config_predictions)
                    run_predictions.merge_predictions(config_predictions)

            # Print best results for this dataset
            self.print_best_predictions(run_dataset_predictions, global_dataset_predictions, dataset, dataset_name, dataset_prediction_path)


            # Generate best score tab report
            datasets_predictions[dataset_name] = {
                "global_predictions": global_dataset_predictions,
                "run_predictions": run_dataset_predictions,
                "dataset": dataset,
                "dataset_name": dataset_name
            }

        # if self.plots_visible:
        #     import matplotlib.pyplot as plt
        #     plt.show(block=True)

        return run_predictions, datasets_predictions




    def print_best_predictions(self, run_dataset_predictions: Predictions, global_dataset_predictions: Predictions,
                               dataset: SpectroDataset, name: str, dataset_prediction_path: str):
        if run_dataset_predictions.num_predictions > 0:
            best = run_dataset_predictions.get_best(ascending=True if dataset.is_regression else False)
            print(f"{TROPHY}Best prediction in run for dataset '{name}': {Predictions.pred_long_string(best)}")
            if self.enable_tab_reports:
                best_by_partition = run_dataset_predictions.get_entry_partitions(best)
                tab_report, tab_report_csv_file = TabReportManager.generate_best_score_tab_report(best_by_partition)
                print(tab_report)
                if tab_report_csv_file:
                    # Filename: Report_best_{pipeline_id}_{model_name}_{pred_id}.csv
                    filename = f"Report_best_{best['config_name']}_{best['model_name']}_{best['id']}.csv"
                    self.saver.save_file(filename, tab_report_csv_file)
            if self.save_files:
                # Filename: Best_prediction_{pipeline_id}_{model_name}_{pred_id}.csv
                prediction_name = f"Best_prediction_{best['config_name']}_{best['model_name']}_{best['id']}.csv"
                prediction_path = self.saver.base_path / prediction_name
                Predictions.save_predictions_to_csv(best["y_true"], best["y_pred"], prediction_path)

        if global_dataset_predictions.num_predictions > 0:
            global_dataset_predictions.save_to_file(dataset_prediction_path)
        #     best_overall = global_dataset_predictions.get_best()
        #     print(f"🏆 Best prediction overall for dataset '{name}': {Predictions.pred_long_string(best_overall)}")
        #     if self.enable_tab_reports:
        #         overall_best_by_partition = global_dataset_predictions.get_entry_partitions(best_overall)
        #         tab_report, tab_report_csv_file = TabReportManager.generate_best_score_tab_report(overall_best_by_partition)
        #         print(tab_report)
        #         if tab_report_csv_file:
        #             filename = f"{datetime.now().strftime('%m-%d_%Hh%M%Ss')}_Report_best_overall_({best_overall['config_name']}_{best_overall['model_name']})_[{best_overall['id']}].csv"
        #             self.saver.save_file(filename, tab_report_csv_file, into_dataset=True)
        #     if self.save_files:
        #         prediction_name = f"{datetime.now().strftime('%m-%d_%Hh%M%Ss')}_Prediction_best_({best_overall['config_name']}_{best_overall['model_name']})_[{best_overall['id']}].csv"
        #         prediction_path = self.saver.base_path / name / prediction_name
        #         Predictions.save_predictions_to_csv(best_overall["y_true"], best_overall["y_pred"], prediction_path)
        print("=" * 120)

    def export_best_for_dataset(self, dataset_name: str, mode: str = "predictions") -> Optional[Path]:
        """Export best results for a dataset to exports/ folder.

        Delegates to SimulationSaver for file operations.

        Args:
            dataset_name: Dataset name (matches global prediction JSON filename)
            mode: Export mode - "predictions", "template", "trained", or "full"

        Returns:
            Path to export directory, or None if no predictions found
        """
        return self.saver.export_best_for_dataset(
            dataset_name,
            self.workspace_path,
            self.runs_dir,
            mode
        )


    def prepare_replay(self, selection_obj: Union[Dict[str, Any], str], dataset_config: DatasetConfigs, verbose: int = 0):
        config_path, target_model = self.saver.get_predict_targets(selection_obj)
        target_model.pop("y_pred", None)  # Remove potentially large arrays if present
        target_model.pop("y_true", None)
        self.config_path = config_path
        self.target_model = target_model
        self.model_weights = target_model['weights'] if target_model else None

        # Extract pipeline_uid from target_model if available
        pipeline_uid = target_model.get('pipeline_uid') if target_model else None

        # print(f"🚀 Starting prediction using config: {config_path} on {len(dataset_config.configs)} dataset configuration(s)."
            #   if target_model else "")

        # 2. Load pipeline configuration
        # config_path format: "dataset_name/0001_pipeline_hash"
        # Extract just the pipeline directory (0001_pipeline_hash)
        pipeline_dir_name = Path(config_path).parts[-1] if '/' in config_path or '\\' in config_path else config_path
        config_dir = self.saver.base_path / pipeline_dir_name
        pipeline_json = config_dir / "pipeline.json"

        if verbose > 0:
            print(f"{SEARCH}Loading {pipeline_json}, {config_dir / 'metadata.json'}")

        if not pipeline_json.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_json}")

        with open(pipeline_json, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)

        if isinstance(pipeline_data, dict) and "steps" in pipeline_data:
            steps = pipeline_data["steps"]
        else:
            steps = pipeline_data

        # 3. Load binaries from manifest system
        if not pipeline_uid:
            raise ValueError(
                f"No pipeline_uid found in prediction metadata. "
                f"This prediction was created with an older version of nirs4all. "
                f"Please retrain the model to use the new manifest system."
            )

        # Load from manifest
        # In the new architecture, pipelines are directly in the run directory
        from nirs4all.pipeline.manifest_manager import ManifestManager
        manifest_manager = ManifestManager(self.saver.base_path)

        manifest_path = self.saver.base_path / pipeline_uid / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Pipeline UID: {pipeline_uid}\n"
                f"The model artifacts may have been deleted or moved."
            )

        print(f"{SEARCH}Loading from manifest: {pipeline_uid}")
        manifest = manifest_manager.load_manifest(pipeline_uid)
        self.binary_loader = BinaryLoader.from_manifest(manifest, self.saver.base_path)

        return steps


    def predict(
        self,
        prediction_obj: Union[Dict[str, Any], str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "prediction_dataset",
        all_predictions: bool = False,
        verbose: int = 0
    ):
        """
        Run prediction using a saved model on new dataset.

        Args:
            prediction_obj: Reference to saved model/prediction (Dict or file path)
            dataset: Dataset definition (DatasetConfigs, SpectroDataset, numpy arrays, Dict, or file path)
            dataset_name: Optional name for array-based datasets
            all_predictions: Whether to return all predictions
            verbose: Verbosity level

        Returns:
            Predictions for the specified model
        """
        print("=" * 120)
        print(f"\033[94m{ROCKET}Starting Nirs4all prediction(s)\033[0m")
        print("=" * 120)

        # Normalize dataset input
        dataset_config = self._normalize_dataset(dataset, dataset_name=dataset_name)

        self.mode = "predict"
        self.verbose = verbose

        # Initialize saver for prediction mode
        # Extract run directory from prediction_obj
        if isinstance(prediction_obj, dict):
            if 'run_dir' in prediction_obj:
                run_dir = Path(prediction_obj['run_dir'])
            elif 'config_path' in prediction_obj:
                # config_path format: "dataset_name/0001_pipeline_hash"
                # We need to find the actual run dir: "YYYY-MM-DD_dataset_name"
                config_path = prediction_obj['config_path']
                dataset_name = Path(config_path).parts[0]  # Extract dataset name

                # Find the most recent run directory matching this dataset
                matching_dirs = sorted(
                    self.runs_dir.glob(f"*_{dataset_name}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if matching_dirs:
                    run_dir = matching_dirs[0]
                else:
                    raise ValueError(f"No run directory found for dataset: {dataset_name}")
            else:
                # Fallback: use most recent run directory
                run_dirs = sorted(self.runs_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                run_dir = run_dirs[0] if run_dirs else self.runs_dir
        else:
            # For string prediction_obj, use most recent run directory
            run_dirs = sorted(self.runs_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            run_dir = run_dirs[0] if run_dirs else self.runs_dir

        # Initialize saver and manifest manager
        self.current_run_dir = run_dir
        self.saver = SimulationSaver(run_dir, save_files=self.save_files)
        self.manifest_manager = ManifestManager(run_dir)

        steps = self.prepare_replay(prediction_obj, dataset_config, verbose=verbose)

        run_predictions = Predictions()
        for config, name in dataset_config.configs:
            dataset = dataset_config.get_dataset(config, name)
            config_predictions = Predictions()
            self._run_single(steps, "prediction", dataset, config_predictions)
            run_predictions.merge_predictions(config_predictions)
            # print(run_predictions)

        if all_predictions:
            res = {}
            for pred in run_predictions.to_dicts():
                res[pred['dataset_name']] = {}
                res[pred['dataset_name']][pred['id']] = pred['y_pred']
                return res, run_predictions



        # print(self.target_model)
        single_pred = run_predictions.get_similar(
            model_name=self.target_model.get('model_name', None),
            step_idx=self.target_model.get('step_idx', None),
            fold_id=self.target_model.get('fold_id', None),
            partition='test'  # Always return test partition for predict
        )

        if single_pred is None:
            raise ValueError("No matching prediction found for the specified model criteria. Predict failed.")

        print(f"{CHECK}Predicted with: {single_pred['model_name']} [{single_pred['id']}]")
        filename = f"Predict_[{single_pred['id']}].csv"
        y_pred = single_pred["y_pred"]
        prediction_path = self.saver.base_path / filename
        Predictions.save_predictions_to_csv(y_pred=y_pred, filepath=prediction_path)

        return single_pred["y_pred"], run_predictions

    def explain(
        self,
        prediction_obj: Union[Dict[str, Any], str],
        dataset: Union[DatasetConfigs, SpectroDataset, np.ndarray, Tuple[np.ndarray, ...], Dict, List[Dict], str, List[str]],
        dataset_name: str = "explain_dataset",
        shap_params: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        plots_visible: bool = True
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate SHAP explanations for a saved model.

        Args:
            prediction_obj: Reference to saved model/prediction (Dict or file path)
            dataset: Dataset definition (DatasetConfigs, SpectroDataset, numpy arrays, Dict, or file path)
            dataset_name: Optional name for array-based datasets
            shap_params: SHAP analysis parameters
            verbose: Verbosity level

        Returns:
            Tuple of (shap_results, output_directory)
        """
        print("=" * 120)
        print(f"\033[94m{SEARCH}Starting SHAP Explanation Analysis\033[0m")
        print("=" * 120)

        # Normalize dataset input
        dataset_config = self._normalize_dataset(dataset, dataset_name=dataset_name)

        self.mode = "explain"
        # Default SHAP parameters
        if shap_params is None:
            shap_params = {}
        shap_params.setdefault('n_samples', 200)
        shap_params.setdefault('visualizations', ['spectral', 'summary'])
        shap_params.setdefault('explainer_type', 'auto')
        shap_params.setdefault('bin_size', 20)
        shap_params.setdefault('bin_stride', 10)
        shap_params.setdefault('bin_aggregation', 'sum')

        # Step 1: Enable model capture and run prediction pipeline
        if verbose > 0:
            print("📦 Step 1: Capturing model via prediction pipeline...")

        self._capture_model = True
        self._captured_model = None

        try:
            # Run predict() - it will load the model and capture it via the flag
            config, name = dataset_config.configs[0]
            steps = self.prepare_replay(prediction_obj, dataset_config, verbose=verbose)
            dataset = dataset_config.get_dataset(config, name)
            config_predictions = Predictions()
            dataset, context = self._run_single(steps, "prediction", dataset, config_predictions)

            # Step 2: Extract the captured model
            if self._captured_model is None:
                raise ValueError("Failed to capture model during prediction. Model controller may not support capture.")

            model, controller = self._captured_model

            # Get test data with proper layout
            test_context = context.copy()
            test_context['partition'] = 'test'
            X_test = dataset.x(test_context, layout=controller.get_preferred_layout())
            y_test = dataset.y(test_context)

            # Get feature names (wavelengths if available)
            feature_names = None
            if hasattr(dataset, 'wavelengths') and dataset.wavelengths is not None:
                feature_names = [f"λ{w:.1f}" for w in dataset.wavelengths]

            # Detect task type
            task_type = 'classification' if dataset.task_type and dataset.task_type.is_classification else 'regression'

            # Create output directory
            model_id = self.target_model.get('id', 'unknown')
            output_dir = self.saver.base_path / dataset.name / self.config_path / "explanations" / model_id
            output_dir.mkdir(parents=True, exist_ok=True)

            if verbose > 0:
                print(f"📁 Output directory: {output_dir}")

            # Initialize and run SHAP analyzer
            from nirs4all.visualization.analysis.shap import ShapAnalyzer
            analyzer = ShapAnalyzer()

            shap_results = analyzer.explain_model(
                model=model,
                X=X_test,
                y=y_test,
                feature_names=feature_names,
                task_type=task_type,
                n_background=shap_params['n_samples'],
                explainer_type=shap_params['explainer_type'],
                output_dir=str(output_dir),
                visualizations=shap_params['visualizations'],
                bin_size=shap_params['bin_size'],
                bin_stride=shap_params['bin_stride'],
                bin_aggregation=shap_params['bin_aggregation'],
                plots_visible=plots_visible
            )

            # Add metadata
            shap_results['model_name'] = self.target_model.get('model_name', 'unknown')
            shap_results['model_id'] = model_id
            shap_results['dataset_name'] = dataset.name

            if verbose > 0:
                print(f"\n{CHECK}SHAP explanation completed!")
                print(f"📁 Visualizations saved to: {output_dir}")
                for viz in shap_params['visualizations']:
                    print(f"   • {viz}.png")
                print("=" * 120)

            return shap_results, str(output_dir)

        finally:
            # Always reset capture flag
            self._capture_model = False
            self._captured_model = None

    def _run_single(self, steps: List[Any], config_name: str, dataset: SpectroDataset, config_predictions: 'Predictions') -> SpectroDataset:
        """Run a single pipeline configuration on a single dataset with external prediction store."""
        # Reset runner state for each run
        self.step_number = 0
        self.substep_number = -1
        self.operation_count = 0
        self.step_binaries = {}

        print(f"\033[94m{ROCKET}Starting pipeline {config_name} on dataset {dataset.name}\033[0m")
        print("-" * 120)

        # Compute pipeline hash
        import hashlib
        import json
        pipeline_json = json.dumps(steps, sort_keys=True, default=str).encode('utf-8')
        pipeline_hash = hashlib.md5(pipeline_json).hexdigest()[:6]

        # Create pipeline with sequential numbering
        if self.mode == "train":
            pipeline_config = {"steps": steps}
            pipeline_id, pipeline_dir = self.manifest_manager.create_pipeline(
                name=config_name,
                dataset=dataset.name,
                pipeline_config=pipeline_config,
                pipeline_hash=pipeline_hash
            )
            self.pipeline_uid = pipeline_id

            # Register with SimulationSaver
            self.saver.register(pipeline_id)
        else:
            # For predict/explain modes, don't create new pipeline directories
            pipeline_id = f"temp_{pipeline_hash}"
            # Don't register in predict/explain mode - we're reusing existing pipelines

        if self.mode != "predict" and self.mode != "explain":
            self.saver.save_json("pipeline.json", steps)

        # Initialize context
        context = {"processing": [["raw"]] * dataset.features_sources(), "y": "numeric"}

        try:
            self.run_steps(steps, dataset, context, execution="sequential", prediction_store=config_predictions)
            if self.mode != "predict" and self.mode != "explain":
                self.saver.save_json("pipeline.json", steps)

                if config_predictions.num_predictions > 0:
                    pipeline_best = config_predictions.get_best(ascending=True if dataset.is_regression else False)
                    print(f"{MEDAL_GOLD}Pipeline Best: {Predictions.pred_short_string(pipeline_best)}")
                    if self.verbose > 0:
                        print(f"\033[94m{FLAG}Pipeline {config_name} completed successfully on dataset {dataset.name}\033[0m")
                    print("=" * 120)

        except Exception as e:
            print(f"\033[91m{CROSS}Pipeline {config_name} on dataset {dataset.name} failed: \n{str(e)}\033[0m")
            import traceback
            traceback.print_exc()
            raise

        return dataset, context

    def run_steps(self, steps: List[Any], dataset: SpectroDataset, context: Union[List[Dict[str, Any]], Dict[str, Any]],
                  execution: str = "sequential", prediction_store: Optional['Predictions'] = None,
                  is_substep: bool = False, mode: str = "train") -> Dict[str, Any]:
        """Run a list of steps with enhanced context management and DatasetView support."""

        if not isinstance(steps, list):
            steps = [steps]
        # print(f"\033[94m🔄 Running {len(steps)} steps in {execution} mode\033[0m")

        if execution == "sequential":
            if isinstance(context, list) and len(context) == len(steps):
                # print("🔄 Running steps sequentially with separate contexts")
                for step, ctx in zip(steps, context):
                    self.run_step(step, dataset, ctx, is_substep=is_substep)
                return context[-1]
            elif isinstance(context, dict):
                # print("🔄 Running steps sequentially with shared context")
                for step in steps:
                    context = self.run_step(step, dataset, context, prediction_store, is_substep=is_substep)
                    # print(f"🔹 Updated context after step: {context}")
                self.substep_number = -1  # Reset sub-step number after sequential execution
                return context

        elif execution == "parallel" and self.parallel:
            # print(f"🔄 Running steps in parallel with {self.max_workers} workers")
            with parallel_backend(self.backend, n_jobs=self.max_workers):
                Parallel()(delayed(self.run_step)(step, dataset, context, prediction_store, is_substep=is_substep) for step, context in zip(steps, context))

    def run_step(self, step: Any, dataset: SpectroDataset, context: Dict[str, Any], prediction_store: Optional['Predictions'] = None,
                 *, is_substep: bool = False, propagated_binaries: Any = None) -> Dict[str, Any]:
        """
        Run a single pipeline step with enhanced context management and DatasetView support.
        """
        before_dataset_str = str(dataset)

        step_description = str(step)  # Simple description for now
        if is_substep:
            self.substep_number += 1
            if self.verbose > 0:
                print(f"\033[96m   {PLAY}Sub-step {self.step_number}.{self.substep_number}: {step_description}\033[0m")
        else:
            self.step_number += 1
            self.substep_number = 0  # Reset substep counter for new main step
            self.operation_count = 0
            if self.verbose > 0:
                print(f"\033[92m{DIAMOND}Step {self.step_number}: {step_description}\033[0m")
        # print(f"🔹 Current context: {context}")
        # print(f"🔹 Step config: {step}")

        if step is None:
            if self.verbose > 0:
                print(f"{DIAMOND}No operation defined for this step, skipping.")
            return context

        try:
            operator, controller = None, None
            if isinstance(step, dict):
                if key := next((k for k in step if k in self.WORKFLOW_OPERATORS), None):
                    # print(f"📋 Workflow operation: {key}")
                    if isinstance(step[key], dict) and 'class' in step[key]:
                        if '_runtime_instance' in step[key]:
                            operator = step[key]['_runtime_instance']
                        else:
                            operator = deserialize_component(step[key])
                        controller = self._select_controller(step, keyword=key, operator=operator)
                    elif isinstance(step[key], dict) and 'function' in step[key]:
                        # Function-based operator (e.g., TensorFlow model factory)
                        if '_runtime_instance' in step[key]:
                            operator = step[key]['_runtime_instance']
                        else:
                            operator = deserialize_component(step[key])
                        controller = self._select_controller(step, keyword=key, operator=operator)
                    elif isinstance(step[key], dict):
                        # Dict without 'class' or 'function' key - try to deserialize
                        operator = deserialize_component(step[key])
                        controller = self._select_controller(step, keyword=key, operator=operator)
                    elif isinstance(step[key], str):
                        # String reference to a class/function - deserialize it
                        operator = deserialize_component(step[key])
                        controller = self._select_controller(step, keyword=key, operator=operator)
                    else:
                        # Direct operator instance (e.g., GroupKFold(), nicon)
                        operator = step[key]
                        controller = self._select_controller(step, keyword=key, operator=operator)
                elif key := next((k for k in step if k in self.SERIALIZATION_OPERATORS), None):
                    # print(f"📦 Deserializing dict operation: {key}")
                    operator = deserialize_component(step)
                    controller = self._select_controller(step, operator=operator)
                else:
                    # print(f"🔗 Running dict operation: {step}")
                    controller = self._select_controller(step)
            elif isinstance(step, list):
                # print(f"🔗 Sub-pipeline with {len(step)} steps")
                return self.run_steps(step, dataset, context, execution="sequential", is_substep=True)

            elif isinstance(step, str):
                if key := next((s for s in step.split() if s in self.WORKFLOW_OPERATORS), None):
                    # print(f"📋 Workflow operation: {key}")
                    controller = self._select_controller(key, keyword=key)
                    context["keyword"] = key  # Store keyword in context for controller use
                else:
                    # print(f"📦 Deserializing str operation: {step}")
                    operator = deserialize_component(step)
                    controller = self._select_controller(step, operator=operator, keyword=step)
                    context["keyword"] = step  # Store keyword in context for controller use

            else:
                print(f"{SEARCH}Unknown step type: {type(step).__name__}, executing as operation")
                controller = self._select_controller(step)

            if controller is not None:
                if self.verbose > 1:
                    print(f"🔹 Selected controller: {controller.__class__.__name__}")
                # Check if controller supports prediction mode
                if (self.mode == "predict" or self.mode == "explain") and not controller.supports_prediction_mode():
                    if self.verbose > 0:
                        print(f"{WARNING}Controller {controller.__class__.__name__} does not support prediction mode, skipping step {self.step_number}")
                    return context

                # Load binaries if in prediction mode
                loaded_binaries = propagated_binaries
                if (self.mode == "predict" or self.mode == "explain") and self.binary_loader is not None and loaded_binaries is None:
                    loaded_binaries = self.binary_loader.get_step_binaries(self.step_number)
                    if self.verbose > 1 and loaded_binaries:
                        print(f"{SEARCH}Loaded {', '.join(b[0] for b in loaded_binaries)} binaries for step {self.step_number}")

                context["step_id"] = self.step_number
                return self._execute_controller(
                    controller, step, operator, dataset, context, prediction_store, -1, loaded_binaries
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            if self.continue_on_error:
                print(f"{WARNING}Step failed but continuing: {str(e)}")
            else:
                raise RuntimeError(f"Pipeline step failed: {str(e)}") from e

        finally:
            if self.verbose > 0:
                print("-" * 120)
            after_dataset_str = str(dataset)
            # print(before_dataset_str)
            if before_dataset_str != after_dataset_str and self.verbose > 0:
                print(f"\033[97mUpdate: {after_dataset_str}\033[0m")
                print("-" * 120)

    def _select_controller(self, step: Any, operator: Any = None, keyword: str = ""):
        matches = [cls for cls in CONTROLLER_REGISTRY if cls.matches(step, operator, keyword)]
        if not matches:
            raise TypeError(f"No matching controller found for {step}. Available controllers: {[cls.__name__ for cls in CONTROLLER_REGISTRY]}")
        matches.sort(key=lambda c: c.priority)
        return matches[0]()

    def _execute_controller(  # TODO Choose one option for multi-source datasets and parrallel execution
        self,
        controller: Any,
        step: Any,
        operator: Any,
        dataset: SpectroDataset,
        context: Dict[str, Any],
        prediction_store: Optional['Predictions'] = None,
        source: Union[int, List[int]] = -1,
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None
    ):
        """Execute the controller for the given step and operator."""
        operator_name = operator.__class__.__name__ if operator is not None else ""
        controller_name = controller.__class__.__name__

        if self.verbose > 0:
            if operator is not None:
                print(f"{SMALL_DIAMOND}Executing controller {controller_name} with operator {operator_name}")
            else:
                print(f"{SMALL_DIAMOND}Executing controller {controller_name} without operator")

        # Prediction counting is now handled by config_predictions externally
        # prev_prediction_count = len(config_predictions) if config_predictions else 0

        # Determine if we need a spinner (for model controllers and other long operations)
        is_model_controller = 'model' in controller_name.lower()
        # needs_spinner = is_model_controller
        needs_spinner = False ####TODO DEBUG spinner \r\n

        # Execute with spinner if needed
        if needs_spinner and self.show_spinner and self.verbose == 0:  # Only show spinner when not verbose
            # Create and print the initial message
            controller_display_name = controller_name.replace('Controller', '')
            initial_message = f"{REFRESH}{controller_name} executes {controller_display_name}"

            # Only show test data shape for model controllers
            if is_model_controller:
                y_test_shape = dataset.y({"partition": "test"}).shape
                initial_message += f" (test: {y_test_shape})"

            if operator_name:
                initial_message += f" ({operator_name})"

            # Use spinner context manager for long operations
            with spinner_context(initial_message):
                context, binaries = controller.execute(
                    step,
                    operator,
                    dataset,
                    context,
                    self,
                    source,
                    self.mode,
                    loaded_binaries,
                    prediction_store
                )
        else:
            # Execute without spinner
            context, binaries = controller.execute(
                step,
                operator,
                dataset,
                context,
                self,
                source,
                self.mode,
                loaded_binaries,
                prediction_store
            )

        # Always show final score for model controllers when verbose=0
        is_model_controller = 'model' in controller_name.lower()
        # print("🔹 Controller execution completed")

        # Handle artifact metadata from controllers
        if self.mode == "train" and binaries:
            # Binaries are now List[ArtifactMeta] instead of List[Tuple[str, bytes]]
            # They've already been persisted by controllers, just append to manifest
            if self.manifest_manager and self.pipeline_uid:
                self.manifest_manager.append_artifacts(self.pipeline_uid, binaries)
                if self.verbose > 1:
                    print(f"📦 Appended {len(binaries)} artifacts to manifest")

        return context

    def next_op(self) -> int:
        """Get the next operation ID."""
        self.operation_count += 1
        return self.operation_count


