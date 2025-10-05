"""
PipelineHistory - Execution tracking and serialization support

Tracks pipeline execution with detailed logging and provides
methods to save/load fitted operations and execution state.
"""
import json
import pickle
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import uuid


@dataclass
class StepExecution:
    """Record of a single step execution"""
    step_id: str
    step_number: int
    step_type: str
    step_description: str
    operation_name: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    status: str  # 'running', 'completed', 'failed'
    error_message: Optional[str]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


@dataclass
class PipelineExecution:
    """Record of complete pipeline execution"""
    execution_id: str
    pipeline_config: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime]
    total_duration_seconds: Optional[float]
    status: str  # 'running', 'completed', 'failed'
    steps: List[StepExecution]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.end_time and self.start_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()


class PipelineHistory:
    """
    Tracks pipeline execution history and provides serialization capabilities

    Features:
    - Step-by-step execution logging
    - Fitted operation storage
    - Pipeline state serialization
    - Export/import capabilities
    """

    def __init__(self):
        self.executions: List[PipelineExecution] = []
        self.current_execution: Optional[PipelineExecution] = None
        self.fitted_operations: Dict[str, Any] = {}  # step_id -> fitted operation

    def start_execution(self, pipeline_config: Dict, metadata: Optional[Dict] = None) -> str:
        """Start tracking a new pipeline execution"""
        execution_id = str(uuid.uuid4())

        self.current_execution = PipelineExecution(
            execution_id=execution_id,
            pipeline_config=pipeline_config,
            start_time=datetime.now(),
            end_time=None,
            total_duration_seconds=None,
            status='running',
            steps=[],
            metadata=metadata or {}
        )

        return execution_id

    def start_step(self, step_number: int, step_description: str, step_config: Any,
                   step_type: Optional[str] = None, operation_name: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> StepExecution:
        """Start tracking a step execution"""
        if not self.current_execution:
            raise RuntimeError("No active execution. Call start_execution first.")

        step_id = f"{self.current_execution.execution_id}_step_{step_number}"

        # Determine step type if not provided
        if step_type is None:
            if isinstance(step_config, dict):
                step_type = next(iter(step_config.keys())) if step_config else "operation"
            elif isinstance(step_config, list):
                step_type = "sub_pipeline"
            else:
                step_type = "operation"

        step_exec = StepExecution(
            step_id=step_id,
            step_number=step_number,
            step_type=step_type,
            step_description=step_description,
            operation_name=operation_name,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            status='running',
            error_message=None,
            metadata=metadata or {}
        )

        self.current_execution.steps.append(step_exec)
        return step_exec

    def complete_step(self, step_id: str, metadata: Optional[Dict] = None):
        """Mark a step as completed"""
        if not self.current_execution:
            return

        for step in self.current_execution.steps:
            if step.step_id == step_id:
                step.end_time = datetime.now()
                step.status = 'completed'
                if metadata:
                    step.metadata.update(metadata)
                break

    def fail_step(self, step_id: str, error_message: str, metadata: Optional[Dict] = None):
        """Mark a step as failed"""
        if not self.current_execution:
            return

        for step in self.current_execution.steps:
            if step.step_id == step_id:
                step.end_time = datetime.now()
                step.status = 'failed'
                step.error_message = error_message
                if metadata:
                    step.metadata.update(metadata)
                break

    def store_fitted_operation(self, step_id: str, operation: Any):
        """Store a fitted operation for later use"""
        self.fitted_operations[step_id] = operation

    def complete_execution(self, metadata: Optional[Dict] = None):
        """Complete the current execution"""
        if not self.current_execution:
            return

        self.current_execution.end_time = datetime.now()
        self.current_execution.status = 'completed'
        if metadata:
            self.current_execution.metadata.update(metadata)

        # Check if any steps failed
        if any(step.status == 'failed' for step in self.current_execution.steps):
            self.current_execution.status = 'completed_with_errors'

        self.executions.append(self.current_execution)
        self.current_execution = None

    def fail_execution(self, error_message: str, metadata: Optional[Dict] = None):
        """Mark the current execution as failed"""
        if not self.current_execution:
            return

        self.current_execution.end_time = datetime.now()
        self.current_execution.status = 'failed'
        self.current_execution.metadata['error_message'] = error_message
        if metadata:
            self.current_execution.metadata.update(metadata)

        self.executions.append(self.current_execution)
        self.current_execution = None

    def get_execution_summary(self, execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of execution(s)"""
        if execution_id:
            execution = next((e for e in self.executions if e.execution_id == execution_id), None)
            if not execution:
                raise ValueError(f"Execution {execution_id} not found")
            executions = [execution]
        else:
            executions = self.executions

        summary = {
            "total_executions": len(executions),
            "executions": []
        }

        for execution in executions:
            exec_summary = {
                "execution_id": execution.execution_id,
                "status": execution.status,
                "duration": execution.total_duration_seconds,
                "total_steps": len(execution.steps),
                "completed_steps": sum(1 for s in execution.steps if s.status == 'completed'),
                "failed_steps": sum(1 for s in execution.steps if s.status == 'failed'),
                "start_time": execution.start_time.isoformat(),
                "end_time": execution.end_time.isoformat() if execution.end_time else None
            }
            summary["executions"].append(exec_summary)

        return summary

    def save_execution_log(self, filepath: Union[str, Path], execution_id: Optional[str] = None):
        """Save execution log to JSON file"""
        if execution_id:
            execution = next((e for e in self.executions if e.execution_id == execution_id), None)
            if not execution:
                raise ValueError(f"Execution {execution_id} not found")
            data = asdict(execution)
        else:
            data = {
                "executions": [asdict(e) for e in self.executions],
                "fitted_operations_keys": list(self.fitted_operations.keys())
            }

        # Convert datetime objects to ISO strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        data = convert_datetime(data)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def save_fitted_operations(self, filepath: Union[str, Path]):
        """Save fitted operations to pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.fitted_operations, f)

    def create_pipeline_bundle(self, bundle_path: Union[str, Path],
                             include_dataset: bool = False, dataset = None):
        """
        Create a complete pipeline bundle with all artifacts

        Bundle contains:
        - pipeline_config.json: Original pipeline configuration
        - execution_log.json: Detailed execution history
        - fitted_operations.pkl: All fitted operations
        - dataset.pkl: Dataset (if included)
        """
        bundle_path = Path(bundle_path)

        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Save execution log
            self.save_execution_log("temp_execution_log.json")
            zipf.write("temp_execution_log.json", "execution_log.json")
            Path("temp_execution_log.json").unlink()

            # Save fitted operations
            self.save_fitted_operations("temp_fitted_ops.pkl")
            zipf.write("temp_fitted_ops.pkl", "fitted_operations.pkl")
            Path("temp_fitted_ops.pkl").unlink()

            # Save pipeline config
            if self.executions:
                config = self.executions[-1].pipeline_config
                with open("temp_config.json", 'w') as f:
                    json.dump(config, f, indent=2)
                zipf.write("temp_config.json", "pipeline_config.json")
                Path("temp_config.json").unlink()

            # Save dataset if requested
            if include_dataset and dataset:
                with open("temp_dataset.pkl", 'wb') as f:
                    pickle.dump(dataset, f)
                zipf.write("temp_dataset.pkl", "dataset.pkl")
                Path("temp_dataset.pkl").unlink()

    def load_pipeline_bundle(self, bundle_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a complete pipeline bundle"""
        bundle_path = Path(bundle_path)
        result = {}

        with zipfile.ZipFile(bundle_path, 'r') as zipf:
            # Load execution log
            if "execution_log.json" in zipf.namelist():
                with zipf.open("execution_log.json") as f:
                    result["execution_log"] = json.load(f)

            # Load fitted operations
            if "fitted_operations.pkl" in zipf.namelist():
                with zipf.open("fitted_operations.pkl") as f:
                    result["fitted_operations"] = pickle.load(f)
                    self.fitted_operations = result["fitted_operations"]

            # Load pipeline config
            if "pipeline_config.json" in zipf.namelist():
                with zipf.open("pipeline_config.json") as f:
                    result["pipeline_config"] = json.load(f)

            # Load dataset
            if "dataset.pkl" in zipf.namelist():
                with zipf.open("dataset.pkl") as f:
                    result["dataset"] = pickle.load(f)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary format"""
        return {
            'executions': [asdict(execution) for execution in self.executions],
            'fitted_operations_count': len(self.fitted_operations),
            'export_timestamp': datetime.now().isoformat()
        }

    def to_json(self) -> str:
        """Convert history to JSON string"""
        data = self.to_dict()

        # Convert datetime objects to ISO format strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        return json.dumps(convert_datetime(data), indent=2, default=str)

    def save_json(self, filepath: Union[str, Path]):
        """Save history as JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    def save_pickle(self, filepath: Union[str, Path]):
        """Save complete history as pickle file"""
        data = {
            'history': self,
            'fitted_operations': self.fitted_operations,
            'timestamp': datetime.now()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def save_bundle(self, filepath: Union[str, Path]):
        """Save complete pipeline bundle as zip file"""
        bundle_data = {
            'history': self.to_dict(),
            'fitted_operations': self.fitted_operations,
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }
        }

        self._save_zip_bundle(Path(filepath), bundle_data)

    def _save_zip_bundle(self, filepath: Path, pipeline_data: Dict[str, Any]):
        """Save as zip bundle with separate files"""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save execution log as JSON
            log_path = temp_path / "execution_log.json"
            self.save_execution_log(log_path)

            # Save fitted operations as pickle
            if pipeline_data.get('fitted_operations'):
                ops_path = temp_path / "fitted_operations.pkl"
                with open(ops_path, 'wb') as f:
                    pickle.dump(pipeline_data['fitted_operations'], f)

            # Save pipeline config as JSON
            config_path = temp_path / "pipeline_config.json"
            if self.executions:
                config = self.executions[-1].pipeline_config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

            # Save complete pipeline data
            complete_path = temp_path / "pipeline_data.pkl"
            with open(complete_path, 'wb') as f:
                pickle.dump(pipeline_data, f)

            # Create zip bundle
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)

    def _save_pickle_bundle(self, filepath: Path, pipeline_data: Dict[str, Any]):
        """Save as single pickle file"""
        complete_data = {
            **pipeline_data,
            'execution_history': self.to_dict()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(complete_data, f)

    def _save_json_bundle(self, filepath: Path, pipeline_data: Dict[str, Any]):
        """Save as JSON (excluding non-serializable fitted operations)"""
        json_data = {
            'execution_history': self.to_dict(),
            'config': pipeline_data.get('config', {}),
            'metadata': pipeline_data.get('metadata', {}),
            # Note: fitted_operations are not JSON-serializable
            'fitted_operations_info': {
                'count': len(pipeline_data.get('fitted_operations', {})),
                'types': [type(op).__name__ for op in pipeline_data.get('fitted_operations', {}).values()]
            }
        }

        # Convert datetime objects
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        json_data = convert_datetime(json_data)

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
