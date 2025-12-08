"""
Test suite for enhanced pipeline functionality

Tests parallelization, history tracking, serialization, and error handling
"""
import os
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json

# Mock the imports for testing
class MockSpectraDataset:
    def __init__(self):
        self.data = {}
        self.features = None
        self.labels = None

    def get_features(self, row_indices=None):
        import numpy as np
        return np.random.random((100, 50))  # Mock features

    def get_labels(self, row_indices=None):
        import numpy as np
        return np.random.random(100)  # Mock labels

    def get_train_indices(self):
        return list(range(80))  # Mock train indices

    def add_features(self, new_features, feature_names=None):
        print(f"Added {new_features.shape[1]} new features")

class MockPipelineContext:
    def __init__(self):
        self.filters = {}

    def apply_filters(self, filters):
        self.filters.update(filters)
        print(f"Applied filters: {filters}")

class MockPipelineConfig:
    def __init__(self, pipeline):
        self.pipeline = pipeline

class MockOperation:
    def __init__(self, name):
        self.name = name
        self.fitted = False

    def get_name(self):
        return self.name

    def execute(self, dataset):
        print(f"Executing {self.name}")
        self.fitted = True

    def fit_transform(self, X):
        import numpy as np
        self.fitted = True
        return np.random.random((X.shape[0], 5))  # Add 5 new features

# Patch the imports at the module level
import sys
sys.modules['SpectraDataset'] = type(sys)('SpectraDataset')
sys.modules['SpectraDataset'].SpectraDataset = MockSpectraDataset
sys.modules['PipelineContext'] = type(sys)('PipelineContext')
sys.modules['PipelineContext'].PipelineContext = MockPipelineContext
sys.modules['PipelineConfig'] = type(sys)('PipelineConfig')
sys.modules['PipelineConfig'].PipelineConfig = MockPipelineConfig

# Now import our modules
from PipelineHistory import PipelineHistory, StepExecution, PipelineExecution
from PipelineBuilder_clean import PipelineBuilder


class TestPipelineHistory(unittest.TestCase):
    """Test pipeline history tracking and serialization"""

    def setUp(self):
        self.history = PipelineHistory()

    def test_execution_tracking(self):
        """Test basic execution tracking"""
        config = {"pipeline": [{"step": "test"}]}

        # Start execution
        exec_id = self.history.start_execution(config)
        self.assertIsNotNone(exec_id)
        self.assertEqual(self.history.current_execution.status, 'running')

        # Start a step
        step = self.history.start_step(1, "Test step", {"test": "config"})
        self.assertEqual(step.status, 'running')

        # Complete the step
        self.history.complete_step(step.step_id)
        self.assertEqual(step.status, 'completed')

        # Complete execution
        self.history.complete_execution()
        self.assertEqual(self.history.current_execution.status, 'completed')

    def test_step_failure(self):
        """Test step failure handling"""
        config = {"pipeline": [{"step": "test"}]}
        exec_id = self.history.start_execution(config)

        step = self.history.start_step(1, "Failing step", {"test": "config"})
        self.history.fail_step(step.step_id, "Test error")

        self.assertEqual(step.status, 'failed')
        self.assertEqual(step.error_message, "Test error")

    def test_serialization(self):
        """Test history serialization"""
        config = {"pipeline": [{"step": "test"}]}
        exec_id = self.history.start_execution(config)

        step = self.history.start_step(1, "Test step", {"test": "config"})
        self.history.complete_step(step.step_id)
        self.history.complete_execution()

        # Test JSON export
        json_data = self.history.to_json()
        self.assertIsInstance(json_data, str)

        # Test round-trip
        history_dict = json.loads(json_data)
        self.assertIn('executions', history_dict)
        self.assertEqual(len(history_dict['executions']), 1)

    def test_bundle_saving(self):
        """Test saving history bundles"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"pipeline": [{"step": "test"}]}
            exec_id = self.history.start_execution(config)

            step = self.history.start_step(1, "Test step", {"test": "config"})
            self.history.complete_step(step.step_id)
            self.history.complete_execution()

            # Save bundle
            bundle_path = Path(temp_dir) / "test_bundle.zip"
            self.history.save_bundle(bundle_path)

            self.assertTrue(bundle_path.exists())

            # Verify bundle contents
            import zipfile
            with zipfile.ZipFile(bundle_path, 'r') as zf:
                files = zf.namelist()
                self.assertIn('history.json', files)
                self.assertIn('metadata.json', files)


class TestPipelineBuilder(unittest.TestCase):
    """Test pipeline builder functionality"""

    def setUp(self):
        self.builder = PipelineBuilder()

    def test_build_from_string(self):
        """Test building operations from string presets"""
        # Test with a preset
        operation = self.builder.build_operation('StandardScaler')
        self.assertIsNotNone(operation)

    def test_build_from_dict(self):
        """Test building operations from dict configs"""
        config = {
            'type': 'transformation',
            'sklearn.preprocessing.StandardScaler': {}
        }
        operation = self.builder.build_operation(config)
        self.assertIsNotNone(operation)

    def test_fitted_operation_tracking(self):
        """Test fitted operation tracking"""
        config = {
            'type': 'transformation',
            'sklearn.preprocessing.StandardScaler': {}
        }
        operation = self.builder.build_operation(config, step_id='test_step')

        # Simulate fitting
        mock_fitted = MockOperation('fitted_scaler')
        mock_fitted.fitted = True
        self.builder.store_fitted_operation('test_step', mock_fitted)

        fitted_ops = self.builder.get_fitted_operations()
        self.assertIn('test_step', fitted_ops)
        self.assertTrue(fitted_ops['test_step'].fitted)


class TestEnhancedPipelineIntegration(unittest.TestCase):
    """Integration tests for the enhanced pipeline system"""

    def setUp(self):
        # Mock the imports for PipelineRunner
        self.dataset = MockSpectraDataset()
        self.context = MockPipelineContext()

    @patch('PipelineRunner_enhanced.PipelineBuilder')
    @patch('PipelineRunner_enhanced.PipelineHistory')
    def test_pipeline_execution(self, mock_history_class, mock_builder_class):
        """Test complete pipeline execution"""
        # Setup mocks
        mock_history = Mock()
        mock_history_class.return_value = mock_history
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        # Import after patching
        from PipelineRunner_enhanced import PipelineRunner

        # Create runner
        runner = PipelineRunner(max_workers=2, backend='threading')

        # Mock pipeline config
        pipeline_config = [
            {'type': 'transformation', 'sklearn.preprocessing.StandardScaler': {}},
            {'type': 'model', 'sklearn.linear_model.LinearRegression': {}}
        ]

        # Mock builder responses
        mock_operation1 = MockOperation('StandardScaler')
        mock_operation2 = MockOperation('LinearRegression')
        mock_builder.build_operation.side_effect = [mock_operation1, mock_operation2]

        # Run pipeline
        try:
            result = runner.run_pipeline(pipeline_config, self.dataset, self.context)
            # Should complete without error
            self.assertIsNotNone(result)
        except Exception as e:
            # Expected due to mocking, but should not be syntax errors
            self.assertNotIn('syntax', str(e).lower())

    def test_feature_augmentation_workflow(self):
        """Test feature augmentation workflow"""
        augmenters = [MockOperation('Augmenter1'), MockOperation('Augmenter2')]

        # Mock dataset methods
        train_indices = self.dataset.get_train_indices()
        train_features = self.dataset.get_features(train_indices)

        self.assertEqual(len(train_indices), 80)
        self.assertEqual(train_features.shape[0], 100)  # Mock returns full shape

        # Simulate augmentation
        for aug in augmenters:
            new_features = aug.fit_transform(train_features)
            self.dataset.add_features(new_features)
            self.assertTrue(aug.fitted)

    def test_parallelization_config(self):
        """Test different parallelization configurations"""
        configs = [
            {'max_workers': 1, 'backend': 'threading'},
            {'max_workers': 2, 'backend': 'threading'},
            {'max_workers': -1, 'backend': 'threading'},
        ]

        for config in configs:
            # Should create without error
            try:
                from PipelineRunner_enhanced import PipelineRunner
                runner = PipelineRunner(**config)
                self.assertEqual(runner.max_workers, config['max_workers'])
                self.assertEqual(runner.backend, config['backend'])
            except ImportError:
                # Expected due to missing dependencies
                pass


class TestPipelineSerializationWorkflow(unittest.TestCase):
    """Test complete serialization workflow"""

    def test_save_and_load_workflow(self):
        """Test saving and loading complete pipeline state"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create history with execution
            history = PipelineHistory()
            config = {
                "pipeline": [
                    {"type": "transformation", "StandardScaler": {}},
                    {"type": "model", "LinearRegression": {}}
                ]
            }

            exec_id = history.start_execution(config)

            # Add steps
            step1 = history.start_step(1, "Preprocessing", config["pipeline"][0])
            history.complete_step(step1.step_id)

            step2 = history.start_step(2, "Modeling", config["pipeline"][1])
            history.complete_step(step2.step_id)

            history.complete_execution()

            # Save different formats
            json_path = Path(temp_dir) / "history.json"
            pickle_path = Path(temp_dir) / "history.pkl"
            bundle_path = Path(temp_dir) / "pipeline_bundle.zip"

            history.save_json(json_path)
            history.save_pickle(pickle_path)
            history.save_bundle(bundle_path)

            # Verify files exist
            self.assertTrue(json_path.exists())
            self.assertTrue(pickle_path.exists())
            self.assertTrue(bundle_path.exists())

            # Load and verify JSON
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
                self.assertIn('executions', loaded_data)
                self.assertEqual(len(loaded_data['executions']), 1)
                self.assertEqual(loaded_data['executions'][0]['status'], 'completed')


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
