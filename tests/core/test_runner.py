import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch
from nirs4all.core.runner import ExperimentRunner
from nirs4all.core.config import Config


class TestExperimentRunner:
    """Tests pour le ExperimentRunner."""
    
    def setup_method(self):
        """Configuration initiale pour les tests."""
        # Créer un répertoire temporaire pour les résultats
        self.results_dir = "test_results_dir"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Créer une configuration de base pour les tests
        self.config = Config(
            dataset="mock_dataset",
            experiment={
                "metrics": ["mse", "r2"],
                "action": "train",
                "training_params": {"loss": "mse", "epochs": 1}
            }
        )
        
    def teardown_method(self):
        """Nettoyage après les tests."""
        # Supprimer le répertoire temporaire si nécessaire
        import shutil
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        
    @patch("nirs4all.core.runner.get_dataset")
    @patch("nirs4all.core.runner.run_pipeline")
    @patch("nirs4all.core.runner.ModelManagerFactory")
    @patch("nirs4all.core.runner.ExperimentManager")
    def test_train(self, mock_experiment_manager, mock_model_manager_factory, 
                 mock_run_pipeline, mock_get_dataset):
        """Test de la méthode _train."""
        # Configurer les mocks
        mock_model_manager = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.y_test_init = np.array([1, 2, 3])
        
        # Initialiser ExperimentRunner
        mock_manager = MagicMock()
        mock_experiment_manager.return_value = mock_manager
        runner = ExperimentRunner([self.config], self.results_dir)
        
        # Remplacer _evaluate_and_save_results par un mock
        runner._evaluate_and_save_results = MagicMock()
        
        # Appeler _train
        metrics = ["mse", "r2"]
        training_params = {"loss": "mse", "epochs": 5}
        runner._train(mock_model_manager, mock_dataset, training_params, metrics, "regression")
        
        # Vérifier que les méthodes du model_manager sont appelées correctement
        mock_model_manager.train.assert_called_once_with(
            mock_dataset, training_params=training_params, metrics=metrics
        )
        mock_model_manager.save_model.assert_called_once()
        runner._evaluate_and_save_results.assert_called_once_with(
            mock_model_manager, mock_dataset, metrics, task="regression"
        )
    
    @patch("nirs4all.core.runner.get_dataset")
    @patch("nirs4all.core.runner.run_pipeline")
    @patch("nirs4all.core.runner.ModelManagerFactory")
    @patch("nirs4all.core.runner.ExperimentManager")
    @patch("nirs4all.core.runner.FineTunerFactory")
    def test_fine_tune(self, mock_finetuner_factory, mock_experiment_manager, 
                      mock_model_manager_factory, mock_run_pipeline, mock_get_dataset):
        """Test de la méthode _fine_tune."""
        # Configurer les mocks
        mock_model_manager = MagicMock()
        mock_dataset = MagicMock()
        mock_finetuner = MagicMock()
        mock_finetuner_factory.get_fine_tuner.return_value = mock_finetuner
        
        # Le finetuner retourne les "meilleurs" paramètres
        best_params = {"n_estimators": 100, "max_depth": 10}
        mock_finetuner.finetune.return_value = best_params
        
        # Initialiser ExperimentRunner
        mock_manager = MagicMock()
        mock_experiment_manager.return_value = mock_manager
        runner = ExperimentRunner([self.config], self.results_dir)
        
        # Remplacer _evaluate_and_save_results par un mock
        runner._evaluate_and_save_results = MagicMock()
        
        # Appeler _fine_tune
        metrics = ["mse", "r2"]
        finetune_params = {"tuner": "optuna", "n_trials": 10}
        training_params = {"loss": "mse", "epochs": 5}
        runner._fine_tune(mock_model_manager, mock_dataset, finetune_params, training_params, metrics, "regression")
        
        # Vérifier que le finetuner est créé et utilisé correctement
        mock_finetuner_factory.get_fine_tuner.assert_called_once_with("optuna", mock_model_manager)
        mock_finetuner.finetune.assert_called_once_with(
            mock_dataset, finetune_params, training_params, metrics=metrics, task="regression"
        )
        mock_model_manager.save_model.assert_called_once()
        runner._evaluate_and_save_results.assert_called_once_with(
            mock_model_manager, mock_dataset, metrics, best_params, task="regression"
        )
        
    @patch("nirs4all.core.runner.get_dataset")
    @patch("nirs4all.core.runner.run_pipeline")
    @patch("nirs4all.core.runner.ModelManagerFactory")
    @patch("nirs4all.core.runner.ExperimentManager")
    def test_predict(self, mock_experiment_manager, mock_model_manager_factory, 
                   mock_run_pipeline, mock_get_dataset):
        """Test de la méthode _predict."""
        # Configurer les mocks
        mock_model_manager = MagicMock()
        mock_dataset = MagicMock()
        
        # Initialiser ExperimentRunner
        mock_manager = MagicMock()
        mock_experiment_manager.return_value = mock_manager
        runner = ExperimentRunner([self.config], self.results_dir)
        
        # Remplacer _evaluate_and_save_results par un mock
        runner._evaluate_and_save_results = MagicMock()
        
        # Appeler _predict
        metrics = ["mse", "r2"]
        runner._predict(mock_model_manager, mock_dataset, metrics, "regression")
        
        # Vérifier que _evaluate_and_save_results est appelée correctement
        runner._evaluate_and_save_results.assert_called_once_with(
            mock_model_manager, mock_dataset, metrics, task="regression"
        )
    
    @patch("nirs4all.core.runner.get_dataset")
    @patch("nirs4all.core.runner.run_pipeline")
    @patch("nirs4all.core.runner.ModelManagerFactory")
    @patch("nirs4all.core.runner.ExperimentManager")
    def test_evaluate_and_save_results_single_fold(self, mock_experiment_manager, 
                                                 mock_model_manager_factory,
                                                 mock_run_pipeline, mock_get_dataset):
        """Test de la méthode _evaluate_and_save_results avec un seul pli."""
        # Configurer les mocks
        mock_model_manager = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.y_test_init = np.array([1, 2, 3])
        mock_dataset.inverse_transform = lambda x: x  # Identité pour simplifier
        
        # Le modèle prédit un seul ensemble de valeurs (pas de plis multiples)
        y_pred = np.array([1.1, 2.1, 2.9])
        mock_model_manager.predict.return_value = y_pred
        
        # L'évaluation retourne un dict de scores
        scores = {"mse": 0.01, "r2": 0.98}
        mock_model_manager.evaluate.return_value = scores
        
        # Initialiser ExperimentRunner
        mock_manager = MagicMock()
        mock_experiment_manager.return_value = mock_manager
        runner = ExperimentRunner([self.config], self.results_dir)
        
        # Appeler _evaluate_and_save_results
        metrics = ["mse", "r2"]
        runner._evaluate_and_save_results(mock_model_manager, mock_dataset, metrics, task="regression")
        
        # Vérifier que predict et evaluate sont appelés correctement
        mock_model_manager.predict.assert_called_once_with(
            mock_dataset, "regression", return_all=True, raw_class_output=False
        )
        mock_model_manager.evaluate.assert_called_once()
        mock_manager.save_results.assert_called_once_with(
            mock_model_manager, y_pred, mock_dataset.y_test_init, metrics, None, [scores]
        )
    
    @patch("nirs4all.core.runner.get_dataset")
    @patch("nirs4all.core.runner.run_pipeline")
    @patch("nirs4all.core.runner.ModelManagerFactory")
    @patch("nirs4all.core.runner.ExperimentManager")
    def test_run_with_train_action(self, mock_experiment_manager, mock_model_manager_factory,
                                  mock_run_pipeline, mock_get_dataset):
        """Test de la méthode run avec une action d'entraînement."""
        # Configurer les mocks
        mock_dataset = MagicMock()
        mock_dataset.num_classes = 2
        mock_dataset.y_train_init = np.array([0, 1, 0, 1])
        mock_dataset.y_test_init = np.array([0, 1])
        mock_get_dataset.return_value = mock_dataset
        mock_run_pipeline.return_value = mock_dataset
        
        mock_model_manager = MagicMock()
        mock_model_manager_factory.get_model_manager.return_value = mock_model_manager
        
        # Initialiser ExperimentRunner
        mock_manager = MagicMock()
        mock_experiment_manager.return_value = mock_manager
        
        # Créer un config avec action=train
        config = Config(
            dataset="mock_dataset",
            model="mock_model",
            experiment={
                "action": "train",
                "metrics": ["accuracy"],
                "task": "classification",
                "training_params": {"loss": "binary_crossentropy", "epochs": 1}
            }
        )
        
        # Patch les méthodes d'ExperimentRunner
        with patch.object(ExperimentRunner, "_train") as mock_train:
            mock_train.return_value = (None, ["accuracy"], None)  # Updated return value
                            
            # Exécuter run
            runner = ExperimentRunner([config], self.results_dir)
            runner.run()
            
            # Vérifier que les méthodes appropriées sont appelées
            mock_get_dataset.assert_called_once()
            mock_run_pipeline.assert_called_once()
            mock_model_manager_factory.get_model_manager.assert_called_once()
            mock_train.assert_called_once()
    
    @patch("nirs4all.core.runner.get_dataset")
    @patch("nirs4all.core.runner.run_pipeline")
    @patch("nirs4all.core.runner.ModelManagerFactory")
    @patch("nirs4all.core.runner.ExperimentManager")
    def test_train_then_predict(
        self, mock_experiment_manager, mock_model_manager_factory, 
        mock_run_pipeline, mock_get_dataset
    ):
        """Test end-to-end: train a model, save it, then predict using the saved model."""
        # Configurer les mocks
        mock_model_manager = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.y_test_init = np.array([1, 2, 3])
        mock_dataset.inverse_transform = lambda x: x
        mock_model_manager.predict.return_value = np.array([1.1, 2.1, 2.9])
        mock_model_manager.evaluate.return_value = {"mse": 0.01, "r2": 0.98}

        # Initialiser ExperimentRunner
        mock_manager = MagicMock()
        mock_experiment_manager.return_value = mock_manager
        runner = ExperimentRunner([self.config], self.results_dir)

        # TRAIN
        metrics = ["mse", "r2"]
        training_params = {"loss": "mse", "epochs": 5}
        runner._evaluate_and_save_results = MagicMock(return_value=(mock_model_manager.predict.return_value, [mock_model_manager.evaluate.return_value], None))
        runner._train(mock_model_manager, mock_dataset, training_params, metrics, "regression")
        mock_model_manager.train.assert_called_once_with(
            mock_dataset, training_params=training_params, metrics=metrics
        )
        mock_model_manager.save_model.assert_called_once()
        runner._evaluate_and_save_results.assert_called_with(
            mock_model_manager, mock_dataset, metrics, task="regression"
        )

        # RESET for predict
        runner._evaluate_and_save_results.reset_mock()
        mock_model_manager.train.reset_mock()
        mock_model_manager.save_model.reset_mock()

        # PREDICT
        runner._predict(mock_model_manager, mock_dataset, metrics, "regression")
        runner._evaluate_and_save_results.assert_called_once_with(
            mock_model_manager, mock_dataset, metrics, task="regression"
        )
