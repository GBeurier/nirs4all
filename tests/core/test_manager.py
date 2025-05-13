import pytest
import os
import json
import shutil
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from nirs4all.core.manager import ExperimentManager, NumpyEncoder, sanitize_folder_name
from nirs4all.core.config import Config


class TestManagerUtils:
    """Tests pour les fonctions utilitaires du module manager."""
    
    def test_sanitize_folder_name(self):
        """Test de la fonction sanitize_folder_name."""
        assert sanitize_folder_name("test folder") == "test folder"
        assert sanitize_folder_name("test/folder") == "testfolder"
        assert sanitize_folder_name("test$folder!") == "testfolder"
        assert sanitize_folder_name("test_folder-123") == "test_folder-123"
        assert sanitize_folder_name("test  folder") == "test  folder"
        
    def test_numpy_encoder(self):
        """Test de l'encodeur JSON personnalisé pour NumPy."""
        # Créer des données NumPy
        data = {
            "int": np.int32(42),
            "float": np.float32(3.14),
            "array": np.array([1, 2, 3])
        }
        
        # Encoder en JSON
        json_str = json.dumps(data, cls=NumpyEncoder)
        
        # Décoder pour vérifier
        decoded = json.loads(json_str)
        
        assert decoded["int"] == 42
        assert decoded["float"] == pytest.approx(3.14)
        assert decoded["array"] == [1, 2, 3]


class TestExperimentManager:
    """Tests pour la classe ExperimentManager."""
    
    def setup_method(self):
        """Configuration initiale pour les tests."""
        # Créer un répertoire temporaire pour les résultats
        self.results_dir = "test_experiment_manager"
        # Créer un manager
        self.manager = ExperimentManager(self.results_dir, resume_mode="restart", verbose=0)
        
    def teardown_method(self):
        """Nettoyage après les tests."""
        # Close logger handlers to release the log file
        if hasattr(self.manager, 'logger') and self.manager.logger:
            for handler in self.manager.logger.handlers[:]:
                handler.close()
                self.manager.logger.removeHandler(handler)
                
        # Supprimer le répertoire temporaire
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
    
    def test_setup_logger(self):
        """Test de la configuration du logger."""
        logger = self.manager._setup_logger()
        assert logger.name == "ExperimentManager"
        assert logger.level == 10  # DEBUG
        assert len(logger.handlers) > 0
        
    def test_extract_dataset_name(self):
        """Test de l'extraction du nom du dataset."""
        # Cas 1: chaîne de caractères
        assert self.manager._extract_dataset_name("test_dataset") == "test_dataset"
        
        # Cas 2: objet avec attribut name
        mock_dataset = MagicMock()
        mock_dataset.name = "dataset_with_name"
        assert self.manager._extract_dataset_name(mock_dataset) == "dataset_with_name"
        
        # Cas 3: objet sans attribut name
        assert self.manager._extract_dataset_name({}) == "unknown_dataset"
        
    def test_extract_model_name(self):
        """Test de l'extraction du nom du modèle."""
        # Cas 1: dictionnaire avec clé 'class'
        model_config = {"class": "sklearn.linear_model.LinearRegression"}
        assert self.manager._extract_model_name(model_config) == "LinearRegression"
        
        # Cas 2: dictionnaire avec clé 'path'
        model_config = {"path": "models/model.pkl"}
        model_name = self.manager._extract_model_name(model_config)
        assert model_name == "model", f"Expected 'model' but got '{model_name}'"
        
        # Cas 3: fonction callable
        def test_model():
            pass
        assert self.manager._extract_model_name(test_model) == "test_model"
        
        # Cas 4: chaîne de caractères
        assert self.manager._extract_model_name("models/test_model.h5") == "test_model"
        
        # Cas 5: autre type
        assert self.manager._extract_model_name(123) == "unknown_model"
        
    def test_prepare_experiment(self):
        """Test de la préparation d'une expérience."""
        # Créer une configuration
        config = Config(
            dataset="test_dataset",
            model={"class": "sklearn.linear_model.LinearRegression"},
            experiment={"metrics": ["mse", "r2"]}
        )
        
        # Préparer l'expérience
        self.manager.prepare_experiment(config)
        
        # Vérifier que le chemin d'expérience a été créé
        assert self.manager.experiment_path is not None
        assert os.path.exists(self.manager.experiment_path)
        
        # Vérifier que le fichier de configuration a été créé
        config_file = os.path.join(self.manager.experiment_path, "config.json")
        assert os.path.exists(config_file)
        
        # Vérifier que les informations d'expérience ont été mises à jour
        assert self.manager.experiment_info["dataset_name"] == "test_dataset"
        assert "LinearRegression" in self.manager.experiment_info["model_name"]
        
    def test_is_experiment_completed(self):
        """Test de la détection d'une expérience terminée."""
        # Créer un chemin d'expérience temporaire
        experiment_path = os.path.join(self.results_dir, "test_experiment")
        os.makedirs(experiment_path, exist_ok=True)
        
        # Cas 1: expérience incomplète (aucun fichier)
        assert not self.manager.is_experiment_completed(experiment_path)
        
        # Cas 2: expérience incomplète (seulement un fichier)
        os.makedirs(os.path.join(experiment_path, "model"), exist_ok=True)
        assert not self.manager.is_experiment_completed(experiment_path)
        
        # Cas 3: expérience complète (les deux fichiers existent et metrics.json n'est pas vide)
        with open(os.path.join(experiment_path, "metrics.json"), "w") as f:
            json.dump({"weighted": {"mse": 0.1, "r2": 0.9}}, f)
        assert self.manager.is_experiment_completed(experiment_path)
        
    def test_save_results(self):
        """Test de la sauvegarde des résultats."""
        # Configurer le manager pour avoir un experiment_path
        config = Config(
            dataset="test_dataset",
            model={"class": "test_model"},
            experiment={"metrics": ["mse", "r2"]}
        )
        self.manager.prepare_experiment(config)
        
        # Préparer les données de test
        model_manager = MagicMock()
        model_manager.evaluate.return_value = {"mse": 0.1, "r2": 0.9}
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.0])
        metrics = ["mse", "r2"]
        
        # Appeler save_results
        self.manager.save_results(model_manager, y_pred, y_true, metrics)
        
        # Vérifier que les fichiers ont été créés
        assert os.path.exists(os.path.join(self.manager.experiment_path, "metrics.json"))
        assert os.path.exists(os.path.join(self.manager.experiment_path, "predictions.csv"))
        
        # Vérifier le contenu des fichiers
        with open(os.path.join(self.manager.experiment_path, "metrics.json"), "r") as f:
            metrics_json = json.load(f)
        
        # La clé spécifique dans metrics_json peut varier en fonction de l'implémentation
        # Dans l'implémentation actuelle, on peut avoir 'weighted' ou d'autres clés
        # Vérifions que la structure contient au moins une clé avec les métriques attendues
        has_metrics = False
        for key, value in metrics_json.items():
            if isinstance(value, dict) and "mse" in value and "r2" in value:
                has_metrics = True
                assert value["mse"] == 0.1
                assert value["r2"] == 0.9
                break
        assert has_metrics, "No metrics found in the JSON output"
        
        # Vérifier les prédictions CSV
        predictions_df = pd.read_csv(os.path.join(self.manager.experiment_path, "predictions.csv"))
        assert "y_true" in predictions_df.columns
        assert len(predictions_df) == 3
        
        # Vérifie qu'au moins une colonne contient des prédictions
        pred_columns = [col for col in predictions_df.columns if "y_pred" in col]
        assert len(pred_columns) > 0
        
    def test_make_config_serializable(self):
        """Test de la conversion d'une configuration en format sérialisable."""
        # Créer une configuration avec différents types d'objets
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        config = Config(
            dataset="test_dataset",
            x_pipeline=StandardScaler(),
            model=LinearRegression(),
            experiment={"metrics": ["mse", "r2"]}
        )
        
        # Convertir en format sérialisable
        serializable = self.manager.make_config_serializable(config)
        
        # Vérifier que la conversion a été correctement effectuée
        assert isinstance(serializable, dict)
        assert serializable["dataset"] == "test_dataset"
        assert isinstance(serializable["x_pipeline"], dict)
        assert "class" in serializable["x_pipeline"]
        assert "StandardScaler" in serializable["x_pipeline"]["class"]
        assert isinstance(serializable["model"], dict)
        assert "class" in serializable["model"]
        assert "LinearRegression" in serializable["model"]["class"]
        
    def test_update_centralized_results(self):
        """Test de la mise à jour des résultats centralisés."""
        # Configurer le manager pour avoir un experiment_path et experiment_info
        config = Config(
            dataset="test_dataset",
            model={"class": "test_model"},
            experiment={"metrics": ["mse", "r2"]}
        )
        self.manager.prepare_experiment(config)
        
        # Préparer les données de test
        metrics = {"mse": 0.1, "r2": 0.9}
        best_params = {"n_estimators": 100}
        
        # Appeler update_centralized_results
        self.manager.update_centralized_results(metrics, best_params)
        
        # Vérifier que les fichiers experiments.json ont été créés
        dataset_exp_path = os.path.join(self.results_dir, "test_dataset", "experiments.json")
        model_exp_path = os.path.join(self.results_dir, "test_dataset", "test_model", "experiments.json")
        
        assert os.path.exists(dataset_exp_path)
        assert os.path.exists(model_exp_path)
        
        # Vérifier le contenu des fichiers
        with open(dataset_exp_path, "r") as f:
            dataset_exp = json.load(f)
        
        assert len(dataset_exp) == 1
        assert "scores" in dataset_exp[0]
        assert dataset_exp[0]["scores"]["mse"] == 0.1
        assert dataset_exp[0]["scores"]["r2"] == 0.9
        assert "best_params" in dataset_exp[0]
        assert dataset_exp[0]["best_params"]["n_estimators"] == 100
