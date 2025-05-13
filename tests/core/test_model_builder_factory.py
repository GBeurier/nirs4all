import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from nirs4all.core.model.model_builder_factory import ModelBuilderFactory
from nirs4all.utils.backend_utils import is_tensorflow_available, is_torch_available
from unittest.mock import MagicMock, patch
from nirs4all.core.utils import framework

class DummyDataset:
    """Classe simulant un dataset pour les tests."""
    
    def __init__(self):
        self.x_train = np.random.rand(100, 5)
        self.y_train = np.random.rand(100, 1)
        self.folds = None
        self.num_classes = 1
        
    def x_train_(self, format_str=None):
        """Simule la méthode x_train_ du dataset."""
        return self.x_train


class TestModelBuilderFactory:
    """Tests pour la fabrique de constructeurs de modèle."""
    
    def setup_method(self):
        """Configuration initiale pour chaque test."""
        self.dataset = DummyDataset()
        
    def test_build_from_instance(self):
        """Teste la construction d'un modèle à partir d'une instance."""
        model_instance = LinearRegression()
        built_model = ModelBuilderFactory.build_single_model(model_instance, self.dataset, "regression")
        
        assert isinstance(built_model, LinearRegression)
        assert built_model is model_instance  # Vérifie que c'est bien la même instance

    @patch.object(ModelBuilderFactory, 'import_class')
    def test_build_from_string_class_path(self, mock_import_class):
        """Teste la construction d'un modèle à partir d'un chemin de classe."""
        # Configurer le mock pour retourner la classe LinearRegression
        mock_import_class.return_value = LinearRegression
        
        model_config = "sklearn.linear_model.LinearRegression"
        built_model = ModelBuilderFactory.build_single_model(model_config, self.dataset, "regression")
        
        # Vérifier que import_class a été appelé avec le bon chemin
        mock_import_class.assert_called_once_with(model_config)
        
        # Vérifier que le modèle est bien du type attendu
        assert built_model is not None, "The model was not built"
        assert isinstance(built_model, LinearRegression)
        
    def test_build_from_dict(self):
        """Teste la construction d'un modèle à partir d'un dictionnaire."""
        model_config = {
            "class": "sklearn.ensemble.RandomForestRegressor",
            "params": {
                "n_estimators": 10,
                "max_depth": 3
            }
        }
        # Patcher la méthode import_class pour qu'elle retourne la classe RandomForestRegressor
        with patch.object(ModelBuilderFactory, 'import_class', return_value=RandomForestRegressor):
            built_model = ModelBuilderFactory.build_single_model(model_config, self.dataset, "regression")
        
            assert isinstance(built_model, RandomForestRegressor)
            assert built_model.n_estimators == 10
            assert built_model.max_depth == 3

    def test_detect_framework(self):
        """Teste la détection du framework à partir d'un modèle."""
        model = LinearRegression()
        framework = ModelBuilderFactory.detect_framework(model)
        assert framework == "sklearn"
        
    def test_multiple_models_with_folds(self):
        """Teste la création de plusieurs modèles avec des plis."""
        # Configurer le dataset avec des plis
        self.dataset.folds = [0, 1, 2]  # Simuler 3 plis
        
        # Utiliser patch pour éviter les problèmes avec build_single_model
        with patch.object(ModelBuilderFactory, 'build_single_model', return_value=LinearRegression()):
            model_config = LinearRegression()
            models, framework = ModelBuilderFactory.build_models(model_config, self.dataset, "regression")
            
            assert len(models) == 3  # Devrait créer un modèle pour chaque pli
            assert framework == "sklearn"
            assert all(isinstance(m, LinearRegression) for m in models)  # Tous les modèles devraient être du même type
    
    @pytest.mark.tensorflow
    def test_tensorflow_model_if_available(self):
        """Teste la création d'un modèle TensorFlow si disponible."""
        if not is_tensorflow_available():
            pytest.skip("TensorFlow n'est pas disponible pour ce test")
            
        import tensorflow as tf
        
        @framework('tensorflow')
        def create_simple_tf_model(input_shape=(5,)):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dense(1)
            ])
            return model
        
        # Ajouter un attribut framework à la fonction pour que ModelBuilderFactory puisse le détecter
        create_simple_tf_model.framework = "tensorflow"
        
        model_config = create_simple_tf_model
        built_model = ModelBuilderFactory.build_single_model(model_config, self.dataset, "regression")
        
        assert isinstance(built_model, tf.keras.Model)
        
    def test_force_params(self):
        """Teste l'application de paramètres forcés lors de la construction."""
        model_config = {
            "class": "sklearn.ensemble.RandomForestRegressor",
            "params": {
                "n_estimators": 10,
                "max_depth": 3
            }
        }
        
        # Forcer certains paramètres
        force_params = {
            "n_estimators": 20,
            "random_state": 42
        }
        
        # Patcher la méthode import_class pour qu'elle retourne la classe RandomForestRegressor
        with patch.object(ModelBuilderFactory, 'import_class', return_value=RandomForestRegressor):
            built_model = ModelBuilderFactory.build_single_model(model_config, self.dataset, "regression", 
                                                                 force_params=force_params)
            
            assert isinstance(built_model, RandomForestRegressor)
            assert built_model.n_estimators == 20  # Devrait utiliser la valeur forcée
            assert built_model.max_depth == 3  # Devrait conserver la valeur originale
            assert built_model.random_state == 42  # Devrait ajouter le nouveau paramètre
