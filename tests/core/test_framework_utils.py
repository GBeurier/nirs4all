import pytest
import numpy as np
from nirs4all.core.utils import framework, TF_AVAILABLE, TORCH_AVAILABLE


class TestFrameworkUtils:
    """Tests pour les utilitaires de gestion des frameworks ML."""

    def test_framework_decorator(self):
        """Teste le décorateur framework pour marquer des fonctions."""
        # Décore une fonction avec le framework "sklearn"
        @framework("sklearn")
        def create_sklearn_model():
            return "sklearn_model"
        
        # Vérifie que l'attribut framework est correctement ajouté
        assert hasattr(create_sklearn_model, "framework")
        assert create_sklearn_model.framework == "sklearn"
        
        # Vérifie que la fonction fonctionne normalement
        result = create_sklearn_model()
        assert result == "sklearn_model"
    
    @pytest.mark.tensorflow
    def test_tensorflow_decorator(self):
        """Teste le décorateur framework avec TensorFlow."""
        if not TF_AVAILABLE:
            pytest.skip("TensorFlow n'est pas disponible")
        
        @framework("tensorflow")
        def create_tf_model(input_shape=(10,)):
            import tensorflow as tf
            return tf.keras.Sequential([
                tf.keras.layers.Dense(5, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dense(1)
            ])
        
        assert create_tf_model.framework == "tensorflow"
        
        # Vérifie que la fonction renvoie bien un modèle TensorFlow
        model = create_tf_model()
        import tensorflow as tf
        assert isinstance(model, tf.keras.Model)
    
    @pytest.mark.torch
    def test_torch_decorator(self):
        """Teste le décorateur framework avec PyTorch."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch n'est pas disponible")
        
        @framework("pytorch")
        def create_torch_model(input_size=10):
            import torch.nn as nn
            return nn.Sequential(
                nn.Linear(input_size, 5),
                nn.ReLU(),
                nn.Linear(5, 1)
            )
        
        assert create_torch_model.framework == "pytorch"
        
        # Vérifie que la fonction renvoie bien un modèle PyTorch
        model = create_torch_model()
        import torch.nn as nn
        assert isinstance(model, nn.Module)
    
    def test_multiple_decorated_functions(self):
        """Teste que plusieurs fonctions peuvent être décorées indépendamment."""
        @framework("sklearn")
        def func_a():
            return "A"
        
        @framework("tensorflow")
        def func_b():
            return "B"
        
        @framework("pytorch")
        def func_c():
            return "C"
        
        assert func_a.framework == "sklearn"
        assert func_b.framework == "tensorflow"
        assert func_c.framework == "pytorch"