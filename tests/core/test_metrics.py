import pytest
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from nirs4all.core.metrics import get_metric, metric_mappings
from nirs4all.utils.backend_utils import is_tensorflow_available


class TestMetrics:
    """Tests pour les métriques du module core."""

    def test_get_sklearn_metric(self):
        """Teste la récupération des métriques scikit-learn."""
        # Test avec nom de métrique sklearn
        metric_func = get_metric('r2', framework='sklearn')
        assert metric_func == r2_score
        
        metric_func = get_metric('neg_mean_squared_error', framework='sklearn')
        assert metric_func == mean_squared_error
        
        metric_func = get_metric('neg_mean_absolute_error', framework='sklearn')
        assert metric_func == mean_absolute_error
        
    def test_get_metric_with_abbreviation(self):
        """Teste la récupération des métriques par abréviation."""
        # Test avec abréviation
        metric_func = get_metric('r2', framework='sklearn')
        assert metric_func == r2_score
        
        metric_func = get_metric('mse', framework='sklearn')
        assert metric_func == mean_squared_error
        
        metric_func = get_metric('mae', framework='sklearn')
        assert metric_func == mean_absolute_error
    
    def test_get_metric_invalid(self):
        """Teste le comportement avec une métrique invalide."""
        with pytest.raises(ValueError):
            get_metric('invalid_metric_name', framework='sklearn')
    
    @pytest.mark.tensorflow
    def test_get_tensorflow_metric(self):
        """Teste la récupération des métriques TensorFlow."""
        if not is_tensorflow_available():
            pytest.skip("TensorFlow n'est pas disponible")
        
        import tensorflow as tf
        
        # Test avec nom de métrique TensorFlow
        metric_class = get_metric('R2Score', framework='tensorflow')
        assert metric_class == tf.keras.metrics.R2Score
        
        metric_class = get_metric('MeanSquaredError', framework='tensorflow')
        assert metric_class == tf.keras.metrics.MeanSquaredError
        
        metric_class = get_metric('MeanAbsoluteError', framework='tensorflow')
        assert metric_class == tf.keras.metrics.MeanAbsoluteError
        
    @pytest.mark.tensorflow
    def test_get_metric_with_framework_inference(self):
        """Teste l'inférence automatique du framework."""
        if not is_tensorflow_available():
            pytest.skip("TensorFlow n'est pas disponible")
        
        import tensorflow as tf
        
        # Sans spécifier le framework, devrait retourner la classe TensorFlow s'il est disponible
        metric = get_metric('r2')
        assert metric == tf.keras.metrics.R2Score
        
    def test_get_metric_with_class_instance(self):
        """Teste l'utilisation de get_metric avec une classe directement."""
        # Si on passe une classe directement, elle devrait être renvoyée telle quelle
        class DummyMetric:
            pass
        
        metric = get_metric(DummyMetric)
        assert metric == DummyMetric
        
    def test_metric_mappings_content(self):
        """Vérifie le contenu de la liste des mappings de métriques."""
        # Vérifie que les métriques les plus courantes sont présentes
        metric_names = [m[0] for m in metric_mappings if m[0] is not None]
        assert "r2" in metric_names
        assert "accuracy" in metric_names
        assert "neg_mean_squared_error" in metric_names
        
        # Vérifie les mappings TensorFlow
        tf_names = [m[1] for m in metric_mappings if m[1] is not None]
        assert "R2Score" in tf_names
        assert "MeanSquaredError" in tf_names
        assert "Accuracy" in tf_names
        
    def test_get_metric_with_full_path(self):
        """Teste l'importation de métriques par chemin complet."""
        # Test avec un chemin d'importation complet
        metric = get_metric("sklearn.metrics.accuracy_score")
        from sklearn.metrics import accuracy_score
        assert metric == accuracy_score
        
        # Test avec un chemin invalide
        with pytest.raises((ValueError, ImportError, AttributeError)):
            get_metric("invalid.module.path")
            
    @pytest.mark.tensorflow
    def test_tensorflow_metric_usage(self):
        """Teste l'utilisation concrète d'une métrique TensorFlow."""
        if not is_tensorflow_available():
            pytest.skip("TensorFlow n'est pas disponible")
            
        import tensorflow as tf
        import numpy as np
        
        # Création de données
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.0, 4.1])
        
        # Récupération de la métrique
        metric_class = get_metric('MeanSquaredError', framework='tensorflow')
        
        # Utilisation de la métrique
        metric = metric_class()
        metric.update_state(y_true, y_pred)
        result = metric.result().numpy()
        
        # Calcul manuel pour comparaison
        expected = np.mean((y_true - y_pred) ** 2)
        assert np.isclose(result, expected)
