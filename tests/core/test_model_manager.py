import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from nirs4all.core.model.model_manager import (
    BaseModelManager,
    SklearnModelManager,
    ModelManagerFactory,
    detect_task_type,
    prepare_y,
    METRIC_ABBREVIATIONS
)
from nirs4all.utils.backend_utils import is_tensorflow_available


class TestModelManagerUtils:
    """Tests pour les fonctions utilitaires de model_manager.py"""

    def test_detect_task_type(self):
        """Test de la détection du type de tâche à partir de loss et metrics."""
        # Classification
        assert detect_task_type('binary_crossentropy', []) == 'classification'
        assert detect_task_type('categorical_crossentropy', []) == 'classification'
        assert detect_task_type('mse', ['accuracy']) == 'classification'
        
        # Regression (par défaut)
        assert detect_task_type('mse', ['mse', 'mae']) == 'regression'
        assert detect_task_type('mean_squared_error', []) == 'regression'

    def test_metric_abbreviations(self):
        """Test que les abréviations de métriques sont correctement mappées."""
        assert 'r2' in METRIC_ABBREVIATIONS
        assert METRIC_ABBREVIATIONS['r2'] == 'r2'
        
        assert 'mse' in METRIC_ABBREVIATIONS
        assert METRIC_ABBREVIATIONS['mse'] == 'neg_mean_squared_error'
        
        assert 'acc' in METRIC_ABBREVIATIONS
        assert METRIC_ABBREVIATIONS['acc'] == 'accuracy'

    @pytest.mark.skipif(not is_tensorflow_available(), reason="TensorFlow n'est pas disponible")
    def test_prepare_y_classification_tensorflow(self):
        """Test de la préparation des étiquettes pour la classification avec TensorFlow."""
        import tensorflow
        
        # Préparer des données simples
        y_train = np.array([0, 1, 2, 0, 1])
        y_val = np.array([2, 0, 1, 2, 0])
        
        # Créer un modèle TF de base
        model = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tensorflow.keras.layers.Dense(3, activation='softmax')  # 3 classes
        ])
        
        # Appliquer la fonction prepare_y avec sparse_categorical_crossentropy
        y_train_prep, y_val_prep, model_prep, num_classes = prepare_y(
            y_train, y_val, model, 'tensorflow', 'sparse_categorical_crossentropy', 'classification'
        )
        
        # Vérifier les résultats
        assert num_classes == 3
        assert np.array_equal(y_train_prep, y_train)  # Pas de changement pour sparse_categorical
        assert np.array_equal(y_val_prep, y_val)
        
        # Appliquer la fonction prepare_y avec categorical_crossentropy
        y_train_prep, y_val_prep, model_prep, num_classes = prepare_y(
            y_train, y_val, model, 'tensorflow', 'categorical_crossentropy', 'classification'
        )
        
        # Vérifier les résultats pour one-hot encoding
        assert num_classes == 3
        assert y_train_prep.shape == (5, 3)  # One-hot encodé
        assert y_val_prep.shape == (5, 3)
        # Vérifier que la somme de chaque ligne de one-hot est 1
        assert np.all(np.sum(y_train_prep, axis=1) == 1)
        assert np.all(np.sum(y_val_prep, axis=1) == 1)

    def test_prepare_y_regression_sklearn(self):
        """Test de la préparation des étiquettes pour la régression avec scikit-learn."""
        y_train = np.array([0.5, 1.2, 2.3, 0.1, 1.7])
        y_val = np.array([2.1, 0.8, 1.5, 2.9, 0.3])
        
        # Mock d'un modèle scikit-learn
        model = MagicMock()
        
        # Appliquer la fonction prepare_y
        y_train_prep, y_val_prep, model_prep, num_classes = prepare_y(
            y_train, y_val, model, 'sklearn', 'mse', 'regression'
        )
        
        # Vérifier les résultats
        assert num_classes is None  # Pas applicable en régression
        assert np.array_equal(y_train_prep, y_train)
        assert np.array_equal(y_val_prep, y_val)
        
        # Le modèle ne devrait pas être modifié pour sklearn
        assert model_prep is model


class DummyDataset:
    """Classe simulant un dataset pour les tests."""
    
    def __init__(self, n_samples=100, n_features=5, n_folds=1):
        self.x_train = np.random.rand(n_samples, n_features)
        self.y_train = np.random.rand(n_samples, 1)
        self.x_test = np.random.rand(n_samples // 2, n_features)
        self.y_test = np.random.rand(n_samples // 2, 1)
        self.n_folds = n_folds
        self.folds = list(range(n_folds)) if n_folds > 1 else None
        
    def fold_data(self, format_type='concat', no_folds=False):
        """Simuler la méthode fold_data du dataset."""
        if no_folds or self.n_folds == 1:
            yield (self.x_train, self.y_train, self.x_test, self.y_test)
        else:
            # Diviser les données en n_folds
            fold_size = len(self.x_train) // self.n_folds
            for i in range(self.n_folds):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < self.n_folds - 1 else len(self.x_train)
                
                # Créer un pli en utilisant une partie des données comme validation
                x_train_fold = np.concatenate([self.x_train[:start_idx], self.x_train[end_idx:]])
                y_train_fold = np.concatenate([self.y_train[:start_idx], self.y_train[end_idx:]])
                x_val_fold = self.x_train[start_idx:end_idx]
                y_val_fold = self.y_train[start_idx:end_idx]
                
                yield (x_train_fold, y_train_fold, x_val_fold, y_val_fold)
                
    def x_test_(self, format_type='concat'):
        """Simuler la méthode x_test_ du dataset."""
        return self.x_test


class TestSklearnModelManager:
    """Tests pour le SklearnModelManager."""
    
    def setup_method(self):
        """Initialisation avant chaque test."""
        self.dataset = DummyDataset()
        self.model = LinearRegression()
        self.models = [self.model]
        self.model_config = {'type': 'LinearRegression'}
        self.manager = SklearnModelManager(self.models, self.model_config)
        
    def test_initialization(self):
        """Test de l'initialisation du manager."""
        assert self.manager.models == self.models
        assert self.manager.model_config == self.model_config
        assert self.manager.framework == 'sklearn'
        
    def test_train(self):
        """Test de l'entraînement d'un modèle scikit-learn."""
        # Définir les métriques explicitement
        metrics = ['mse', 'r2']
        
        # Remplacer fit par un mock pour vérifier qu'il est appelé
        self.model.fit = MagicMock()
        
        # Patcher la méthode fold_data pour qu'elle retourne une seule fois les données
        mock_dataset = MagicMock()
        mock_dataset.fold_data.return_value = [(self.dataset.x_train, self.dataset.y_train, 
                                              self.dataset.x_test, self.dataset.y_test)]
        
        # Appeler train
        self.manager.train(mock_dataset, training_params={'epochs': 1}, metrics=metrics)
        
        # Vérifier que fit a été appelé une fois
        self.model.fit.assert_called_once()
        
    def test_predict(self):
        """Test de la prédiction avec un modèle scikit-learn."""
        # Remplacer predict par un mock qui retourne un array de valeurs prédites
        y_pred = np.random.rand(self.dataset.x_test.shape[0], 1)
        self.model.predict = MagicMock(return_value=y_pred)
        
        # Appeler predict
        result = self.manager.predict(self.dataset)
        
        # Vérifier que predict a été appelé une fois et retourne le bon résultat
        self.model.predict.assert_called_once()
        assert np.array_equal(result, y_pred)
        
    def test_predict_multiple_models(self):
        """Test de la prédiction avec plusieurs modèles scikit-learn."""
        # Créer plusieurs modèles
        model1 = LinearRegression()
        model2 = RandomForestRegressor(n_estimators=5)
        models = [model1, model2]
        
        # Prédictions pour chaque modèle
        y_pred1 = np.ones((self.dataset.x_test.shape[0], 1))
        y_pred2 = np.ones((self.dataset.x_test.shape[0], 1)) * 2
        
        # Configurer les mocks pour predict
        model1.predict = MagicMock(return_value=y_pred1)
        model2.predict = MagicMock(return_value=y_pred2)
        
        # Créer le manager avec les multiples modèles
        manager = SklearnModelManager(models, self.model_config)
        
        # Appeler predict avec return_all=False pour obtenir la moyenne
        result = manager.predict(self.dataset, return_all=False)
        
        # Vérifier que la moyenne est calculée correctement
        expected_mean = np.mean([y_pred1, y_pred2], axis=0)
        assert np.array_equal(result, expected_mean)
        
        # Appeler predict avec return_all=True pour obtenir toutes les prédictions
        result_all = manager.predict(self.dataset, return_all=True)
        
        # Vérifier que toutes les prédictions sont retournées
        assert len(result_all) == 2
        assert np.array_equal(result_all[0], y_pred1)
        assert np.array_equal(result_all[1], y_pred2)


@pytest.mark.skipif(not is_tensorflow_available(), reason="TensorFlow n'est pas disponible")
class TestTFModelManager:
    """Tests pour le TFModelManager."""
    
    def setup_method(self):
        """Initialisation avant chaque test."""
        import tensorflow as tf
        
        # Créer un modèle TensorFlow simple
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        
        self.dataset = DummyDataset()
        self.models = [self.model]
        self.model_config = {'type': 'TensorFlowModel'}
        
        # Importer et créer le manager
        from nirs4all.core.model.model_manager import TFModelManager
        self.manager = TFModelManager(self.models, self.model_config)
        
    def test_initialization(self):
        """Test de l'initialisation du manager."""
        assert self.manager.models == self.models
        assert self.manager.model_config == self.model_config
        assert self.manager.framework == 'tensorflow'
        
    def test_train(self, monkeypatch):
        """Test de l'entraînement d'un modèle TensorFlow."""
        import tensorflow as tf
        
        # Créer un mock pour la méthode fit du modèle
        fit_mock = MagicMock()
        monkeypatch.setattr(self.model, "fit", fit_mock)
        
        # Créer un mock pour la méthode compile du modèle
        compile_mock = MagicMock()
        monkeypatch.setattr(self.model, "compile", compile_mock)
        
        # Paramètres d'entraînement
        training_params = {
            'loss': 'mse',
            'optimizer': 'adam',
            'epochs': 5,
            'batch_size': 16,
            'early_stopping': False
        }
        
        # Appeler train
        self.manager.train(
            dataset=self.dataset,
            training_params=training_params,
            metrics=['mse', 'mae'],
            no_folds=True
        )
        
        # Vérifier que compile et fit ont été appelés
        compile_mock.assert_called_once()
        fit_mock.assert_called_once()
        
    def test_predict_regression(self):
        """Test de la prédiction pour la régression avec un modèle TensorFlow."""
        import tensorflow as tf
        
        # Définir un mock pour la méthode predict du modèle
        y_pred = np.random.rand(self.dataset.x_test.shape[0], 1)
        self.model.predict = MagicMock(return_value=y_pred)
        
        # Appeler predict
        result = self.manager.predict(self.dataset, task='regression', no_folds=True)
        
        # Vérifier que predict a été appelé et que le résultat est correct
        self.model.predict.assert_called_once()
        assert np.array_equal(result, y_pred)
        
    def test_predict_classification(self):
        """Test de la prédiction pour la classification avec un modèle TensorFlow."""
        import tensorflow as tf
        
        # Configurer un modèle pour la classification
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
        ])
        
        # Remplacer le modèle dans le manager
        self.manager.models = [model]
        
        # Définir un mock pour la méthode predict du modèle
        # Simuler des probabilités de classe
        y_pred = np.array([
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5]
        ])
        model.predict = MagicMock(return_value=y_pred)
        
        # Appeler predict avec raw_class_output=False
        result = self.manager.predict(
            self.dataset,
            task='classification',
            no_folds=True,
            raw_class_output=False
        )
        
        # Vérifier que predict a été appelé et que les prédictions sont les indices des classes maximales
        expected = np.array([1, 0, 2])  # Indices des valeurs max par ligne
        assert np.array_equal(result, expected)
        
        # Appeler predict avec raw_class_output=True
        result_raw = self.manager.predict(
            self.dataset,
            task='classification',
            no_folds=True,
            raw_class_output=True
        )
        
        # Vérifier que les probabilités brutes sont retournées
        assert np.array_equal(result_raw, y_pred)


class TestModelManagerFactory:
    """Tests pour ModelManagerFactory."""
    
    @pytest.mark.skipif(not is_tensorflow_available(), reason="TensorFlow n'est pas disponible")
    def test_get_model_manager_tensorflow(self):
        """Test de la création d'un manager pour TensorFlow."""
        # Patch pour ModelBuilderFactory.build_models
        with patch("nirs4all.core.model.model_manager.ModelBuilderFactory.build_models") as mock_build_models:
            import tensorflow as tf
            
            # Configurer le mock pour retourner un modèle TensorFlow
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
                tf.keras.layers.Dense(1)
            ])
            mock_build_models.return_value = ([model], "tensorflow")
            
            # Appeler get_model_manager
            dataset = DummyDataset()
            model_config = {'type': 'TensorFlowModel'}
            manager = ModelManagerFactory.get_model_manager(model_config, dataset, 'regression')
            
            # Vérifier que le bon type de manager est retourné
            from nirs4all.core.model.model_manager import TFModelManager
            assert isinstance(manager, TFModelManager)
            
    def test_get_model_manager_sklearn(self):
        """Test de la création d'un manager pour scikit-learn."""
        # Patch pour ModelBuilderFactory.build_models
        with patch("nirs4all.core.model.model_manager.ModelBuilderFactory.build_models") as mock_build_models:
            # Configurer le mock pour retourner un modèle scikit-learn
            model = LinearRegression()
            mock_build_models.return_value = ([model], "sklearn")
            
            # Appeler get_model_manager
            dataset = DummyDataset()
            model_config = {'type': 'LinearRegression'}
            manager = ModelManagerFactory.get_model_manager(model_config, dataset, 'regression')
            
            # Vérifier que le bon type de manager est retourné
            assert isinstance(manager, SklearnModelManager)
            
    def test_get_model_manager_unsupported(self):
        """Test de la gestion des frameworks non supportés."""
        # Patch pour ModelBuilderFactory.build_models
        with patch("nirs4all.core.model.model_manager.ModelBuilderFactory.build_models") as mock_build_models:
            # Configurer le mock pour retourner un framework non supporté
            mock_build_models.return_value = (["dummy_model"], "unsupported_framework")
            
            # Vérifier qu'une exception est levée
            dataset = DummyDataset()
            model_config = {'type': 'UnsupportedModel'}
            with pytest.raises(ValueError):
                ModelManagerFactory.get_model_manager(model_config, dataset, 'regression')


class TestBaseModelManager:
    """Tests pour BaseModelManager."""
    
    def test_evaluate(self):
        """Test de la méthode evaluate de BaseModelManager."""
        # Préparation des données
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5.1])
        
        # Liste des métriques à évaluer
        metrics = ['mse', 'r2', 'mae']
        
        # Appel à la méthode evaluate
        result = BaseModelManager.evaluate(y_true, y_pred, metrics)
        
        # Vérifier que chaque métrique est correctement calculée
        assert 'mse' in result
        assert 'r2' in result
        assert 'mae' in result
        assert isinstance(result['mse'], float)
        assert isinstance(result['r2'], float)
        assert isinstance(result['mae'], float)
        
        # Calculer manuellement pour comparer
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        expected_mse = mean_squared_error(y_true, y_pred)
        expected_r2 = r2_score(y_true, y_pred)
        expected_mae = mean_absolute_error(y_true, y_pred)
        
        assert pytest.approx(result['mse'], abs=1e-6) == expected_mse
        assert pytest.approx(result['r2'], abs=1e-6) == expected_r2
        assert pytest.approx(result['mae'], abs=1e-6) == expected_mae
        
    def test_evaluate_with_invalid_metric(self):
        """Test de la gestion des métriques invalides."""
        # Préparation des données
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5.1])
        
        # Liste des métriques, incluant une métrique invalide
        metrics = ['mse', 'invalid_metric']
        
        # Appel à la méthode evaluate
        result = BaseModelManager.evaluate(y_true, y_pred, metrics)
        
        # Vérifier que 'mse' est calculé correctement
        assert 'mse' in result
        assert isinstance(result['mse'], float)
        
        # Vérifier que l'erreur pour la métrique invalide est capturée
        assert 'invalid_metric' in result
        assert isinstance(result['invalid_metric'], str)
        assert "Error" in result['invalid_metric']