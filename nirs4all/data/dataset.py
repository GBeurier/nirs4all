# dataset.py

from dataclasses import dataclass
from typing import Optional, Any, List
import numpy as np

@dataclass
class Dataset:
    """
    A class representing a dataset with different stages of data processing.
    """
    # id: Optional[str] = None  # Dataset ID
    # name: Optional[str] = None  # Dataset name
    
    _x_train: Optional[np.ndarray] = None  # Training data - 4D array (augmentations, samples, transformations, features)
    _y_train: Optional[np.ndarray] = None  # Training labels
    _group_train: Optional[np.ndarray] = None  # Training groups
    _y_train_init: Optional[np.ndarray] = None  # Initial training labels
    
    _x_test: Optional[np.ndarray] = None  # Testing data - 4D array (augmentations, samples, transformations, features) or 1D boolean array
    _y_test: Optional[np.ndarray] = None
    _group_test: Optional[np.ndarray] = None
    _y_test_init: Optional[np.ndarray] = None  # Initial testing labels
    
    _folds: Optional[List[tuple[np.ndarray, np.ndarray]]] = None
    y_transformer: Optional[Any] = None
    num_classes = 0
    
    # Categorical data handling
    y_train_categorical_info: Optional[dict] = None  # Store categorical info for y_train
    y_test_categorical_info: Optional[dict] = None  # Store categorical info for y_test
    y_train_column_names: Optional[List[str]] = None  # Store original column names for y_train
    y_test_column_names: Optional[List[str]] = None  # Store original column names for y_test
    
    def _transform_single_column_categorical(self, codes_column: np.ndarray, categories: list) -> list:
        """Helper to transform a single column of numerical codes to categorical strings."""
        transformed_column = []
        for code_val in codes_column:
            try:
                # Attempt to convert to float first for robustness, then int for indexing
                int_code = int(round(float(code_val)))
                if 0 <= int_code < len(categories):
                    transformed_column.append(categories[int_code])
                else:
                    transformed_column.append(None)  # Out of bounds, return None as per test
            except (ValueError, TypeError):
                transformed_column.append(None)  # Non-convertible, return None as per test
        return transformed_column

    def _get_key_for_single_col_transform(self, categorical_info: dict, column_names: Optional[List[str]]) -> Optional[str]:
        """
        Determines the key to use from categorical_info for a single prediction column.
        """
        if not categorical_info:
            return None

        # If column_names are provided, they guide the key selection.
        if column_names:
            if len(column_names) == 1 and column_names[0] in categorical_info:
                return column_names[0]  # Direct match for single named column
            # For multiple column_names, find the first one that is categorical
            for name in column_names:
                if name in categorical_info:
                    return name
            return None  # No matching categorical column found in column_names

        # If no column_names, but categorical_info is unambiguous (single key)
        if not column_names and len(categorical_info) == 1:
            return list(categorical_info.keys())[0]
        
        return None

    def has_categorical_columns(self, target: str = 'train') -> bool:
        """Check if the dataset contains categorical columns for the specified target."""
        categorical_info = self.y_train_categorical_info if target == 'train' else self.y_test_categorical_info
        return categorical_info is not None and len(categorical_info) > 0
    
    def inverse_transform_categorical(self, y_pred: np.ndarray, target: str = 'train') -> np.ndarray:
        """
        Convert numerical predictions back to original categorical string values.
        
        Parameters:
            y_pred: Numerical predictions array. Assumed to have columns in the same order as original Y.
            target: Which categorical info and column names to use ('train' or 'test').
            
        Returns:
            Array with original categorical string values. Object dtype for columns with strings.
        """
        categorical_info = self.y_train_categorical_info if target == 'train' else self.y_test_categorical_info
        column_names = self.y_train_column_names if target == 'train' else self.y_test_column_names

        # Check if transformation is needed/possible
        if not categorical_info or not self.has_categorical_columns(target=target):
            return y_pred

        original_shape = y_pred.shape
        y_pred_2d = y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred
        num_pred_cols = y_pred_2d.shape[1]

        # Case 1: Single prediction column
        if num_pred_cols == 1:
            cat_info_key = self._get_key_for_single_col_transform(categorical_info, column_names)
            if cat_info_key:
                info = categorical_info.get(cat_info_key, {})
                categories = info.get('categories')
                if categories is not None:  # Ensure categories list exists
                    transformed_col = self._transform_single_column_categorical(y_pred_2d[:, 0], categories)
                    return np.array(transformed_col, dtype=object).reshape(original_shape)
            # If key or categories not found, return original predictions
            return y_pred

        # Case 2: Multiple prediction columns
        # This requires column_names to map predictions to categories correctly.
        if not column_names or num_pred_cols != len(column_names):
            return y_pred  # Cannot process multi-column without matching column_names

        # Proceed with multi-column transformation
        result = y_pred_2d.astype(object).copy()  # Ensure object type for strings
        for col_idx in range(num_pred_cols):
            col_name = column_names[col_idx]
            if col_name in categorical_info:
                info = categorical_info.get(col_name, {})
                categories = info.get('categories')
                if categories is not None:  # Ensure categories list exists
                    transformed_data = self._transform_single_column_categorical(y_pred_2d[:, col_idx], categories)
                    result[:, col_idx] = transformed_data
                # else: If no categories for this specific column, it remains as is (original numeric data)
            # else: If col_name not in categorical_info, it remains as is (original numeric data)
        
        return result.reshape(original_shape)
    
    def inverse_transform(self, y_pred: np.ndarray, target: str = 'train') -> np.ndarray:
        """
        Apply all inverse transformations to predictions:
        1. First apply any numerical transformer (e.g., StandardScaler)
        2. Then convert categorical codes back to original string values
        
        Parameters:
            y_pred: Predictions to transform
            target: Which categorical info to use ('train' or 'test') for categorical transformation
            
        Returns:
            Transformed predictions
        """
        result = y_pred
        if self.y_transformer is not None:
            original_shape = y_pred.shape
            if y_pred.ndim == 1:
                y_pred_2d = y_pred.reshape(-1, 1)
            else:
                y_pred_2d = y_pred
            
            transformed_2d = self.y_transformer.inverse_transform(y_pred_2d)
            
            if transformed_2d.shape != y_pred_2d.shape and original_shape == transformed_2d.shape:
                result = transformed_2d  # Some transformers might change shape, e.g. OneHotEncoder inverse
            elif y_pred.ndim == 1 and transformed_2d.shape[1] == 1:
                result = transformed_2d.ravel()
            else:
                result = transformed_2d.reshape(original_shape) if transformed_2d.size == y_pred.size else transformed_2d

        if self.has_categorical_columns(target=target):
            result = self.inverse_transform_categorical(result, target=target)
            
        return result
    
    def filter_x(self, data, union_type='concat', indices=None, disable_augmentation=False):
        n_augmentations, n_samples, n_transformations, n_features = data.shape

        if indices is not None:
            data = data[:, indices, :, :]
            n_samples = data.shape[1]
            
        if disable_augmentation:
            n_augmentations = 1
            data = data[0, :, :, :]
    
        total_samples = n_augmentations * n_samples

        if union_type is None:
            return data

        if union_type == 'concat' or union_type == 'c':
            return data.reshape(total_samples, n_transformations * n_features)
        elif union_type == 'interlaced' or union_type == 'i':
            data = data.reshape(total_samples, n_transformations, n_features)
            return data.reshape(total_samples, n_transformations * n_features, order='F')
        elif union_type == 'transpose_union' or union_type == 'tu':
            return data.reshape(total_samples, n_transformations, n_features)
        elif union_type == 'union' or union_type == 'u':
            return np.transpose(data.reshape(total_samples, n_transformations, n_features), (0, 2, 1))
        else:
            raise ValueError(f"Invalid union type: {union_type}")
    
    def filter_y(self, data, indices=None, disable_augmentation=False):
        filtered_y = data[indices] if indices is not None else data
        if disable_augmentation:
            return filtered_y
        return np.repeat(filtered_y, self._x_train.shape[0], axis=0)
    
    def fold_data(self, union_type='concat', no_folds=False):
        if self.folds is None or len(self.folds) == 0 or no_folds:
            yield self.x_train_(union_type), self.y_train, self.x_test_(union_type), self.y_test
        else:
            for (train_indices, test_indices) in self.folds:
                x_train_fold = self.x_train_(union_type, train_indices)
                y_train_fold = self.y_train_(train_indices)
                x_val_fold = self.x_train_(union_type, test_indices)
                y_val_fold = self.y_train_(test_indices)
                yield x_train_fold, y_train_fold, x_val_fold, y_val_fold
    
    @property
    def x_train(self) -> np.ndarray:
        return self._x_train
    
    @property
    def x_test(self) -> np.ndarray:
        return self._x_test
    
    def x_train_(self, union_type='concat', indices=None, disable_augmentation=False) -> np.ndarray:
        return self.filter_x(self._x_train, union_type, indices, disable_augmentation)
    
    def x_test_(self, union_type='concat', indices=None, disable_augmentation=False) -> np.ndarray:
        return self.filter_x(self._x_test, union_type, indices, disable_augmentation)
    
    def y_train_(self, indices=None, disable_augmentation=False) -> np.ndarray:
        return self.filter_y(self._y_train, indices, disable_augmentation)
    
    def y_test_(self, indices=None, disable_augmentation=False) -> np.ndarray:
        return self.filter_y(self._y_test, indices, disable_augmentation)
    
    @property
    def raw_x_train(self) -> np.ndarray:
        return self._x_train
    
    @property
    def raw_x_test(self) -> np.ndarray:
        return self._x_test
    
    @property
    def y_train(self) -> np.ndarray:
        if self._x_train is None:
            return self._y_train
        n_augmentations = self._x_train.shape[0]
        return np.repeat(self._y_train, n_augmentations, axis=0)
    
    @property
    def y_test(self) -> np.ndarray:
        if self._x_test is None:
            return self._y_test
        
        y_test = self._y_test if not self.test_is_indices else self._y_train[self._x_test]
        n_augmentations = self._x_test.shape[0]
        y_test = np.repeat(y_test, n_augmentations, axis=0)
        return y_test
    
    @property
    def raw_y_train(self) -> np.ndarray:
        return self._y_train
    
    @property
    def raw_y_test(self) -> np.ndarray:
        return self._y_test
    
    @property
    def y_train_init(self) -> np.ndarray:
        return self._y_train_init
    
    @property
    def y_test_init(self) -> np.ndarray:
        return self._y_test_init
    
    @property
    def group_train(self) -> np.ndarray:
        return self._group_train
    
    @property
    def group_test(self) -> np.ndarray:
        group_test = self._group_test if not self.test_is_indices else self._group_train[self._x_test]
        n_augmentations = self._x_train.shape[0]
        group_test = np.repeat(group_test, n_augmentations, axis=0)
        return group_test
    
    @property
    def folds(self) -> List[tuple[np.ndarray, np.ndarray]]:
        return self._folds
    
    @property
    def test_is_indices(self) -> bool:
        if self._x_test is None:
            return False
        return self._x_test.ndim == 1
    
    @x_train.setter
    def x_train(self, value: np.ndarray):
        if value.ndim not in [2, 4]:
            raise ValueError(f"Invalid x_train shape: {value.shape}")
        
        if value.ndim == 2:  # convert to 4D
            value = value[np.newaxis, :, np.newaxis, :]
        
        self._x_train = value
    
    @x_test.setter
    def x_test(self, value: np.ndarray):
        if value.ndim not in [1, 2, 4]:
            raise ValueError(f"Invalid x_test shape: {value.shape}")
        
        if value.ndim == 2:  # convert to 4D
            value = value[np.newaxis, :, np.newaxis, :]
        
        self._x_test = value
        
    @raw_x_train.setter
    def raw_x_train(self, value: np.ndarray):
        if value.ndim not in [4]:
            raise ValueError(f"Invalid raw_x_train shape: {value.shape}")
        
        self._x_train = value
        
    @raw_x_test.setter
    def raw_x_test(self, value: np.ndarray):
        if value.ndim not in [4]:
            raise ValueError(f"Invalid raw_x_test shape: {value.shape}")
        
        self._x_test = value
    
    @y_train.setter
    def y_train(self, value: np.ndarray):
        if value.ndim not in [1, 2]:
            raise ValueError(f"Invalid y_train shape: {value.shape}")

        if value.ndim == 1:  # convert to 2D
            value = value[:, np.newaxis]

        if self._x_train.shape[1] != value.shape[0]:
            raise ValueError(f"Invalid y_train shape: {value.shape}. Expected {self._x_train.shape[1]} samples.")

        self._y_train = value
        
    @y_train_init.setter
    def y_train_init(self, value: np.ndarray):
        if value.ndim not in [1, 2]:
            raise ValueError(f"Invalid y_train shape: {value.shape}")

        if value.ndim == 1:
            value = value[:, np.newaxis]
            
        if self._x_train.shape[1] != value.shape[0]:
            raise ValueError(f"Invalid y_train shape: {value.shape}. Expected {self._x_train.shape[1]} samples.")
        
        self._y_train = value
        self._y_train_init = value
    
    @y_test.setter
    def y_test(self, value: np.ndarray):
        if self.test_is_indices:
            raise ValueError("Cannot set y_test for index-based test data.")
        
        if value.ndim not in [1, 2]:
            raise ValueError(f"Invalid y_test shape: {value.shape}")
        
        if value.ndim == 1:
            value = value[:, np.newaxis]
        
        if self._x_test.shape[1] != value.shape[0]:
            raise ValueError(f"Invalid y_test shape: {value.shape}. Expected {self._x_test.shape[1]} samples.")
        
        self._y_test = value
    
    @y_test_init.setter
    def y_test_init(self, value: np.ndarray):
        if self.test_is_indices:
            raise ValueError("Cannot set y_test for index-based test data.")

        if value.ndim not in [1, 2]:
            raise ValueError(f"Invalid y_test shape: {value.shape}")

        if value.ndim == 1:
            value = value[:, np.newaxis]

        if self._x_test.shape[1] != value.shape[0]:
            raise ValueError(f"Invalid y_test shape: {value.shape}. Expected {self._x_test.shape[1]} samples.")

        self._y_test = value
        self._y_test_init = value
    
    @folds.setter
    def folds(self, value: List[tuple[np.ndarray, np.ndarray]]):
        self._folds = value

    def __str__(self) -> str:
        return self.to_str('concat')
    
    def to_str(self, agg='union') -> str:
        line1 = "Dataset("
        if self._x_train is not None:
            line1 += f"x_train:{self.x_train_(agg).shape} - "
        if self._y_train is not None:
            line1 += f"y_train:{self.y_train.shape}, "
        if self._x_test is not None:
            line1 += f"x_test:{self.x_test_(agg).shape} - "
        if self._y_test is not None:
            line1 += f"y_test:{self.y_test.shape})"
        extra = ""
        if self._folds is not None:
            folds_str = ', '.join([f"{len(train)}-{len(test)}" for train, test in self.folds])
            extra += f"Folds size: {folds_str}"
        if self.group_train is not None:
            extra += f"Groups: {self.group_train.shape} - {self.group_test.shape}"
        line2 = extra if len(extra) == 0 else f"\n{extra}"
        return f"{line1}{line2}"
