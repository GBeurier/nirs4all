"""Base classes for model operators.

This module defines the abstract base classes for model operators
used in nirs4all pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModelOperator(ABC):
    """Abstract base class for all model operators.

    Model operators are building blocks in pipelines that represent
    machine learning models (sklearn, tensorflow, pytorch, etc.).

    The actual execution logic is handled by corresponding controllers
    in the nirs4all.controllers.models module.
    """

    @abstractmethod
    def get_controller_type(self) -> str:
        """Return the type of controller that handles this operator.

        Returns
        -------
        str
            Controller type identifier (e.g., 'sklearn', 'tensorflow', 'pytorch')
        """
        pass

    @abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this operator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this operator and
            contained subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> 'BaseModelOperator':
        """Set the parameters of this operator.

        Parameters
        ----------
        **params : dict
            Operator parameters.

        Returns
        -------
        self : BaseModelOperator
            Operator instance.
        """
        pass
