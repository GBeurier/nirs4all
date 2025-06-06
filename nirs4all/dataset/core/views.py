from __future__ import annotations
import numpy as np
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .blocks import Block


class TensorView:
    """
    Vue paresseuse : (Block, rows, transform?).
    `.eval()` = ndarray (copie unique si transform not None).
    """
    __slots__ = ("block", "rows", "transform")

    def __init__(self, block: Block, rows, transform: Callable | None = None) -> None:
        self.block = block
        self.rows = rows
        self.transform = transform

    def eval(self) -> np.ndarray:
        """Évalue la vue en créant une copie si nécessaire."""
        data = self.block.slice(self.rows)
        if self.transform is not None:
            return self.transform(data)
        return data
