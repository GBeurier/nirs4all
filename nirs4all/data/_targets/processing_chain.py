"""Processing chain management for target transformations."""

from typing import Dict, List, Optional, Set

from sklearn.base import TransformerMixin


class ProcessingChain:
    """
    Manages the chain of processing transformations applied to target data.

    Tracks the lineage of processing steps, their relationships, and
    associated transformers. Provides efficient ancestry queries with caching.

    Attributes:
        _processing_ids (list of str): Ordered list of all processing names
        _processing_set (set of str): Set for O(1) existence checks
        _ancestors (dict of str to str): Maps each processing to its immediate parent
        _transformers (dict of str to TransformerMixin): Maps each processing to its transformer
        _ancestry_cache (dict of str to list of str): Cached full ancestry chains for performance

    Examples:
    >>> chain = ProcessingChain()
    >>> chain.add_processing('raw', None, None)
    >>> chain.add_processing('numeric', 'raw', encoder)
    >>> chain.add_processing('scaled', 'numeric', scaler)
    >>> chain.get_ancestry('scaled')
    ['raw', 'numeric', 'scaled']
    >>> chain.get_path('scaled', 'raw')
    (['scaled', 'numeric', 'raw'], 'inverse')
    """

    def __init__(self):
        """Initialize empty processing chain."""
        self._processing_ids: List[str] = []
        self._processing_set: Set[str] = set()
        self._ancestors: Dict[str, str] = {}
        self._transformers: Dict[str, TransformerMixin] = {}
        self._ancestry_cache: Dict[str, List[str]] = {}

    def add_processing(self,
                      name: str,
                      ancestor: Optional[str] = None,
                      transformer: Optional[TransformerMixin] = None) -> None:
        """
        Register a new processing step in the chain.

        Args:
            name (str): Unique name for this processing
            ancestor (str, optional): Name of parent processing (None for root)
            transformer (TransformerMixin, optional): Transformer used to create this processing from ancestor

        Raises:
            ValueError: If processing name already exists
            ValueError: If ancestor doesn't exist

        Notes:
        Invalidate ancestry cache for new processing.
        """
        if name in self._processing_set:
            raise ValueError(f"Processing '{name}' already exists")

        if ancestor is not None and ancestor not in self._processing_set:
            raise ValueError(f"Ancestor '{ancestor}' does not exist")

        self._processing_ids.append(name)
        self._processing_set.add(name)

        if ancestor is not None:
            self._ancestors[name] = ancestor

        if transformer is not None:
            self._transformers[name] = transformer

        # Invalidate ancestry cache for new processing
        self._ancestry_cache.clear()

    def has_processing(self, name: str) -> bool:
        """
        Check if a processing exists.

        Args:
            name (str): Processing name to check

        Returns:
            bool: True if processing exists
        """
        return name in self._processing_set

    def get_transformer(self, name: str) -> Optional[TransformerMixin]:
        """
        Get the transformer for a processing.

        Args:
            name (str): Processing name

        Returns:
            TransformerMixin or None: The transformer, or None if no transformer exists
        """
        return self._transformers.get(name)

    def get_ancestry(self, name: str) -> List[str]:
        """
        Get the full ancestry chain for a processing.

        Args:
            name (str): Processing name

        Returns:
            list of str: Processing names from root to the specified processing

        Raises:
            ValueError: If processing doesn't exist

        Notes:
        Results are cached for performance. Cache is invalidated when
        new processings are added.
        """
        if not self.has_processing(name):
            raise ValueError(f"Processing '{name}' not found")

        # Check cache first
        if name in self._ancestry_cache:
            return self._ancestry_cache[name].copy()

        # Build ancestry chain
        ancestry = []
        current = name

        while current is not None:
            ancestry.append(current)
            current = self._ancestors.get(current)

        ancestry.reverse()

        # Cache and return
        self._ancestry_cache[name] = ancestry
        return ancestry.copy()

    def get_path(self,
                from_processing: str,
                to_processing: str) -> tuple[List[str], str]:
        """
        Find transformation path between two processings.

        Args:
            from_processing (str): Starting processing name
            to_processing (str): Target processing name

        Returns:
            path (list of str): Sequence of processing names to traverse
            direction (str): Either 'forward', 'inverse', or 'mixed'

        Raises:
            ValueError: If either processing doesn't exist
            ValueError: If no path exists between processings

        Examples:
        >>> path, direction = chain.get_path('scaled', 'numeric')
        >>> path
        ['scaled', 'numeric']
        >>> direction
        'inverse'
        """
        if not self.has_processing(from_processing):
            raise ValueError(f"Processing '{from_processing}' not found")
        if not self.has_processing(to_processing):
            raise ValueError(f"Processing '{to_processing}' not found")

        from_ancestry = self.get_ancestry(from_processing)
        to_ancestry = self.get_ancestry(to_processing)

        # Find common ancestor
        common_ancestor = None
        for ancestor in reversed(from_ancestry):
            if ancestor in to_ancestry:
                common_ancestor = ancestor
                break

        if common_ancestor is None:
            raise ValueError(
                f"No common ancestor between '{from_processing}' and '{to_processing}'"
            )

        # Build path
        path = []

        # Path from from_processing to common ancestor (inverse direction)
        current = from_processing
        while current != common_ancestor:
            path.append(current)
            current = self._ancestors[current]

        # Add common ancestor
        if common_ancestor != to_processing:
            path.append(common_ancestor)

            # Path from common ancestor to to_processing (forward direction)
            forward_path = []
            current = to_processing
            while current != common_ancestor:
                forward_path.append(current)
                current = self._ancestors[current]
            path.extend(reversed(forward_path))
        else:
            path.append(common_ancestor)

        # Determine direction
        if from_processing == to_processing:
            direction = 'identity'
        elif to_processing in from_ancestry:
            direction = 'inverse'
        elif from_processing in to_ancestry:
            direction = 'forward'
        else:
            direction = 'mixed'

        return path, direction

    @property
    def processing_ids(self) -> List[str]:
        """
        Get list of all processing IDs.

        Returns:
            list of str: Copy of processing IDs list
        """
        return self._processing_ids.copy()

    @property
    def num_processings(self) -> int:
        """
        Get number of processings in the chain.

        Returns:
            int: Number of registered processings
        """
        return len(self._processing_ids)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ProcessingChain(processings={len(self._processing_ids)})"
