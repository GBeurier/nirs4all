from typing import Any, Dict, Union, Type, List
import importlib
import inspect

class PipelineBuilder:
    """
    Transforme une configuration brute en instances prêtes à l'exécution.

    - Dict avec clé 'class' (+ 'params') → instancie la classe.
    - Type (classe) → instancie sans paramètres.
    - Instance → conserve telle quelle.
    - String → recherche dans `presets_map` et replace récursivement ; sinon conservé.
    - List → transforme récursivement chacun de ses éléments.
    - Dict sans 'class' → étape de contrôle : transforme récursivement chaque valeur.
    """
    def __init__(
        self,
        presets_map: Dict[str, Union[Type[Any], Dict[str, Any]]]
    ) -> None:
        self.presets_map = presets_map

    def build_pipeline(self, cfg: List[Any]) -> List[Any]:
        # print("Building pipeline")
        out = [self._build_item(item) for item in cfg]
        return out

    def _build_item(self, item: Any) -> Any:
        # 1) dictionnaire avec clé 'class' → instantiation
        if isinstance(item, dict) and 'class' in item:
            # print(f"Instantiating dict class {item['class']}")
            cls_node = item['class']
            params = item.get('params', {}) or {}
            mod_name, _, cls_or_func_name = cls_node.rpartition(".")
            try:
                mod = importlib.import_module(mod_name)
                cls_or_func = getattr(mod, cls_or_func_name)
                return cls_or_func(**params)
                
            except (ImportError, AttributeError):
                return item

        # 2) liste → recursion
        if isinstance(item, list):
            # print(f"Recursing into list {item}")
            return [self._build_item(elem) for elem in item]

        # 3) string → lookup presets_map
        if isinstance(item, str):
            # print(f"Looking up string {item}")
            preset = self.presets_map.get(item)
            if preset is None:
                # pas de preset → on conserve le string
                return item
            return self._build_item(preset)

        # 4) type (classe) → instanciation sans param
        if inspect.isclass(item):
            # print(f"Instantiating class {item}")
            return item()

        # 5) dict sans 'class' → étape controle: map values
        if isinstance(item, dict):
            # print(f"Mapping dict {item}")
            return {k: self._build_item(v) for k, v in item.items()}

        # 6) autre (instance, None, littéral) → ne pas toucher
        # print(f"Returning item {item}")
        return item

# Exemple d'utilisation :
#
# presets = {
#     'uncluster': {'class': SomeUnclusterClass},
#     'PlotData': {'class': PlotData},
#     'PlotClusters': {'class': PlotClusters},
#     'PlotResults': {'class': PlotResults},
# }
# builder = PipelineBuilder(presets)
# ready_cfg = builder.build_config(config)
# pipeline = ready_cfg['pipeline']  # prêt à runner