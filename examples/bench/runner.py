import logging
import inspect
from typing import Any, Dict, List, Optional, Union
from sklearn.base import TransformerMixin, BaseEstimator
from spectraset import SpectraDataset

logger = logging.getLogger(__name__)

class PipelineStep:
    """Wrapper pour un transformer ou estimator prêt à runner."""
    def __init__(self, obj: Union[TransformerMixin, BaseEstimator]):
        self.obj = obj

    def run(self, data: SpectraDataset, prefix: str = ""):
        # transformer
        if isinstance(self.obj, TransformerMixin):
            logger.info(f"[Transformer] {self.obj}")
            # X = data.X(set="train")
            # Xt = self.obj.fit(X)
            # x = self.obj.transform(X)

            # data["X"] = Xt
            print(prefix, "Transformer", self.obj)
            
        # estimator
        elif isinstance(self.obj, BaseEstimator):
            logger.info(f"[Estimator] {self.obj}")
            # model = self.obj.fit(data["X"], data["y"])
            # data["model"] = model
            print(prefix, "Estimator", self.obj)
            
        else:
            # autre type (None, int, float, str, dict, list, etc.)
            # on ne fait rien
            print(prefix, "Unknwon Step", self.obj)
            
        # fallback
        logger.warning(f"[PipelineStep] objet inconnu : {self.obj}")
        

class ControlStep:
    """
    Étape de contrôle : 
      - key est le nom ("feature_augmentation", "cluster", "branch", "uncluster", ...)
      - spec est la valeur (liste, dict, None, etc.)
    """
    def __init__(self, key: str, spec: Any, runner: "PipelineRunner"):
        self.key = key
        self.spec = spec
        self.runner = runner

    def run(self, data: SpectraDataset, prefix: str = ""):
        logger.info(f"[ControlStep] {self.key}")
        # si c'est une liste, on considère un sous‐pipeline
        # print("ControlStep", self.key)
        
        if isinstance(self.spec, list):
            results = []
            print(prefix, self.key, "[]")
            for item in self.spec:
                # chaque item devient un mini-runner
                sub_runner = PipelineRunner([item], self.runner.globals_map)
                sub_runner.run(data, prefix + "-")
                # res = sub_runner.run(data)
                # results.append(res)
                # print("Sub-runner", sub_runner)
            # data[self.key] = results
        # si c'est un dict, on l'injecte tel quel
        elif isinstance(self.spec, dict):
            # data[self.key] = self.spec
            print(prefix, self.key, "{}")
            sub_runner = PipelineRunner([self.spec], self.runner.globals_map)
            sub_runner.run(data, prefix + "-")
        else:
            print(prefix, self.key)
            # data[self.key] = self.spec

class PipelineRunner:
    """
    Exécute votre pipeline, avec auto‐instanciation depuis strings, classes, dicts.
    
    Params
    ------
    pipeline_def : List[Any]
      Votre liste brute (strings, dicts, classes, instances, listes).
    globals_map : Dict[str, Any]
      Mapping des noms → classes/fonctions pour `eval("FooBar(...)")`.
    """
    def __init__(self, 
                 pipeline_def: List[Any],
                 globals_map: Optional[Dict[str, Any]] = None):
        self.globals_map = globals_map or {}
        self.steps = [self._make_step(item) for item in pipeline_def]

    def _instantiate(self, item: Any) -> Any:
        """Si string, classe ou dict{'class':…}, on instancie; sinon on renvoie tel quel."""
        # 1) None
        if item is None:
            return None

        # 2) string
        if isinstance(item, str):
            # ex. "MinMaxScaler()", ou "uncluster"
            if "(" in item and item.strip().endswith(")"):
                try:
                    return eval(item, self.globals_map)
                except Exception as e:
                    logger.warning(f"Échec eval('{item}'): {e}")
                    return item  # fallback
            # chaîne sans '()' → contrôle ou preset
            return item

        # 3) dict {'class':..., 'params':...}
        if isinstance(item, dict) and "class" in item:
            cls = item["class"]
            params = item.get("params", {}) or {}
            # cls peut être string ou type
            if isinstance(cls, str):
                inst = self._instantiate(cls)
            else:
                inst = cls
            if inspect.isclass(inst):
                return inst(**params)
            # si c'est déjà une instance
            return inst

        # 4) type → instancier sans args
        if inspect.isclass(item):
            return item()

        # 5) liste → resolve récursivement
        if isinstance(item, list):
            return [self._instantiate(x) for x in item]

        # 6) instance ou autre → on passe
        return item

    def _make_step(self, raw: Any):
        """
        → PipelineStep si transformer/estimator
        → PipelineRunner si liste
        → ControlStep sinon (dict ou string)
        """
        inst = self._instantiate(raw)

        # Transformer / Estimator ?
        if isinstance(inst, (TransformerMixin, BaseEstimator)):
            return PipelineStep(inst)

        # Sous-pipeline
        if isinstance(inst, list):
            return PipelineRunner(inst, self.globals_map)

        # Contrôle : dict ou string
        if isinstance(raw, dict):
            key = next(iter(raw))
            # print(raw, key)
            spec = inst[key]
            return ControlStep(key, spec, runner=self)
        if isinstance(inst, str):
            # ex. "uncluster", "PlotData"…
            # print(">>>", inst)
            return ControlStep(inst, None, runner=self)

        # tout le reste (numérique, None…)  
        return PipelineStep(raw)

    def run(self, data: SpectraDataset, prefix: str = ""):
        """
        Exécute toutes les étapes dans l'ordre.
        `data` doit au minimum contenir : {"X":…, "y":…}.
        """
        for step in self.steps:
            step.run(data, prefix=prefix)
