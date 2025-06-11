# pipeline/runtime_pipeline.py
import json, uuid
from typing import Dict, List
from .serialization import serialize_component, deserialize_component

class RuntimeNode:
    def __init__(self, op, parents: list[str] | None = None):
        self.id: str = op.id                      # hérite de PipelineOperation
        self.parents: list[str] = parents or []
        self.step_cfg = serialize_component(op.step, include_runtime=False)
        self.operator_cfg = serialize_component(op.operator, include_runtime=False)
        self.controller = op.controller.__class__.__name__

    # --- (dé-)sérialisation -------------------------------------------------
    def to_dict(self):
        return {
            "id": self.id,
            "parents": self.parents,
            "step": self.step_cfg,
            "operator": self.operator_cfg,
            "controller": self.controller,
        }

    @classmethod
    def from_dict(cls, d):
        dummy = type("Dummy", (), {})()           # petit hack pour porter l’id
        dummy.id = d["id"]
        node = cls.__new__(cls)
        RuntimeNode.__init__(node, dummy)         # type: ignore
        node.parents = d["parents"]
        node.step_cfg = d["step"]
        node.operator_cfg = d["operator"]
        node.controller = d["controller"]
        return node

    @property
    def operator(self):
        return deserialize_component(self.operator_cfg)

class RuntimePipeline:
    def __init__(self):
        self.nodes: Dict[str, RuntimeNode] = {}

    # -----------------------------------------------------------------------
    def add_op(self, op, parents: list[str] | None = None):
        node = RuntimeNode(op, parents)
        self.nodes[node.id] = node
        return node.id

    # -----------------------------------------------------------------------
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {"nodes": [n.to_dict() for n in self.nodes.values()]},
                f, indent=2
            )

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            j = json.load(f)
        rt = cls()
        for nd in j["nodes"]:
            node = RuntimeNode.from_dict(nd)
            rt.nodes[node.id] = node
        return rt
