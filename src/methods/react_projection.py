from functools import partial
from typing import Callable, Dict, List

import torch
import torch.fx as fx
from torchvision.models.feature_extraction import get_graph_node_names

from src.methods.projection import Projection
from src.methods.react import condition_fn, insert_fn, reactify


class ReActProjection(Projection):
    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: List[str],
        pooling_name: str = "max",
        graph_nodes_names_thr: Dict[str, float] = {"flatten": 1.0},
        insert_node_fn: Callable = insert_fn,
        aggregation_method=None,
        *args,
        **kwargs,
    ):
        self.graph_nodes_names_thr = graph_nodes_names_thr
        self.insert_node_fn = insert_node_fn
        for node_name, thr in self.graph_nodes_names_thr.items():
            model = reactify(
                model,
                condition_fn=partial(condition_fn, equals_to=node_name),
                insert_fn=partial(self.insert_node_fn, thr=thr),
            )

        super().__init__(model, features_nodes, pooling_name, aggregation_method, *args, **kwargs)
