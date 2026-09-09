"""
npc_array.py - NumPy-like interface for language models and ML at scale

This module provides NPCArray, a vectorized abstraction for model populations
that enables ensemble interactions, evolutionary optimization, and parallel
inference across heterogeneous model types (LLMs, sklearn, torch, etc.)

Core concepts:
- NPCArray wraps a collection of models (LLMs, ML models, or NPCs)
- Operations are lazy - they build a computation graph
- .collect() materializes results with automatic parallelization (like Spark)
- Same interface for single items (treated as 1D array of length 1)

Example:
    >>> models = NPCArray.from_llms(['gpt-4', 'claude-3', 'llama3'])
    >>> result = models.infer(prompts).filter(lambda r: len(r) > 100).vote()
    >>> result.collect()
"""

from __future__ import annotations
import copy
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    TYPE_CHECKING, Literal, Sequence
)
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    import polars as pl
    import pandas as pd

class OpType(Enum):
    SOURCE = "source"
    INFER = "infer"
    PREDICT = "predict"
    FIT = "fit"
    FORWARD = "forward"
    MAP = "map"
    FILTER = "filter"
    REDUCE = "reduce"
    CHAIN = "chain"
    EVOLVE = "evolve"
    JINX = "jinx"

@dataclass
class GraphNode:
    """A node in the lazy computation graph"""
    op_type: OpType
    params: Dict[str, Any] = field(default_factory=dict)
    parents: List['GraphNode'] = field(default_factory=list)
    result: Any = None
    shape: Optional[Tuple[int, ...]] = None

@dataclass
class ModelSpec:
    """Specification for a model in the array"""
    model_type: Literal["llm", "sklearn", "torch", "npc", "custom"]
    model_ref: Any
    provider: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.model_type, str(self.model_ref), self.provider))

@dataclass
class ResponseTensor:
    """
    Container for vectorized model outputs with shape information.
    Similar to numpy ndarray but for model responses.
    """
    data: np.ndarray
    model_specs: List[ModelSpec]
    prompts: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __getitem__(self, key):
        """NumPy-style indexing"""
        result_data = self.data[key]
        if isinstance(result_data, np.ndarray):
            if isinstance(key, int):
                new_specs = [self.model_specs[key]]
            elif isinstance(key, slice):
                new_specs = self.model_specs[key]
            elif isinstance(key, tuple) and len(key) == 2:
                model_key, prompt_key = key
                if isinstance(model_key, int):
                    new_specs = [self.model_specs[model_key]]
                else:
                    new_specs = self.model_specs[model_key] if isinstance(model_key, slice) else self.model_specs
                new_prompts = self.prompts[prompt_key] if self.prompts and isinstance(prompt_key, (int, slice)) else self.prompts
            else:
                new_specs = self.model_specs
            return ResponseTensor(
                data=result_data if result_data.ndim > 0 else np.array([result_data]),
                model_specs=new_specs if isinstance(new_specs, list) else [new_specs],
                prompts=self.prompts,
                metadata=self.metadata
            )
        return result_data

    def tolist(self) -> List:
        """Convert to nested Python list"""
        return self.data.tolist()

    def flatten(self) -> List:
        """Flatten to 1D list"""
        return self.data.flatten().tolist()

class NPCArray:
    """
    NumPy-like array for model populations.

    Supports:
    - LLMs (via provider/model name)
    - sklearn models (fitted or specs)
    - PyTorch models
    - NPCs (from npcpy)
    - Custom model wrappers

    All operations are lazy until .compute() is called.
    """

    def __init__(
        self,
        specs: List[ModelSpec],
        graph: Optional[GraphNode] = None
    ):
        self._specs = specs
        self._graph = graph or GraphNode(
            op_type=OpType.SOURCE,
            params={"specs": specs},
            shape=(len(specs),)
        )

    @classmethod
    def from_llms(
        cls,
        models: Union[str, List[str]],
        providers: Optional[Union[str, List[str]]] = None,
        **config
    ) -> 'NPCArray':
        """
        Create NPCArray from LLM model names.

        Args:
            models: Single model name or list of model names
            providers: Optional provider(s) - auto-detected if not provided
            **config: Additional config passed to all models

        Example:
            >>> arr = NPCArray.from_llms(['gpt-4', 'claude-3', 'llama3'])
            >>> arr = NPCArray.from_llms('gpt-4')
        """
        if isinstance(models, str):
            models = [models]

        if providers is None:
            providers = [None] * len(models)
        elif isinstance(providers, str):
            providers = [providers] * len(models)
        elif len(providers) == 1:
            providers = providers * len(models)

        specs = [
            ModelSpec(
                model_type="llm",
                model_ref=model,
                provider=provider,
                config=config.copy()
            )
            for model, provider in zip(models, providers)
        ]

        return cls(specs)

    @classmethod
    def from_npcs(cls, npcs: Union[Any, List[Any]]) -> 'NPCArray':
        """
        Create NPCArray from NPC objects.

        Args:
            npcs: Single NPC or list of NPCs from npcpy
        """
        if not isinstance(npcs, list):
            npcs = [npcs]

        specs = [
            ModelSpec(
                model_type="npc",
                model_ref=npc,
                provider=getattr(npc, 'provider', None),
                config={"model": getattr(npc, 'model', None)}
            )
            for npc in npcs
        ]

        return cls(specs)

    @classmethod
    def from_sklearn(
        cls,
        models: Union[Any, List[Any]],
        fitted: bool = True
    ) -> 'NPCArray':
        """
        Create NPCArray from sklearn models.

        Args:
            models: Fitted sklearn model(s) or estimator class names
            fitted: Whether models are already fitted
        """
        if not isinstance(models, list):
            models = [models]

        specs = [
            ModelSpec(
                model_type="sklearn",
                model_ref=model,
                config={"fitted": fitted}
            )
            for model in models
        ]

        return cls(specs)

    @classmethod
    def from_torch(
        cls,
        models: Union[Any, List[Any]],
        device: str = "cpu"
    ) -> 'NPCArray':
        """
        Create NPCArray from PyTorch models.

        Args:
            models: PyTorch nn.Module(s)
            device: Device to run inference on
        """
        if not isinstance(models, list):
            models = [models]

        specs = [
            ModelSpec(
                model_type="torch",
                model_ref=model,
                config={"device": device}
            )
            for model in models
        ]

        return cls(specs)

    @classmethod
    def from_specs(
        cls,
        specs: List[Dict[str, Any]]
    ) -> 'NPCArray':
        """
        Create NPCArray from model specification dicts.

        Args:
            specs: List of dicts with 'type', and type-specific params

        Example:
            >>> specs = [
            ...     {'type': 'RandomForest', 'n_estimators': 100},
            ...     {'type': 'XGBoost', 'max_depth': 5}
            ... ]
            >>> arr = NPCArray.from_specs(specs)
        """
        model_specs = [
            ModelSpec(
                model_type="sklearn",
                model_ref=spec.get('type'),
                config={k: v for k, v in spec.items() if k != 'type'}
            )
            for spec in specs
        ]

        return cls(model_specs)

    @classmethod
    def meshgrid(cls, **param_ranges) -> 'NPCArray':
        """
        Create NPCArray from cartesian product of parameters.

        Args:
            **param_ranges: Parameter name -> list of values

        Example:
            >>> arr = NPCArray.meshgrid(
            ...     model=['gpt-4', 'claude-3'],
            ...     temperature=[0.0, 0.5, 1.0]
            ... )
            >>> arr.shape
        """
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]

        specs = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            model = config.pop('model', None)
            if not model:
                raise ValueError("No model specified in ModelSpec config.")
            provider = config.pop('provider', None)
            specs.append(ModelSpec(
                model_type="llm",
                model_ref=model,
                provider=provider,
                config=config
            ))

        return cls(specs)

    @classmethod
    def from_matrix(
        cls,
        matrix: List[Dict[str, Any]]
    ) -> 'NPCArray':
        """
        Create NPCArray from a matrix of model configurations.

        This is particularly useful for defining model arrays in Jinx templates
        where you want explicit control over each model configuration.

        Args:
            matrix: List of model configuration dicts. Each dict should have:
                - 'model': model name/reference (required)
                - 'provider': provider name (optional)
                - 'type': model type - 'llm', 'npc', 'sklearn', 'torch' (default: 'llm')
                - Any additional config parameters

        Example:
            >>>
            >>> matrix = [
            ...     {'model': 'gpt-4', 'provider': 'openai', 'temperature': 0.7},
            ...     {'model': 'claude-3-opus', 'provider': 'anthropic', 'temperature': 0.5},
            ...     {'model': 'your-local-model', 'provider': 'ollama', 'temperature': 0.8},
            ... ]
            >>> arr = NPCArray.from_matrix(matrix)

            >>>
            >>> matrix = [
            ...     {'model': 'gpt-4', 'type': 'llm', 'provider': 'openai'},
            ...     {'model': my_npc, 'type': 'npc'},
            ...     {'model': sklearn_model, 'type': 'sklearn'},
            ... ]
        """
        specs = []
        for config in matrix:
            model_type = config.get('type', 'llm')
            model_ref = config.get('model')
            provider = config.get('provider')

            extra_config = {
                k: v for k, v in config.items()
                if k not in ('type', 'model', 'provider')
            }

            specs.append(ModelSpec(
                model_type=model_type,
                model_ref=model_ref,
                provider=provider,
                config=extra_config
            ))

        return cls(specs)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the model array"""
        return (len(self._specs),)

    @property
    def specs(self) -> List[ModelSpec]:
        """Model specifications"""
        return self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        types = [s.model_type for s in self._specs]
        return f"NPCArray(shape={self.shape}, types={types})"

    def infer(
        self,
        prompts: Union[str, List[str]],
        **kwargs
    ) -> 'LazyResult':
        """
        Queue inference across all models for given prompts.

        Args:
            prompts: Single prompt or list of prompts
            **kwargs: Additional inference params (temperature, etc.)

        Returns:
            LazyResult with shape (n_models, n_prompts)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        new_node = GraphNode(
            op_type=OpType.INFER,
            params={"prompts": prompts, **kwargs},
            parents=[self._graph],
            shape=(len(self._specs), len(prompts))
        )

        return LazyResult(self._specs, new_node, prompts=prompts)

    def predict(
        self,
        X: Any,
        **kwargs
    ) -> 'LazyResult':
        """
        Queue prediction for sklearn/ML models.

        Args:
            X: Input features (array-like)
            **kwargs: Additional predict params

        Returns:
            LazyResult with predictions
        """
        new_node = GraphNode(
            op_type=OpType.PREDICT,
            params={"X": X, **kwargs},
            parents=[self._graph],
            shape=(len(self._specs), len(X) if hasattr(X, '__len__') else 1)
        )

        return LazyResult(self._specs, new_node)

    def forward(
        self,
        inputs: Any,
        **kwargs
    ) -> 'LazyResult':
        """
        Queue forward pass for PyTorch models.

        Args:
            inputs: Input tensor(s)
            **kwargs: Additional forward params

        Returns:
            LazyResult with outputs
        """
        new_node = GraphNode(
            op_type=OpType.FORWARD,
            params={"inputs": inputs, **kwargs},
            parents=[self._graph],
            shape=(len(self._specs),)
        )

        return LazyResult(self._specs, new_node)

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        **kwargs
    ) -> 'NPCArray':
        """
        Queue fitting for all models.

        For LLMs, this means fine-tuning.
        For sklearn/torch, this means training.

        Args:
            X: Training features
            y: Training targets (optional for unsupervised)
            **kwargs: Additional fit params (epochs, method, etc.)

        Returns:
            New NPCArray with fitted model specs
        """
        new_node = GraphNode(
            op_type=OpType.FIT,
            params={"X": X, "y": y, **kwargs},
            parents=[self._graph],
            shape=self.shape
        )

        return NPCArray(self._specs, new_node)

    def evolve(
        self,
        fitness_scores: List[float],
        mutate_fn: Optional[Callable] = None,
        crossover_fn: Optional[Callable] = None,
        selection: str = "tournament",
        elite_ratio: float = 0.1
    ) -> 'NPCArray':
        """
        Evolve the model population based on fitness scores.

        Args:
            fitness_scores: Fitness score for each model
            mutate_fn: Custom mutation function
            crossover_fn: Custom crossover function
            selection: Selection strategy ('tournament', 'roulette', 'rank')
            elite_ratio: Fraction of top performers to keep unchanged

        Returns:
            New NPCArray with evolved population
        """
        new_node = GraphNode(
            op_type=OpType.EVOLVE,
            params={
                "fitness_scores": fitness_scores,
                "mutate_fn": mutate_fn,
                "crossover_fn": crossover_fn,
                "selection": selection,
                "elite_ratio": elite_ratio
            },
            parents=[self._graph],
            shape=self.shape
        )

        return NPCArray(self._specs, new_node)

    def jinx(
        self,
        jinx_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'LazyResult':
        """
        Execute a Jinx workflow across all models in the array.

        Each model in the array will be used as the 'npc' context for the jinx,
        allowing you to run the same workflow template with different models.

        Args:
            jinx_name: Name of the jinx workflow to execute (e.g., 'analyze', 'summarize')
            inputs: Input values for the jinx template variables
            **kwargs: Additional execution parameters

        Returns:
            LazyResult with workflow outputs from each model

        Example:
            >>> models = NPCArray.from_llms(['gpt-4', 'claude-3'])
            >>> results = models.jinx('analyze', inputs={'topic': 'AI safety'}).collect()
        """
        new_node = GraphNode(
            op_type=OpType.JINX,
            params={
                "jinx_name": jinx_name,
                "inputs": inputs or {},
                **kwargs
            },
            parents=[self._graph],
            shape=(len(self._specs),)
        )

        return LazyResult(self._specs, new_node)

class LazyResult:
    """
    Lazy result from model operations.

    Builds computation graph without executing until .compute() is called.
    Supports chaining operations like map, filter, reduce.
    """

    def __init__(
        self,
        specs: List[ModelSpec],
        graph: GraphNode,
        prompts: Optional[List[str]] = None
    ):
        self._specs = specs
        self._graph = graph
        self._prompts = prompts
        self._computed = False
        self._result: Optional[ResponseTensor] = None

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Expected shape of result"""
        return self._graph.shape

    def map(self, fn: Callable[[Any], Any]) -> 'LazyResult':
        """
        Apply function to each response.

        Args:
            fn: Function to apply to each response

        Example:
            >>> result.map(lambda r: len(r))
            >>> result.map(json.loads)
        """
        new_node = GraphNode(
            op_type=OpType.MAP,
            params={"fn": fn},
            parents=[self._graph],
            shape=self._graph.shape
        )

        return LazyResult(self._specs, new_node, self._prompts)

    def filter(self, predicate: Callable[[Any], bool]) -> 'LazyResult':
        """
        Filter responses by predicate.

        Args:
            predicate: Function returning True for responses to keep

        Example:
            >>> result.filter(lambda r: len(r) > 100)
            >>> result.filter(lambda r: 'error' not in r.lower())
        """
        new_node = GraphNode(
            op_type=OpType.FILTER,
            params={"predicate": predicate},
            parents=[self._graph],
            shape=None
        )

        return LazyResult(self._specs, new_node, self._prompts)

    def reduce(
        self,
        method: Union[str, Callable] = "vote",
        axis: int = 0,
        **kwargs
    ) -> 'LazyResult':
        """
        Reduce responses along an axis.

        Args:
            method: Reduction method or custom function
                - 'vote': Majority voting
                - 'mean': Average (for numeric)
                - 'concat': Concatenate strings
                - 'consensus': LLM-based consensus
                - 'best': Select by score
                - callable: Custom reduction
            axis: Axis to reduce (0=models, 1=prompts)
            **kwargs: Additional params for reduction

        Example:
            >>> result.reduce('vote', axis=0)
            >>> result.reduce('mean', axis=1)
        """
        new_node = GraphNode(
            op_type=OpType.REDUCE,
            params={"method": method, "axis": axis, **kwargs},
            parents=[self._graph],
            shape=self._compute_reduced_shape(axis)
        )

        return LazyResult(self._specs, new_node, self._prompts)

    def _compute_reduced_shape(self, axis: int) -> Optional[Tuple[int, ...]]:
        """Compute shape after reduction"""
        if self._graph.shape is None:
            return None
        shape = list(self._graph.shape)
        if axis < len(shape):
            shape.pop(axis)
        return tuple(shape) if shape else (1,)

    def chain(
        self,
        fn: Callable[[List[Any]], str],
        n_rounds: int = 1
    ) -> 'LazyResult':
        """
        Chain outputs through a synthesis function.

        Useful for debate/discussion patterns where outputs
        feed back as context for next round.

        Args:
            fn: Function taking all responses, returning synthesis prompt
            n_rounds: Number of chain rounds

        Example:
            >>> def debate_round(responses):
            ...     return f"Consider these perspectives: {responses}. Synthesize."
            >>> result.chain(debate_round, n_rounds=3)
        """
        new_node = GraphNode(
            op_type=OpType.CHAIN,
            params={"fn": fn, "n_rounds": n_rounds},
            parents=[self._graph],
            shape=self._graph.shape
        )

        return LazyResult(self._specs, new_node, self._prompts)

    def vote(self, axis: int = 0) -> 'LazyResult':
        """Shorthand for reduce('vote', axis)"""
        return self.reduce('vote', axis=axis)

    def consensus(self, axis: int = 0, model: str = None) -> 'LazyResult':
        """Shorthand for reduce('consensus', axis)"""
        return self.reduce('consensus', axis=axis, model=model)

    def variance(self) -> 'LazyResult':
        """Compute variance/disagreement across models"""
        return self.map(_compute_response_variance)

    def argmax(self, scores: List[float]) -> 'LazyResult':
        """Select responses corresponding to max scores"""
        return self.reduce('best', scores=scores)

    def explain(self) -> str:
        """
        Print explanation of the computation graph.

        Returns:
            String representation of the DAG
        """
        lines = ["Computation Graph:"]
        self._explain_node(self._graph, lines, depth=0)
        explanation = "\n".join(lines)
        print(explanation)
        return explanation

    def _explain_node(self, node: GraphNode, lines: List[str], depth: int):
        indent = "  " * depth
        params_str = {k: v for k, v in node.params.items() if k not in ('fn', 'predicate')}
        lines.append(f"{indent}└─ {node.op_type.value}: shape={node.shape}, params={params_str}")
        for parent in node.parents:
            self._explain_node(parent, lines, depth + 1)

    def collect(
        self,
        parallel: bool = True,
        max_workers: int = None,
        progress: bool = False
    ) -> ResponseTensor:
        """
        Execute the computation graph and return results.

        Like Spark's collect(), this materializes the lazy computation.

        Args:
            parallel: Whether to parallelize independent operations
            max_workers: Max parallel workers (default: number of models)
            progress: Show progress bar

        Returns:
            ResponseTensor with materialized results
        """
        if self._computed and self._result is not None:
            return self._result

        executor = GraphExecutor(
            parallel=parallel,
            max_workers=max_workers or len(self._specs),
            progress=progress
        )

        self._result = executor.execute(self._graph, self._specs, self._prompts)
        self._computed = True

        return self._result

    def to_list(self) -> List:
        """Collect and return as Python list"""
        return self.collect().tolist()

    compute = collect

class GraphExecutor:
    """
    Executes the lazy computation graph.

    Handles:
    - Topological ordering
    - Parallel execution of independent nodes
    - Caching of intermediate results
    """

    def __init__(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = False
    ):
        self.parallel = parallel
        self.max_workers = max_workers
        self.progress = progress
        self._cache: Dict[int, Any] = {}

    def execute(
        self,
        root: GraphNode,
        specs: List[ModelSpec],
        prompts: Optional[List[str]] = None
    ) -> ResponseTensor:
        """Execute graph starting from root node"""

        ordered = self._topological_sort(root)

        for node in ordered:
            if id(node) in self._cache:
                continue

            parent_results = [self._cache.get(id(p)) for p in node.parents]

            result = self._execute_node(node, specs, prompts, parent_results)
            self._cache[id(node)] = result

        return self._cache[id(root)]

    def _topological_sort(self, root: GraphNode) -> List[GraphNode]:
        """Return nodes in execution order (leaves first)"""
        visited = set()
        ordered = []

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            ordered.append(node)

        visit(root)
        return ordered

    def _execute_node(
        self,
        node: GraphNode,
        specs: List[ModelSpec],
        prompts: Optional[List[str]],
        parent_results: List[Any]
    ) -> ResponseTensor:
        """Execute a single graph node"""

        handlers = {
            OpType.SOURCE: self._exec_source,
            OpType.INFER: self._exec_infer,
            OpType.PREDICT: self._exec_predict,
            OpType.FORWARD: self._exec_forward,
            OpType.FIT: self._exec_fit,
            OpType.MAP: self._exec_map,
            OpType.FILTER: self._exec_filter,
            OpType.REDUCE: self._exec_reduce,
            OpType.CHAIN: self._exec_chain,
            OpType.EVOLVE: self._exec_evolve,
            OpType.JINX: self._exec_jinx,
        }

        handler = handlers.get(node.op_type)
        if handler is None:
            raise ValueError(f"Unknown operation type: {node.op_type}")

        return handler(node, specs, prompts, parent_results)

    def _exec_source(self, node, specs, prompts, parents) -> ResponseTensor:
        """Source node - just returns specs wrapped"""
        return ResponseTensor(
            data=np.array([s.model_ref for s in specs], dtype=object),
            model_specs=specs,
            prompts=prompts
        )

    def _exec_infer(self, node, specs, prompts, parents) -> ResponseTensor:
        """Execute LLM inference across models and prompts"""
        from npcpy.llm_funcs import get_llm_response

        prompts_list = node.params.get("prompts", prompts or [])
        extra_kwargs = {k: v for k, v in node.params.items() if k != "prompts"}

        n_models = len(specs)
        n_prompts = len(prompts_list)

        tasks = []
        for i, spec in enumerate(specs):
            for j, prompt in enumerate(prompts_list):
                tasks.append((i, j, spec, prompt))

        results = np.empty((n_models, n_prompts), dtype=object)

        if self.parallel and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for i, j, spec, prompt in tasks:
                    future = executor.submit(
                        self._infer_single, spec, prompt, extra_kwargs
                    )
                    futures[future] = (i, j)

                for future in as_completed(futures):
                    i, j = futures[future]
                    try:
                        results[i, j] = future.result()
                    except Exception as e:
                        results[i, j] = f"Error: {e}"
        else:
            for i, j, spec, prompt in tasks:
                try:
                    results[i, j] = self._infer_single(spec, prompt, extra_kwargs)
                except Exception as e:
                    results[i, j] = f"Error: {e}"

        return ResponseTensor(
            data=results,
            model_specs=specs,
            prompts=prompts_list,
            metadata={"operation": "infer", **extra_kwargs}
        )

    def _infer_single(self, spec: ModelSpec, prompt: str, kwargs: Dict) -> str:
        """Single model inference"""
        from npcpy.llm_funcs import get_llm_response

        if spec.model_type == "llm":
            response = get_llm_response(
                prompt,
                model=spec.model_ref,
                provider=spec.provider,
                **{**spec.config, **kwargs}
            )
            return response.get("response", "")

        elif spec.model_type == "npc":
            npc = spec.model_ref
            response = get_llm_response(
                prompt,
                npc=npc,
                **kwargs
            )
            return response.get("response", "")

        else:
            raise ValueError(f"Cannot infer with model type: {spec.model_type}")

    def _exec_predict(self, node, specs, prompts, parents) -> ResponseTensor:
        """Execute sklearn/ML prediction"""
        X = node.params.get("X")

        results = []
        for spec in specs:
            if spec.model_type == "sklearn":
                model = spec.model_ref
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    results.append(pred)
                else:
                    results.append(None)
            else:
                results.append(None)

        return ResponseTensor(
            data=np.array(results, dtype=object),
            model_specs=specs,
            metadata={"operation": "predict"}
        )

    def _exec_forward(self, node, specs, prompts, parents) -> ResponseTensor:
        """Execute PyTorch forward pass"""
        inputs = node.params.get("inputs")

        results = []
        for spec in specs:
            if spec.model_type == "torch":
                model = spec.model_ref
                device = spec.config.get("device", "cpu")
                try:
                    import torch
                    model.to(device)
                    model.eval()
                    with torch.no_grad():
                        output = model(inputs.to(device) if hasattr(inputs, 'to') else inputs)
                    results.append(output)
                except Exception as e:
                    results.append(f"Error: {e}")
            else:
                results.append(None)

        return ResponseTensor(
            data=np.array(results, dtype=object),
            model_specs=specs,
            metadata={"operation": "forward"}
        )

    def _exec_fit(self, node, specs, prompts, parents) -> ResponseTensor:
        """Execute model fitting"""
        X = node.params.get("X")
        y = node.params.get("y")

        fitted_specs = []
        for spec in specs:
            if spec.model_type == "sklearn":
                model = copy.deepcopy(spec.model_ref)
                if hasattr(model, 'fit'):
                    model.fit(X, y)
                new_spec = ModelSpec(
                    model_type="sklearn",
                    model_ref=model,
                    config={**spec.config, "fitted": True}
                )
                fitted_specs.append(new_spec)
            elif spec.model_type == "llm":
                fitted_specs.append(spec)
            else:
                fitted_specs.append(spec)

        return ResponseTensor(
            data=np.array([s.model_ref for s in fitted_specs], dtype=object),
            model_specs=fitted_specs,
            metadata={"operation": "fit"}
        )

    def _exec_map(self, node, specs, prompts, parents) -> ResponseTensor:
        """Apply function to each result"""
        fn = node.params.get("fn")
        parent_result = parents[0] if parents else None

        if parent_result is None:
            raise ValueError("Map requires parent result")

        mapped = np.vectorize(fn, otypes=[object])(parent_result.data)

        return ResponseTensor(
            data=mapped,
            model_specs=parent_result.model_specs,
            prompts=parent_result.prompts,
            metadata={**parent_result.metadata, "mapped": True}
        )

    def _exec_filter(self, node, specs, prompts, parents) -> ResponseTensor:
        """Filter results by predicate"""
        predicate = node.params.get("predicate")
        parent_result = parents[0] if parents else None

        if parent_result is None:
            raise ValueError("Filter requires parent result")

        mask = np.vectorize(predicate)(parent_result.data)
        filtered_data = parent_result.data[mask]

        return ResponseTensor(
            data=filtered_data,
            model_specs=parent_result.model_specs,
            prompts=parent_result.prompts,
            metadata={**parent_result.metadata, "filtered": True}
        )

    def _exec_reduce(self, node, specs, prompts, parents) -> ResponseTensor:
        """Reduce results along axis"""
        method = node.params.get("method", "vote")
        axis = node.params.get("axis", 0)
        parent_result = parents[0] if parents else None

        if parent_result is None:
            raise ValueError("Reduce requires parent result")

        data = parent_result.data

        if method == "vote":
            reduced = self._reduce_vote(data, axis)
        elif method == "mean":
            reduced = np.mean(data, axis=axis)
        elif method == "concat":
            reduced = self._reduce_concat(data, axis)
        elif method == "consensus":
            reduced = self._reduce_consensus(data, axis, node.params)
        elif method == "best":
            scores = node.params.get("scores", [])
            reduced = self._reduce_best(data, scores, axis)
        elif callable(method):
            reduced = np.apply_along_axis(method, axis, data)
        else:
            raise ValueError(f"Unknown reduce method: {method}")

        return ResponseTensor(
            data=np.atleast_1d(reduced),
            model_specs=specs if axis != 0 else [specs[0]],
            prompts=prompts,
            metadata={**parent_result.metadata, "reduced": method}
        )

    def _reduce_vote(self, data: np.ndarray, axis: int) -> np.ndarray:
        """Majority voting reduction"""
        from collections import Counter

        def vote_fn(arr):
            counter = Counter(arr)
            return counter.most_common(1)[0][0] if counter else None

        return np.apply_along_axis(vote_fn, axis, data)

    def _reduce_concat(self, data: np.ndarray, axis: int) -> np.ndarray:
        """Concatenate strings"""
        def concat_fn(arr):
            return "\n---\n".join(str(x) for x in arr)

        return np.apply_along_axis(concat_fn, axis, data)

    def _reduce_consensus(self, data: np.ndarray, axis: int, params: Dict) -> np.ndarray:
        """LLM-based consensus"""
        from npcpy.llm_funcs import get_llm_response

        model = params.get("model")
        if not model:
            raise ValueError("No model specified for consensus reduction.")

        def consensus_fn(arr):
            perspectives = "\n".join(f"- {x}" for x in arr)
            prompt = f"Given these different perspectives:\n{perspectives}\n\nProvide a consensus synthesis:"
            response = get_llm_response(prompt, model=model)
            return response.get("response", "")

        return np.apply_along_axis(consensus_fn, axis, data)

    def _reduce_best(self, data: np.ndarray, scores: List[float], axis: int) -> np.ndarray:
        """Select best by score"""
        if axis == 0:
            best_idx = np.argmax(scores)
            return data[best_idx]
        else:
            return data

    def _exec_chain(self, node, specs, prompts, parents) -> ResponseTensor:
        """Chain responses through synthesis function"""
        fn = node.params.get("fn")
        n_rounds = node.params.get("n_rounds", 1)
        parent_result = parents[0] if parents else None

        if parent_result is None:
            raise ValueError("Chain requires parent result")

        current = parent_result.data

        for _ in range(n_rounds):
            new_prompt = fn(current.tolist())

            infer_node = GraphNode(
                op_type=OpType.INFER,
                params={"prompts": [new_prompt]},
                shape=(len(specs), 1)
            )
            current = self._exec_infer(infer_node, specs, [new_prompt], []).data

        return ResponseTensor(
            data=current,
            model_specs=specs,
            prompts=prompts,
            metadata={**parent_result.metadata, "chained": n_rounds}
        )

    def _exec_evolve(self, node, specs, prompts, parents) -> ResponseTensor:
        """Evolve population based on fitness"""
        import random

        fitness_scores = node.params.get("fitness_scores", [])
        mutate_fn = node.params.get("mutate_fn")
        crossover_fn = node.params.get("crossover_fn")
        elite_ratio = node.params.get("elite_ratio", 0.1)

        n = len(specs)
        n_elite = max(1, int(n * elite_ratio))

        sorted_indices = np.argsort(fitness_scores)[::-1]

        new_specs = [specs[i] for i in sorted_indices[:n_elite]]

        while len(new_specs) < n:
            if crossover_fn and random.random() < 0.5:
                parent1 = specs[random.choice(sorted_indices[:n//2])]
                parent2 = specs[random.choice(sorted_indices[:n//2])]
                child_spec = crossover_fn(parent1, parent2)
            else:
                parent = specs[random.choice(sorted_indices[:n//2])]
                child_spec = mutate_fn(parent) if mutate_fn else parent
            new_specs.append(child_spec)

        return ResponseTensor(
            data=np.array([s.model_ref for s in new_specs], dtype=object),
            model_specs=new_specs,
            metadata={"operation": "evolve", "generation": 1}
        )

    def _exec_jinx(self, node, specs, prompts, parents) -> ResponseTensor:
        """Execute a Jinx workflow across models"""
        from npcpy.npc_compiler import NPC, Jinx

        jinx_name = node.params.get("jinx_name")
        inputs = node.params.get("inputs", {})
        extra_kwargs = {k: v for k, v in node.params.items()
                       if k not in ("jinx_name", "inputs")}

        results = []

        def run_jinx_single(spec: ModelSpec) -> str:
            """Run jinx for a single model spec"""
            try:
                if spec.model_type == "npc":
                    npc = spec.model_ref
                else:
                    npc = NPC(
                        name=f"array_npc_{spec.model_ref}",
                        model=spec.model_ref,
                        provider=spec.provider
                    )

                result = npc.execute_jinx(
                    jinx_name=jinx_name,
                    inputs=inputs,
                    **extra_kwargs
                )
                return result.get("output", str(result))
            except Exception as e:
                return f"Error: {e}"

        if self.parallel and len(specs) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(run_jinx_single, spec): i
                          for i, spec in enumerate(specs)}
                results = [None] * len(specs)
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = f"Error: {e}"
        else:
            results = [run_jinx_single(spec) for spec in specs]

        return ResponseTensor(
            data=np.array(results, dtype=object),
            model_specs=specs,
            metadata={"operation": "jinx", "jinx_name": jinx_name, **inputs}
        )

def _compute_response_variance(responses: List[str]) -> float:
    """Compute semantic variance across responses"""
    if not responses:
        return 0.0

    lengths = [len(r) for r in responses]
    all_words = set()
    word_sets = []
    for r in responses:
        words = set(str(r).lower().split())
        word_sets.append(words)
        all_words.update(words)

    if len(word_sets) < 2:
        return 0.0

    total_overlap = 0
    n_pairs = 0
    for i, ws1 in enumerate(word_sets):
        for ws2 in word_sets[i+1:]:
            if ws1 or ws2:
                overlap = len(ws1 & ws2) / len(ws1 | ws2) if (ws1 | ws2) else 1.0
                total_overlap += overlap
                n_pairs += 1

    avg_overlap = total_overlap / n_pairs if n_pairs > 0 else 1.0
    return 1.0 - avg_overlap

def npc_udf(
    operation: str,
    model_array: NPCArray,
    input_col: 'pl.Expr' = None,
    **kwargs
) -> 'pl.Expr':
    """
    Create a Polars user-defined function for NPC operations.

    Args:
        operation: 'infer', 'predict', 'forward', 'fit'
        model_array: NPCArray to use
        input_col: Polars column expression for input
        **kwargs: Additional operation params

    Example:
        >>> result = df.with_columns(
        ...     npc_udf('infer', models, pl.col('text')).alias('response')
        ... )
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError("Polars required for npc_udf. Install with: pip install polars")

    def apply_fn(inputs: pl.Series) -> pl.Series:
        input_list = inputs.to_list()

        if operation == "infer":
            result = model_array.infer(input_list, **kwargs).compute()
        elif operation == "predict":
            result = model_array.predict(input_list, **kwargs).compute()
        elif operation == "forward":
            result = model_array.forward(input_list, **kwargs).compute()
        else:
            raise ValueError(f"Unknown operation: {operation}")

        output = result.flatten() if result.shape[0] == 1 else result.data[:, 0].tolist()
        return pl.Series(output)

    return input_col.map_elements(apply_fn, return_dtype=pl.Utf8)

def register_polars_namespace():
    """
    Register 'npc' namespace on Polars DataFrames.

    After calling this, you can do:
        >>> df.npc.infer(models, 'text_col')
    """
    try:
        import polars as pl

        @pl.api.register_dataframe_namespace("npc")
        class NPCNamespace:
            def __init__(self, df: pl.DataFrame):
                self._df = df

            def infer(
                self,
                models: NPCArray,
                input_col: str,
                output_col: str = "response",
                **kwargs
            ) -> pl.DataFrame:
                return self._df.with_columns(
                    npc_udf('infer', models, pl.col(input_col), **kwargs)
                    .alias(output_col)
                )

        return True
    except ImportError:
        return False

def infer_matrix(
    prompts: List[str],
    models: List[str] = None,
    providers: List[str] = None,
    **kwargs
) -> ResponseTensor:
    """
    Quick inference across model/prompt matrix.

    Args:
        prompts: List of prompts
        models: List of model names
        providers: List of providers
        **kwargs: Additional params

    Returns:
        ResponseTensor of shape (n_models, n_prompts)
    """
    if models is None:
        raise ValueError("No models specified for ensemble. Pass models=[...].")

    arr = NPCArray.from_llms(models, providers)
    return arr.infer(prompts, **kwargs).compute()

def ensemble_vote(
    prompt: str,
    models: List[str],
    providers: List[str] = None
) -> str:
    """
    Quick ensemble voting across models.

    Args:
        prompt: Single prompt
        models: List of models to query
        providers: Optional providers

    Returns:
        Consensus response string
    """
    arr = NPCArray.from_llms(models, providers)
    result = arr.infer(prompt).vote(axis=0).compute()
    return result.data[0] if result.data.size > 0 else ""
