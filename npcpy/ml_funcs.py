"""
ml_funcs.py - NumPy-like interface for ML model operations

Parallels llm_funcs but for traditional ML:
- sklearn models
- PyTorch models
- Time series models
- Ensemble operations

Same interface pattern as llm_funcs:
- Single call does single operation
- matrix parameter enables cartesian product
- n_samples enables multiple samples
"""

from __future__ import annotations
import copy
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

_sklearn_available = False
_torch_available = False
_xgboost_available = False
_statsmodels_available = False

try:
    import sklearn
    from sklearn.base import clone, BaseEstimator
    _sklearn_available = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    _torch_available = True
except ImportError:
    pass

try:
    import xgboost as xgb
    _xgboost_available = True
except ImportError:
    pass

try:
    import statsmodels.api as sm
    _statsmodels_available = True
except ImportError:
    pass

SKLEARN_MODELS = {
    'LogisticRegression': 'sklearn.linear_model.LogisticRegression',
    'RandomForestClassifier': 'sklearn.ensemble.RandomForestClassifier',
    'GradientBoostingClassifier': 'sklearn.ensemble.GradientBoostingClassifier',
    'SVC': 'sklearn.svm.SVC',
    'KNeighborsClassifier': 'sklearn.neighbors.KNeighborsClassifier',
    'DecisionTreeClassifier': 'sklearn.tree.DecisionTreeClassifier',
    'AdaBoostClassifier': 'sklearn.ensemble.AdaBoostClassifier',
    'GaussianNB': 'sklearn.naive_bayes.GaussianNB',
    'MLPClassifier': 'sklearn.neural_network.MLPClassifier',

    'LinearRegression': 'sklearn.linear_model.LinearRegression',
    'Ridge': 'sklearn.linear_model.Ridge',
    'Lasso': 'sklearn.linear_model.Lasso',
    'ElasticNet': 'sklearn.linear_model.ElasticNet',
    'RandomForestRegressor': 'sklearn.ensemble.RandomForestRegressor',
    'GradientBoostingRegressor': 'sklearn.ensemble.GradientBoostingRegressor',
    'SVR': 'sklearn.svm.SVR',
    'KNeighborsRegressor': 'sklearn.neighbors.KNeighborsRegressor',
    'DecisionTreeRegressor': 'sklearn.tree.DecisionTreeRegressor',
    'MLPRegressor': 'sklearn.neural_network.MLPRegressor',

    'KMeans': 'sklearn.cluster.KMeans',
    'DBSCAN': 'sklearn.cluster.DBSCAN',
    'AgglomerativeClustering': 'sklearn.cluster.AgglomerativeClustering',

    'PCA': 'sklearn.decomposition.PCA',
    'TSNE': 'sklearn.manifold.TSNE',
    'UMAP': 'umap.UMAP',
}

def _import_model_class(model_path: str):
    """Dynamically import a model class from path"""
    parts = model_path.rsplit('.', 1)
    if len(parts) == 2:
        module_path, class_name = parts
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise ValueError(f"Invalid model path: {model_path}")

def _get_model_instance(model_name: str, **kwargs):
    """Get model instance from name"""
    if model_name in SKLEARN_MODELS:
        model_class = _import_model_class(SKLEARN_MODELS[model_name])
        return model_class(**kwargs)
    elif _xgboost_available and model_name.lower().startswith('xgb'):
        if 'classifier' in model_name.lower():
            return xgb.XGBClassifier(**kwargs)
        else:
            return xgb.XGBRegressor(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def fit_model(
    X: Any,
    y: Any = None,
    model: Union[str, Any] = "RandomForestClassifier",
    n_samples: int = 1,
    matrix: Optional[Dict[str, List[Any]]] = None,
    parallel: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit ML model(s) to data.

    Similar interface to get_llm_response but for model fitting.

    Args:
        X: Training features
        y: Training targets (optional for unsupervised)
        model: Model name, class, or instance
        n_samples: Number of models to fit (with different random seeds)
        matrix: Dict of param -> list for grid search
        parallel: Whether to parallelize fitting
        **kwargs: Model hyperparameters

    Returns:
        Dict with:
            - 'model': Fitted model(s)
            - 'models': List of all fitted models (if multiple)
            - 'scores': Training scores if available
    """
    if not _sklearn_available:
        raise ImportError("sklearn required. Install with: pip install scikit-learn")

    def _fit_single(model_instance, X, y, seed=None):
        if seed is not None and hasattr(model_instance, 'random_state'):
            model_instance.random_state = seed
        model_instance.fit(X, y)
        score = None
        if hasattr(model_instance, 'score') and y is not None:
            try:
                score = model_instance.score(X, y)
            except Exception:
                pass
        return {'model': model_instance, 'score': score}

    use_matrix = matrix is not None and len(matrix) > 0
    multi_sample = n_samples and n_samples > 1

    if not use_matrix and not multi_sample:
        if isinstance(model, str):
            model_instance = _get_model_instance(model, **kwargs)
        elif hasattr(model, 'fit'):
            model_instance = clone(model) if _sklearn_available else copy.deepcopy(model)
        else:
            raise ValueError(f"Invalid model: {model}")

        result = _fit_single(model_instance, X, y)
        return {
            'model': result['model'],
            'models': [result['model']],
            'scores': [result['score']] if result['score'] is not None else None
        }

    combos = []
    if use_matrix:
        keys = list(matrix.keys())
        values = [matrix[k] if isinstance(matrix[k], list) else [matrix[k]] for k in keys]
        for combo_values in itertools.product(*values):
            combo = dict(zip(keys, combo_values))
            combos.append(combo)
    else:
        combos = [{}]

    all_tasks = []
    for combo in combos:
        for sample_idx in range(max(1, n_samples)):
            all_tasks.append((combo, sample_idx))

    results = []
    if parallel and len(all_tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(all_tasks))) as executor:
            futures = {}
            for combo, sample_idx in all_tasks:
                merged_kwargs = {**kwargs, **combo}
                if isinstance(model, str):
                    model_instance = _get_model_instance(model, **merged_kwargs)
                else:
                    model_instance = clone(model)
                    for k, v in merged_kwargs.items():
                        if hasattr(model_instance, k):
                            setattr(model_instance, k, v)

                future = executor.submit(_fit_single, model_instance, X, y, sample_idx)
                futures[future] = (combo, sample_idx)

            for future in as_completed(futures):
                combo, sample_idx = futures[future]
                try:
                    result = future.result()
                    result['params'] = combo
                    result['sample_index'] = sample_idx
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e), 'params': combo, 'sample_index': sample_idx})
    else:
        for combo, sample_idx in all_tasks:
            merged_kwargs = {**kwargs, **combo}
            if isinstance(model, str):
                model_instance = _get_model_instance(model, **merged_kwargs)
            else:
                model_instance = clone(model)
                for k, v in merged_kwargs.items():
                    if hasattr(model_instance, k):
                        setattr(model_instance, k, v)

            try:
                result = _fit_single(model_instance, X, y, sample_idx)
                result['params'] = combo
                result['sample_index'] = sample_idx
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'params': combo, 'sample_index': sample_idx})

    models = [r['model'] for r in results if 'model' in r]
    scores = [r['score'] for r in results if 'score' in r and r['score'] is not None]

    return {
        'model': models[0] if models else None,
        'models': models,
        'scores': scores if scores else None,
        'results': results
    }

def predict_model(
    X: Any,
    model: Any,
    n_samples: int = 1,
    matrix: Optional[Dict[str, List[Any]]] = None,
    parallel: bool = True,
    method: str = "predict",
    **kwargs
) -> Dict[str, Any]:
    """
    Make predictions with ML model(s).

    Args:
        X: Input features
        model: Fitted model or list of models
        n_samples: Number of prediction samples (for probabilistic models)
        matrix: Not typically used for prediction
        parallel: Whether to parallelize
        method: 'predict', 'predict_proba', 'transform'
        **kwargs: Additional prediction params

    Returns:
        Dict with:
            - 'predictions': Predictions from first/main model
            - 'all_predictions': All predictions (if multiple models)
    """
    models = model if isinstance(model, list) else [model]

    def _predict_single(m, method_name):
        if hasattr(m, method_name):
            pred_fn = getattr(m, method_name)
            return pred_fn(X, **kwargs)
        elif method_name == "predict_proba" and hasattr(m, "predict"):
            return m.predict(X, **kwargs)
        else:
            raise ValueError(f"Model has no {method_name} method")

    results = []
    if parallel and len(models) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(models))) as executor:
            futures = {executor.submit(_predict_single, m, method): i for i, m in enumerate(models)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    pred = future.result()
                    results.append((idx, pred))
                except Exception as e:
                    results.append((idx, f"Error: {e}"))

        results.sort(key=lambda x: x[0])
        predictions = [r[1] for r in results]
    else:
        predictions = [_predict_single(m, method) for m in models]

    return {
        'predictions': predictions[0] if predictions else None,
        'all_predictions': predictions
    }

def score_model(
    X: Any,
    y: Any,
    model: Any,
    metrics: List[str] = None,
    parallel: bool = True
) -> Dict[str, Any]:
    """
    Score model(s) on test data.

    Args:
        X: Test features
        y: Test targets
        model: Fitted model or list of models
        metrics: List of metric names ('accuracy', 'f1', 'mse', 'r2', etc.)
        parallel: Whether to parallelize

    Returns:
        Dict with scores for each metric
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        mean_squared_error, mean_absolute_error, r2_score
    )

    metric_funcs = {
        'accuracy': accuracy_score,
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'r2': r2_score,
    }

    if metrics is None:
        metrics = ['accuracy']

    models = model if isinstance(model, list) else [model]

    all_scores = []
    for m in models:
        preds = m.predict(X)
        model_scores = {}
        for metric_name in metrics:
            if metric_name in metric_funcs:
                try:
                    model_scores[metric_name] = metric_funcs[metric_name](y, preds)
                except Exception:
                    model_scores[metric_name] = None
        all_scores.append(model_scores)

    return {
        'scores': all_scores[0] if len(all_scores) == 1 else all_scores,
        'all_scores': all_scores
    }

def fit_torch(
    model: Any,
    train_loader: Any,
    epochs: int = 10,
    optimizer: str = "Adam",
    lr: float = 0.001,
    criterion: str = "CrossEntropyLoss",
    device: str = "cpu",
    val_loader: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train PyTorch model.

    Args:
        model: nn.Module instance
        train_loader: DataLoader for training
        epochs: Number of training epochs
        optimizer: Optimizer name
        lr: Learning rate
        criterion: Loss function name
        device: Device to train on
        val_loader: Optional validation DataLoader

    Returns:
        Dict with trained model and training history
    """
    if not _torch_available:
        raise ImportError("PyTorch required. Install with: pip install torch")

    model = model.to(device)

    opt_class = getattr(torch.optim, optimizer)
    opt = opt_class(model.parameters(), lr=lr)

    crit_class = getattr(nn, criterion)
    crit = crit_class()

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                targets = None

            opt.zero_grad()
            outputs = model(inputs)

            if targets is not None:
                loss = crit(outputs, targets)
            else:
                loss = outputs

            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        history['train_loss'].append(epoch_loss / len(train_loader))

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch[0].to(device), batch[1].to(device)
                    else:
                        inputs = batch.to(device)
                        targets = None

                    outputs = model(inputs)
                    if targets is not None:
                        loss = crit(outputs, targets)
                        val_loss += loss.item()

            history['val_loss'].append(val_loss / len(val_loader))

    return {
        'model': model,
        'history': history,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
    }

def forward_torch(
    model: Any,
    inputs: Any,
    device: str = "cpu",
    grad: bool = False
) -> Dict[str, Any]:
    """
    Run forward pass on PyTorch model.

    Args:
        model: nn.Module instance
        inputs: Input tensor or batch
        device: Device to run on
        grad: Whether to compute gradients

    Returns:
        Dict with outputs
    """
    if not _torch_available:
        raise ImportError("PyTorch required. Install with: pip install torch")

    model = model.to(device)
    model.eval()

    if hasattr(inputs, 'to'):
        inputs = inputs.to(device)

    if grad:
        outputs = model(inputs)
    else:
        with torch.no_grad():
            outputs = model(inputs)

    return {
        'outputs': outputs,
        'output_numpy': outputs.cpu().numpy() if hasattr(outputs, 'cpu') else outputs
    }

def fit_timeseries(
    series: Any,
    method: str = "arima",
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit time series model.

    Args:
        series: Time series data (array-like)
        method: 'arima', 'sarima', 'exp_smoothing', 'prophet'
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s) for SARIMA
        **kwargs: Additional model params

    Returns:
        Dict with fitted model and diagnostics
    """
    if method.lower() in ('arima', 'sarima'):
        if not _statsmodels_available:
            raise ImportError("statsmodels required. Install with: pip install statsmodels")

        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if method.lower() == 'sarima' and seasonal_order:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order, **kwargs)
        else:
            model = ARIMA(series, order=order, **kwargs)

        fitted = model.fit()

        return {
            'model': fitted,
            'aic': fitted.aic,
            'bic': fitted.bic,
            'summary': str(fitted.summary())
        }

    elif method.lower() == 'exp_smoothing':
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(series, **kwargs)
        fitted = model.fit()

        return {
            'model': fitted,
            'aic': fitted.aic,
            'sse': fitted.sse
        }

    else:
        raise ValueError(f"Unknown time series method: {method}")

def forecast_timeseries(
    model: Any,
    horizon: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate forecasts from fitted time series model.

    Args:
        model: Fitted time series model
        horizon: Number of periods to forecast
        **kwargs: Additional forecast params

    Returns:
        Dict with forecasts and confidence intervals
    """
    if hasattr(model, 'forecast'):
        forecast = model.forecast(steps=horizon, **kwargs)
    elif hasattr(model, 'predict'):
        forecast = model.predict(start=len(model.data.endog), end=len(model.data.endog) + horizon - 1)
    else:
        raise ValueError("Model has no forecast or predict method")

    result = {'forecast': forecast}

    if hasattr(model, 'get_forecast'):
        fc = model.get_forecast(steps=horizon)
        result['conf_int'] = fc.conf_int()
        result['forecast_mean'] = fc.predicted_mean

    return result

def ensemble_predict(
    X: Any,
    models: List[Any],
    method: str = "vote",
    weights: List[float] = None
) -> Dict[str, Any]:
    """
    Ensemble predictions from multiple models.

    Args:
        X: Input features
        models: List of fitted models
        method: 'vote', 'average', 'weighted', 'stack'
        weights: Model weights for weighted averaging

    Returns:
        Dict with ensemble predictions
    """
    all_preds = []
    for m in models:
        pred = m.predict(X)
        all_preds.append(pred)

    all_preds = np.array(all_preds)

    if method == "vote":
        from scipy import stats
        ensemble_pred, _ = stats.mode(all_preds, axis=0)
        ensemble_pred = ensemble_pred.flatten()

    elif method == "average":
        ensemble_pred = np.mean(all_preds, axis=0)

    elif method == "weighted":
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        weights = np.array(weights).reshape(-1, 1)
        ensemble_pred = np.sum(all_preds * weights, axis=0)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return {
        'predictions': ensemble_pred,
        'individual_predictions': all_preds,
        'method': method
    }

def cross_validate(
    X: Any,
    y: Any,
    model: Union[str, Any],
    cv: int = 5,
    metrics: List[str] = None,
    parallel: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Cross-validate model.

    Args:
        X: Features
        y: Targets
        model: Model name or instance
        cv: Number of folds
        metrics: Metrics to compute
        parallel: Parallelize folds
        **kwargs: Model hyperparameters

    Returns:
        Dict with CV scores
    """
    from sklearn.model_selection import cross_val_score, KFold

    if isinstance(model, str):
        model_instance = _get_model_instance(model, **kwargs)
    else:
        model_instance = model

    if metrics is None:
        metrics = ['accuracy']

    results = {}
    for metric in metrics:
        scoring = metric if metric in ['accuracy', 'f1', 'precision', 'recall', 'r2', 'neg_mean_squared_error'] else None
        if scoring:
            scores = cross_val_score(model_instance, X, y, cv=cv, scoring=scoring)
            results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }

    return results

def serialize_model(model: Any, path: str, format: str = "joblib") -> None:
    """
    Serialize model to file.

    Args:
        model: The model to serialize
        path: File path to write to (required)
        format: Serialization format - "joblib" (default, pickle-based) or
            "safetensors" (PyTorch state_dict only)

    Raises:
        ImportError: If required library is not installed
        ValueError: If format is not supported for the model type

    Note:
        "joblib" uses joblib.dump, which pickles the model. Load a ".joblib"
        file back with deserialize_model(..., trust_source=True) only from a
        source you trust; see deserialize_model for why.
    """
    if format == "safetensors":
        from safetensors.torch import save_file
        if hasattr(model, 'state_dict'):
            save_file(model.state_dict(), path)
        else:
            raise ValueError("safetensors format requires model with state_dict (PyTorch)")
    elif format == "joblib":
        import joblib
        joblib.dump(model, path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'joblib' or 'safetensors'.")

def deserialize_model(path: str, format: str = "auto", trust_source: bool = False) -> Any:
    """
    Deserialize model from file.

    Args:
        path: File path to load from
        format: "auto" (detect from extension), "joblib", or "safetensors"
        trust_source: Must be True to load a "joblib" file. joblib.load is a
            thin wrapper over pickle.load and can execute arbitrary Python
            code embedded in the file (per joblib's own docs: "joblib.load
            relies on the pickle module and can therefore execute arbitrary
            Python code. It should therefore never be used to load files
            from untrusted sources."). Only pass trust_source=True for a file
            you produced yourself or otherwise trust. "safetensors" carries
            no such risk and does not need the flag.

    Returns:
        The deserialized model

    Raises:
        ImportError: If required library is not installed
        ValueError: If format cannot be determined, or a "joblib" file is
            loaded without trust_source=True
    """
    if format == "auto":
        if path.endswith('.safetensors'):
            format = "safetensors"
        elif path.endswith('.joblib'):
            format = "joblib"
        else:
            raise ValueError(f"Cannot auto-detect format for {path}. Use .joblib or .safetensors extension, or specify format explicitly.")

    if format == "safetensors":
        from safetensors.torch import load_file
        return load_file(path)
    elif format == "joblib":
        if not trust_source:
            raise ValueError(
                """Refusing to load a 'joblib' file without trust_source=True:
                joblib.load executes arbitrary Python code embedded in the
                file. Pass deserialize_model(path, trust_source=True) only
                after confirming you trust the source, or re-save the model
                with format='safetensors' if it doesn't need trust."""
            )
        import joblib
        return joblib.load(path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'joblib' or 'safetensors'.")

def get_model_params(model: Any) -> Dict[str, Any]:
    """Get model hyperparameters"""
    if hasattr(model, 'get_params'):
        return model.get_params()
    elif hasattr(model, 'state_dict'):
        return {'type': 'torch', 'params': list(model.state_dict().keys())}
    else:
        return {}

def set_model_params(model: Any, params: Dict[str, Any]) -> Any:
    """Set model hyperparameters"""
    if hasattr(model, 'set_params'):
        return model.set_params(**params)
    else:
        for k, v in params.items():
            if hasattr(model, k):
                setattr(model, k, v)
        return model
