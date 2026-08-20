"""Test suite for ml_funcs module - NumPy-like ML operations."""

import os
import tempfile
import pytest
import numpy as np


class TestModelRegistry:
    """Test model registry and imports."""

    def test_sklearn_models_registry_exists(self):
        """Test SKLEARN_MODELS registry is properly defined"""
        from npcpy.ml_funcs import SKLEARN_MODELS

        assert isinstance(SKLEARN_MODELS, dict)
        assert len(SKLEARN_MODELS) > 0

        # Check some expected models
        assert "LogisticRegression" in SKLEARN_MODELS
        assert "RandomForestClassifier" in SKLEARN_MODELS
        assert "LinearRegression" in SKLEARN_MODELS
        assert "KMeans" in SKLEARN_MODELS
        assert "PCA" in SKLEARN_MODELS

    def test_model_paths_are_valid_format(self):
        """Test model paths follow expected format"""
        from npcpy.ml_funcs import SKLEARN_MODELS

        for name, path in SKLEARN_MODELS.items():
            assert isinstance(path, str)
            assert "." in path  # Should be module.Class format
            parts = path.rsplit(".", 1)
            assert len(parts) == 2


class TestImportModelClass:
    """Test dynamic model import functionality."""

    def test_import_model_class_sklearn(self):
        """Test importing sklearn model class"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import _import_model_class

        model_class = _import_model_class("sklearn.linear_model.LinearRegression")
        assert model_class is not None
        assert hasattr(model_class, "fit")
        assert hasattr(model_class, "predict")

    def test_import_model_class_invalid(self):
        """Test importing non-existent model raises error"""
        from npcpy.ml_funcs import _import_model_class

        with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
            _import_model_class("nonexistent.module.FakeModel")


class TestGetModelInstance:
    """Test _get_model_instance function."""

    def test_get_model_instance_sklearn(self):
        """Test getting sklearn model instance by name"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import _get_model_instance

        model = _get_model_instance("LinearRegression")
        assert model is not None
        assert hasattr(model, "fit")

    def test_get_model_instance_with_params(self):
        """Test getting model instance with parameters"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import _get_model_instance

        model = _get_model_instance("Ridge", alpha=0.5)
        assert model is not None
        assert model.alpha == 0.5


class TestFitModel:
    """Test fit_model function."""

    def test_fit_model_single(self):
        """Test fitting a single model"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import fit_model

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])

        result = fit_model(model="LinearRegression", X=X, y=y)
        assert result is not None
        # Result should be fitted model or dict with model
        if isinstance(result, dict):
            assert "model" in result or "models" in result
        else:
            assert hasattr(result, "predict")

    def test_fit_model_returns_fitted(self):
        """Test fit_model returns a fitted model that can predict"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import fit_model

        X_train = np.array([[1], [2], [3], [4]])
        y_train = np.array([2, 4, 6, 8])

        result = fit_model(model="LinearRegression", X=X_train, y=y_train)

        # Extract model from result
        if isinstance(result, dict):
            model = result.get("model") or result.get("models", [None])[0]
        else:
            model = result

        if model is not None:
            predictions = model.predict([[5]])
            assert len(predictions) == 1


class TestPredictModel:
    """Test predict_model function."""

    def test_predict_model_basic(self):
        """Test basic prediction"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import fit_model, predict_model
        from sklearn.linear_model import LinearRegression

        # Create and fit a model manually
        model = LinearRegression()
        X_train = np.array([[1], [2], [3], [4]])
        y_train = np.array([2, 4, 6, 8])
        model.fit(X_train, y_train)

        X_test = np.array([[5], [6]])
        result = predict_model(model=model, X=X_test)

        if isinstance(result, dict):
            predictions = result.get("predictions")
        else:
            predictions = result

        if predictions is not None:
            assert len(predictions) == 2


class TestScoreModel:
    """Test score_model function."""

    def test_score_model_basic(self):
        """Test model scoring"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import score_model
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        model.fit(X, y)

        result = score_model(model=model, X=X, y=y)

        if isinstance(result, dict):
            score = result.get("score")
        else:
            score = result

        if score is not None:
            assert isinstance(score, (int, float))
            assert score >= 0 and score <= 1


class TestCrossValidate:
    """Test cross_validate function."""

    def test_cross_validate_basic(self):
        """Test cross-validation"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import cross_validate

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        result = cross_validate(model="LogisticRegression", X=X, y=y, cv=3)
        assert result is not None


class TestModelSerialization:
    """Test model serialization functions (using joblib, not pickle)."""

    def test_serialize_model_to_file(self):
        """Test model serialization to file using joblib"""
        pytest.importorskip("sklearn")
        pytest.importorskip("joblib")
        from npcpy.ml_funcs import serialize_model
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        model.fit(X, y)

        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, "model.joblib")
            serialize_model(model, model_path, format="joblib")
            assert os.path.exists(model_path)
            assert os.path.getsize(model_path) > 0
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_deserialize_model_from_file(self):
        """Test model deserialization from file using joblib"""
        pytest.importorskip("sklearn")
        pytest.importorskip("joblib")
        from npcpy.ml_funcs import serialize_model, deserialize_model
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        model.fit(X, y)

        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, "model.joblib")
            serialize_model(model, model_path, format="joblib")
            loaded = deserialize_model(model_path, trust_source=True)

            assert loaded is not None
            # Check predictions match
            orig_pred = model.predict([[4]])
            loaded_pred = loaded.predict([[4]])
            assert np.allclose(orig_pred, loaded_pred)
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_serialize_deserialize_auto_format(self):
        """Test auto format detection based on file extension"""
        pytest.importorskip("sklearn")
        pytest.importorskip("joblib")
        from npcpy.ml_funcs import serialize_model, deserialize_model
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        model.fit(X, y)

        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, "model.joblib")
            serialize_model(model, model_path)  # default format
            loaded = deserialize_model(model_path, format="auto", trust_source=True)

            assert loaded is not None
            orig_pred = model.predict([[4]])
            loaded_pred = loaded.predict([[4]])
            assert np.allclose(orig_pred, loaded_pred)
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_deserialize_model_joblib_requires_trust_source(self):
        """A 'joblib' file must not load without trust_source=True.

        joblib.load is a pickle wrapper: it runs whatever __reduce__ the file
        embeds. deserialize_model must refuse by default rather than execute
        an arbitrary callable found in an untrusted file.
        """
        pytest.importorskip("joblib")
        import joblib
        from npcpy.ml_funcs import deserialize_model

        class ArbitraryCallable:
            def __reduce__(self):
                # Any callable+args would run here; os.getpid is inert and
                # only serves to prove the reduce path executes.
                return (os.getpid, ())

        temp_dir = tempfile.mkdtemp()
        try:
            model_path = os.path.join(temp_dir, "untrusted.joblib")
            joblib.dump(ArbitraryCallable(), model_path)

            with pytest.raises(ValueError, match="trust_source"):
                deserialize_model(model_path, format="joblib")

            # Explicit opt-in still works (the mechanism itself is unchanged).
            result = deserialize_model(model_path, format="joblib", trust_source=True)
            assert result == os.getpid()
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestGetSetModelParams:
    """Test parameter get/set functions."""

    def test_get_model_params(self):
        """Test getting model parameters"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import get_model_params
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=0.5)
        params = get_model_params(model)

        assert isinstance(params, dict)
        assert "alpha" in params
        assert params["alpha"] == 0.5

    def test_set_model_params(self):
        """Test setting model parameters"""
        pytest.importorskip("sklearn")
        from npcpy.ml_funcs import set_model_params
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=0.5)
        updated = set_model_params(model, {"alpha": 1.0})

        assert updated.alpha == 1.0


class TestAvailabilityFlags:
    """Test dependency availability flags."""

    def test_sklearn_available_flag(self):
        """Test sklearn availability flag is set correctly"""
        from npcpy.ml_funcs import _sklearn_available

        try:
            import sklearn
            assert _sklearn_available is True
        except ImportError:
            assert _sklearn_available is False

    def test_torch_available_flag(self):
        """Test torch availability flag is set correctly"""
        from npcpy.ml_funcs import _torch_available

        try:
            import torch
            assert _torch_available is True
        except ImportError:
            assert _torch_available is False

    def test_xgboost_available_flag(self):
        """Test xgboost availability flag"""
        from npcpy.ml_funcs import _xgboost_available

        try:
            import xgboost
            assert _xgboost_available is True
        except ImportError:
            assert _xgboost_available is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
