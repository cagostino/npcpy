"""Tests for embeddings fine-tuning module."""

import pytest
import numpy as np


class TestEmbeddingConfig:
    """Test EmbeddingConfig and HilbertConfig dataclasses."""

    def test_embedding_config_defaults(self):
        from npcpy.ft.embeddings import EmbeddingConfig

        config = EmbeddingConfig()
        assert config.base_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.output_model_path == "models/embedding"
        assert config.device == "cpu"
        assert config.embedding_dim == 384
        assert config.num_train_epochs == 10
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
        assert config.temperature == 0.07
        assert config.margin == 0.5
        assert config.loss_type == "infonce"
        assert config.max_length == 256

    def test_embedding_config_custom(self):
        from npcpy.ft.embeddings import EmbeddingConfig

        config = EmbeddingConfig(
            base_model_name="bert-base-uncased",
            output_model_path="custom/path",
            device="cuda",
            embedding_dim=768,
            loss_type="triplet",
        )
        assert config.base_model_name == "bert-base-uncased"
        assert config.output_model_path == "custom/path"
        assert config.device == "cuda"
        assert config.embedding_dim == 768
        assert config.loss_type == "triplet"

    def test_hilbert_config_defaults(self):
        from npcpy.ft.embeddings import HilbertConfig

        config = HilbertConfig()
        assert config.base_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.output_model_path == "models/hilbert_embedding"
        assert config.use_phase is True
        assert config.lambda_phase == 0.5
        assert config.phase_init_scale == 0.1
        assert config.loss_type == "hilbert_infonce"

    def test_hilbert_config_custom(self):
        from npcpy.ft.embeddings import HilbertConfig

        config = HilbertConfig(
            use_phase=False,
            lambda_phase=0.3,
            loss_type="phase_triplet",
        )
        assert config.use_phase is False
        assert config.lambda_phase == 0.3
        assert config.loss_type == "phase_triplet"


class TestHelpers:
    """Test internal helper functions."""

    def test_complex_tensor_init(self):
        from npcpy.ft.embeddings import ComplexTensor
        import torch

        mag = torch.randn(4, 10)
        ang = torch.randn(4, 10)
        ct = ComplexTensor(mag, ang)
        assert ct.magnitude is mag
        assert ct.angle is ang

    def test_hilbert_similarity_range(self):
        from npcpy.ft.embeddings import ComplexTensor, _hilbert_similarity
        import torch

        # Two identical states should have similarity 1
        mag = torch.ones(1, 5)
        ang = torch.zeros(1, 5)
        ct1 = ComplexTensor(mag, ang)
        ct2 = ComplexTensor(mag, ang)
        sim = _hilbert_similarity(ct1, ct2)
        assert sim.item() == pytest.approx(1.0, abs=1e-5)

        # Two orthogonal states (in phase terms) should have similarity 0
        ang2 = torch.ones(1, 5) * (np.pi / 2)
        ct3 = ComplexTensor(mag, ang2)
        sim2 = _hilbert_similarity(ct1, ct3)
        # Cos(pi/2) = 0, so real part should be near 0
        assert abs(sim2.item()) < 0.1

    def test_hilbert_similarity_matrix_shape(self):
        from npcpy.ft.embeddings import ComplexTensor, _hilbert_similarity_matrix
        import torch

        mag1 = torch.randn(3, 8)
        ang1 = torch.randn(3, 8)
        mag2 = torch.randn(5, 8)
        ang2 = torch.randn(5, 8)
        ct1 = ComplexTensor(mag1, ang1)
        ct2 = ComplexTensor(mag2, ang2)
        sim = _hilbert_similarity_matrix(ct1, ct2)
        assert sim.shape == (3, 5)

    def test_normalize_hilbert(self):
        from npcpy.ft.embeddings import ComplexTensor, _normalize_hilbert
        import torch

        mag = torch.randn(2, 6)
        ang = torch.randn(2, 6)
        ct = ComplexTensor(mag, ang)
        normed = _normalize_hilbert(ct)
        # Check norms are ~1
        norms = (normed.magnitude ** 2).sum(dim=-1).sqrt()
        assert norms[0].item() == pytest.approx(1.0, abs=1e-5)
        assert norms[1].item() == pytest.approx(1.0, abs=1e-5)


class TestTorchAvailability:
    """Test availability flags."""

    def test_torch_available_exists(self):
        from npcpy.ft.embeddings import TORCH_AVAILABLE
        assert isinstance(TORCH_AVAILABLE, bool)

    def test_mlx_available_exists(self):
        from npcpy.ft.embeddings import MLX_AVAILABLE
        assert isinstance(MLX_AVAILABLE, bool)


@pytest.mark.slow
class TestClassicalEmbeddingTraining:
    """Integration tests for classical embedding fine-tuning."""

    @pytest.fixture
    def dummy_data(self):
        anchors = ["This is sentence A", "This is sentence B", "This is sentence C"]
        positives = ["Sentence A paraphrase", "Sentence B paraphrase", "Sentence C paraphrase"]
        negatives = ["Unrelated sentence X", "Unrelated sentence Y", "Unrelated sentence Z"]
        return anchors, positives, negatives

    def test_run_embedding_sft_torch(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_embedding_sft_torch, EmbeddingConfig

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_embedding"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="triplet",
            max_length=32,
        )
        path = run_embedding_sft_torch(anchors, positives, negatives, config=config)
        assert path == config.output_model_path
        import os
        assert os.path.exists(os.path.join(path, "model.pt"))

    def test_run_embedding_sft_infonce(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_embedding_sft_torch, EmbeddingConfig

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_embedding_infonce"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="infonce",
            max_length=32,
        )
        path = run_embedding_sft_torch(anchors, positives, config=config)
        assert path == config.output_model_path

    def test_load_and_encode(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import (
            run_embedding_sft_torch, load_embedding_model, encode_texts, EmbeddingConfig
        )

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_embedding_load"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="infonce",
            max_length=32,
        )
        run_embedding_sft_torch(anchors, positives, config=config)
        base, projector, tokenizer, loaded_config = load_embedding_model(
            config.output_model_path, device="cpu"
        )
        assert loaded_config.base_model_name == config.base_model_name
        assert loaded_config.embedding_dim == config.embedding_dim

        embs = encode_texts(["test sentence"], base, projector, tokenizer, device="cpu", max_length=32)
        assert len(embs) == 1
        assert len(embs[0]) == config.embedding_dim

    def test_evaluate_embeddings(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import (
            run_embedding_sft_torch, load_embedding_model, evaluate_embeddings, EmbeddingConfig
        )

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_embedding_eval"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="triplet",
            max_length=32,
        )
        run_embedding_sft_torch(anchors, positives, negatives, config=config)
        base, projector, tokenizer, _ = load_embedding_model(config.output_model_path, device="cpu")
        metrics = evaluate_embeddings(
            anchors, positives, negatives,
            base, projector, tokenizer, device="cpu", max_length=32
        )
        assert "mrr" in metrics
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert 0.0 <= metrics["mrr"] <= 1.0


@pytest.mark.slow
class TestHilbertEmbeddingTraining:
    """Integration tests for Hilbert-space embedding fine-tuning."""

    @pytest.fixture
    def dummy_data(self):
        anchors = ["This is sentence A", "This is sentence B", "This is sentence C"]
        positives = ["Sentence A paraphrase", "Sentence B paraphrase", "Sentence C paraphrase"]
        negatives = ["Unrelated sentence X", "Unrelated sentence Y", "Unrelated sentence Z"]
        return anchors, positives, negatives

    def test_run_hilbert_embedding_sft_torch(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_hilbert_embedding_sft_torch, HilbertConfig

        anchors, positives, negatives = dummy_data
        config = HilbertConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_hilbert"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="hilbert_infonce",
            max_length=32,
        )
        path = run_hilbert_embedding_sft_torch(anchors, positives, config=config)
        assert path == config.output_model_path
        import os
        assert os.path.exists(os.path.join(path, "model.pt"))

    def test_run_hilbert_phase_triplet(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_hilbert_embedding_sft_torch, HilbertConfig

        anchors, positives, negatives = dummy_data
        config = HilbertConfig(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            output_model_path=str(tmp_path / "test_hilbert_triplet"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="phase_triplet",
            max_length=32,
        )
        path = run_hilbert_embedding_sft_torch(anchors, positives, negatives, config=config)
        assert path == config.output_model_path


class TestFoundationModelTraining:
    """Tests for training embeddings from scratch (no pretrained weights)."""

    @pytest.fixture
    def dummy_data(self):
        anchors = ["This is sentence A", "This is sentence B", "This is sentence C"]
        positives = ["Sentence A paraphrase", "Sentence B paraphrase", "Sentence C paraphrase"]
        negatives = ["Unrelated sentence X", "Unrelated sentence Y", "Unrelated sentence Z"]
        return anchors, positives, negatives

    @pytest.mark.slow
    def test_train_from_scratch_classical(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_embedding_sft_torch, EmbeddingConfig

        anchors, positives, negatives = dummy_data
        config = EmbeddingConfig(
            base_model_name="prajjwal1/bert-tiny",  # Tiny model for fast testing
            output_model_path=str(tmp_path / "test_scratch"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="infonce",
            max_length=32,
            embedding_dim=128,
        )
        path = run_embedding_sft_torch(anchors, positives, config=config)
        assert path == config.output_model_path

    @pytest.mark.slow
    def test_train_from_scratch_hilbert(self, tmp_path, dummy_data):
        from npcpy.ft.embeddings import run_hilbert_embedding_sft_torch, HilbertConfig

        anchors, positives, negatives = dummy_data
        config = HilbertConfig(
            base_model_name="prajjwal1/bert-tiny",
            output_model_path=str(tmp_path / "test_hilbert_scratch"),
            device="cpu",
            num_train_epochs=1,
            batch_size=2,
            loss_type="hilbert_infonce",
            max_length=32,
            embedding_dim=128,
        )
        path = run_hilbert_embedding_sft_torch(anchors, positives, config=config)
        assert path == config.output_model_path


class TestEncodeTextsBatching:
    """`encode_texts` must chunk its forward pass instead of doing one giant pass.

    Pre-fix it tokenized and forwarded the entire `texts` list at once, so peak
    memory scaled with `len(texts)` and large inputs went OOM. `evaluate_embeddings`
    is an in-repo caller that hands it full datasets.

    These tests use a stub tokenizer and a tiny torch module, so they need no
    network and no HuggingFace download (unlike the `@pytest.mark.slow` suites).
    """

    HIDDEN = 8
    VOCAB = 16
    DIM = 4

    @staticmethod
    def _token_id(token, vocab):
        # Deterministic across processes, unlike hash() on str.
        return (sum(ord(c) for c in token) % (vocab - 1)) + 1

    @pytest.fixture
    def stubs(self):
        import torch
        from types import SimpleNamespace

        vocab, hidden, dim = self.VOCAB, self.HIDDEN, self.DIM
        token_id = self._token_id

        class StubTokenizer:
            """Pads each call to that call's own batch maximum, like a real one."""

            def __init__(self):
                self.batches = []

            def __call__(self, texts, padding=True, truncation=True,
                         max_length=256, return_tensors="pt"):
                texts = list(texts)
                self.batches.append(texts)
                token_lists = [t.split()[:max_length] for t in texts]
                width = max(len(t) for t in token_lists)
                input_ids = torch.zeros(len(texts), width, dtype=torch.long)
                attention_mask = torch.zeros(len(texts), width, dtype=torch.long)
                for row, tokens in enumerate(token_lists):
                    for col, tok in enumerate(tokens):
                        input_ids[row, col] = token_id(tok, vocab)
                        attention_mask[row, col] = 1
                return {"input_ids": input_ids, "attention_mask": attention_mask}

        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab, hidden)
                self.forward_batch_sizes = []

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                self.forward_batch_sizes.append(int(input_ids.shape[0]))
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        torch.manual_seed(0)
        model = StubModel()
        projector = torch.nn.Linear(hidden, dim)
        return model, projector, StubTokenizer()

    @staticmethod
    def _texts(n):
        # Deliberately uneven token counts so per-batch padding differs by chunk.
        return [" ".join(["tok%d" % (i + k) for k in range(1 + i % 3)]) for i in range(n)]

    def test_chunks_the_forward_pass_by_batch_size(self, stubs):
        from npcpy.ft.embeddings import encode_texts

        model, projector, tokenizer = stubs
        embeddings = encode_texts(self._texts(7), model, projector, tokenizer,
                                  device="cpu", max_length=32, batch_size=3)

        # 7 texts at batch_size=3 => three forward passes, never one pass of 7.
        assert model.forward_batch_sizes == [3, 3, 1]
        assert [len(b) for b in tokenizer.batches] == [3, 3, 1]
        assert len(embeddings) == 7
        assert all(len(e) == self.DIM for e in embeddings)

    def test_batching_does_not_change_the_embeddings(self, stubs):
        """Chunking must be numerically transparent.

        Padding is per-batch, but `_mean_pooling` divides by the attention mask,
        so pad positions cannot leak into the mean.
        """
        from npcpy.ft.embeddings import encode_texts

        model, projector, tokenizer = stubs
        texts = self._texts(6)

        one_pass = encode_texts(texts, model, projector, tokenizer,
                                device="cpu", max_length=32, batch_size=len(texts))
        per_item = encode_texts(texts, model, projector, tokenizer,
                                device="cpu", max_length=32, batch_size=1)
        uneven = encode_texts(texts, model, projector, tokenizer,
                              device="cpu", max_length=32, batch_size=4)

        assert model.forward_batch_sizes == [6, 1, 1, 1, 1, 1, 1, 4, 2]
        np.testing.assert_allclose(np.array(per_item), np.array(one_pass), atol=1e-6)
        np.testing.assert_allclose(np.array(uneven), np.array(one_pass), atol=1e-6)

    def test_default_batch_size_bounds_peak_memory(self, stubs):
        """The default must not be unbounded: 70 texts cannot be one pass of 70."""
        from npcpy.ft.embeddings import encode_texts

        model, projector, tokenizer = stubs
        embeddings = encode_texts(self._texts(70), model, projector, tokenizer,
                                  device="cpu", max_length=32)

        assert len(embeddings) == 70
        assert max(model.forward_batch_sizes) <= 32
        assert len(model.forward_batch_sizes) > 1

    def test_empty_input_does_no_forward_pass(self, stubs):
        from npcpy.ft.embeddings import encode_texts

        model, projector, tokenizer = stubs
        assert encode_texts([], model, projector, tokenizer, device="cpu") == []
        assert model.forward_batch_sizes == []
        assert tokenizer.batches == []

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_non_positive_batch_size(self, stubs, bad):
        from npcpy.ft.embeddings import encode_texts

        model, projector, tokenizer = stubs
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            encode_texts(self._texts(3), model, projector, tokenizer,
                         device="cpu", batch_size=bad)

    def test_evaluate_embeddings_threads_batch_size_through(self, stubs):
        from npcpy.ft.embeddings import evaluate_embeddings

        model, projector, tokenizer = stubs
        anchors = self._texts(4)
        positives = self._texts(4)
        negatives = self._texts(4)

        metrics = evaluate_embeddings(anchors, positives, negatives, model,
                                      projector, tokenizer, device="cpu",
                                      max_length=32, batch_size=2)

        # Three lists of 4 at batch_size=2 => 6 forward passes of 2, none of 4.
        assert model.forward_batch_sizes == [2, 2, 2, 2, 2, 2]
        assert set(metrics) == {"mrr", "recall@1", "recall@5"}
        assert 0.0 <= metrics["mrr"] <= 1.0
