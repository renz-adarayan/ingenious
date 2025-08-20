"""
Tests for the chunk configuration system (ChunkConfig).

This test suite covers all validation logic, edge cases, and configuration
combinations for the ChunkConfig Pydantic model.
"""

# Import ChunkConfig directly to avoid dependency issues with langchain in chunk/__init__.py
import importlib.util
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

spec = importlib.util.spec_from_file_location(
    "chunk_config",
    str(Path(__file__).parent.parent.parent / "ingenious" / "chunk" / "config.py"),
)
chunk_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunk_config_module)
ChunkConfig = chunk_config_module.ChunkConfig


class TestChunkConfigBasic:
    """Test basic ChunkConfig functionality."""

    def test_default_configuration(self):
        """Test ChunkConfig with default values."""
        config = ChunkConfig()

        assert config.strategy == "recursive"
        assert config.id_path_mode == "rel"
        assert config.id_hash_bits == 64
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 128
        assert config.overlap_unit == "tokens"
        assert config.encoding_name == "cl100k_base"
        assert config.semantic_threshold_percentile == 95
        assert config.separators is None
        assert config.embed_model is None
        assert config.azure_openai_deployment is None

    def test_configuration_is_frozen(self):
        """Test that ChunkConfig is immutable after creation."""
        config = ChunkConfig()

        with pytest.raises(ValidationError):
            config.chunk_size = 512

    def test_custom_configuration(self):
        """Test creating ChunkConfig with custom values."""
        config = ChunkConfig(
            strategy="token",
            chunk_size=512,
            chunk_overlap=64,
            overlap_unit="characters",
            encoding_name="gpt2",
        )

        assert config.strategy == "token"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.overlap_unit == "characters"
        assert config.encoding_name == "gpt2"


class TestChunkConfigValidation:
    """Test validation logic in ChunkConfig."""

    def test_valid_strategies(self):
        """Test all valid chunking strategies."""
        strategies = ["recursive", "markdown", "token", "semantic"]

        for strategy in strategies:
            config = ChunkConfig(strategy=strategy)
            assert config.strategy == strategy

    def test_invalid_strategy(self):
        """Test that invalid strategies raise ValidationError."""
        with pytest.raises(ValidationError):
            ChunkConfig(strategy="invalid_strategy")

    def test_valid_id_path_modes(self):
        """Test all valid id_path_mode values."""
        modes = ["abs", "rel", "hash"]

        for mode in modes:
            config = ChunkConfig(id_path_mode=mode)
            assert config.id_path_mode == mode

    def test_invalid_id_path_mode(self):
        """Test that invalid id_path_mode raises ValidationError."""
        with pytest.raises(ValidationError):
            ChunkConfig(id_path_mode="invalid_mode")

    def test_chunk_size_positive(self):
        """Test that chunk_size must be positive."""
        with pytest.raises(ValidationError):
            ChunkConfig(chunk_size=0)

        with pytest.raises(ValidationError):
            ChunkConfig(chunk_size=-1)

        # Positive values should work
        config = ChunkConfig(chunk_size=1, chunk_overlap=0)
        assert config.chunk_size == 1

    def test_chunk_overlap_non_negative(self):
        """Test that chunk_overlap must be non-negative."""
        with pytest.raises(ValidationError):
            ChunkConfig(chunk_overlap=-1)

        # Zero and positive values should work
        config = ChunkConfig(chunk_overlap=0)
        assert config.chunk_overlap == 0

        config = ChunkConfig(chunk_overlap=50)
        assert config.chunk_overlap == 50


class TestHashBitsValidation:
    """Test id_hash_bits validation logic."""

    def test_hash_bits_range(self):
        """Test id_hash_bits range validation."""
        # Below minimum
        with pytest.raises(ValidationError):
            ChunkConfig(id_hash_bits=16)

        # Above maximum
        with pytest.raises(ValidationError):
            ChunkConfig(id_hash_bits=512)

        # Valid range
        config = ChunkConfig(id_hash_bits=32)
        assert config.id_hash_bits == 32

        config = ChunkConfig(id_hash_bits=256)
        assert config.id_hash_bits == 256

    def test_hash_bits_multiple_of_four(self):
        """Test that id_hash_bits must be multiple of 4."""
        # Valid multiples of 4
        valid_values = [32, 36, 40, 64, 128, 256]
        for value in valid_values:
            config = ChunkConfig(id_hash_bits=value)
            assert config.id_hash_bits == value

        # Invalid non-multiples of 4
        invalid_values = [33, 34, 35, 37, 63, 65]
        for value in invalid_values:
            with pytest.raises(
                ValueError, match="id_hash_bits must be a multiple of 4"
            ):
                ChunkConfig(id_hash_bits=value)

    def test_hash_bits_small_warning(self):
        """Test warning for small id_hash_bits values."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should trigger warning
            config = ChunkConfig(id_hash_bits=32)
            assert len(w) == 1
            assert "id_hash_bits < 48 increases the probability" in str(w[0].message)
            assert config.id_hash_bits == 32

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should not trigger warning
            config = ChunkConfig(id_hash_bits=64)
            assert len(w) == 0
            assert config.id_hash_bits == 64


class TestIdModeValidation:
    """Test id_path_mode and id_base validation."""

    def test_rel_mode_defaults_id_base(self):
        """Test that rel mode sets default id_base to cwd."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/test/path")

            config = ChunkConfig(id_path_mode="rel")
            assert config.id_base == Path("/test/path")

    def test_rel_mode_custom_id_base(self):
        """Test that rel mode accepts custom id_base."""
        custom_path = Path("/custom/path")
        config = ChunkConfig(id_path_mode="rel", id_base=custom_path)
        assert config.id_base == custom_path

    def test_abs_mode_no_id_base(self):
        """Test that abs mode doesn't accept id_base."""
        config = ChunkConfig(id_path_mode="abs")
        assert config.id_base is None

        with pytest.raises(
            ValueError, match="id_base is only applicable when id_path_mode == 'rel'"
        ):
            ChunkConfig(id_path_mode="abs", id_base=Path("/test"))

    def test_hash_mode_no_id_base(self):
        """Test that hash mode doesn't accept id_base."""
        config = ChunkConfig(id_path_mode="hash")
        assert config.id_base is None

        with pytest.raises(
            ValueError, match="id_base is only applicable when id_path_mode == 'rel'"
        ):
            ChunkConfig(id_path_mode="hash", id_base=Path("/test"))


class TestOverlapValidation:
    """Test chunk_overlap validation logic."""

    def test_overlap_smaller_than_size_non_semantic(self):
        """Test overlap must be smaller than size for non-semantic strategies."""
        strategies = ["recursive", "markdown", "token"]

        for strategy in strategies:
            # Valid: overlap < size
            config = ChunkConfig(strategy=strategy, chunk_size=100, chunk_overlap=50)
            assert config.chunk_overlap == 50

            # Invalid: overlap == size
            with pytest.raises(
                ValueError, match="chunk_overlap.*must be smaller than chunk_size"
            ):
                ChunkConfig(strategy=strategy, chunk_size=100, chunk_overlap=100)

            # Invalid: overlap > size
            with pytest.raises(
                ValueError, match="chunk_overlap.*must be smaller than chunk_size"
            ):
                ChunkConfig(strategy=strategy, chunk_size=100, chunk_overlap=150)

    def test_semantic_strategy_ignores_overlap_size_check(self):
        """Test semantic strategy allows overlap >= size."""
        # These should all be valid for semantic strategy
        config = ChunkConfig(strategy="semantic", chunk_size=100, chunk_overlap=100)
        assert config.chunk_overlap == 100

        config = ChunkConfig(strategy="semantic", chunk_size=100, chunk_overlap=150)
        assert config.chunk_overlap == 150


class TestSemanticStrategyValidation:
    """Test semantic strategy specific validations."""

    def test_semantic_requires_tokens_overlap_unit(self):
        """Test semantic strategy requires tokens overlap unit."""
        # Valid: tokens unit
        config = ChunkConfig(strategy="semantic", overlap_unit="tokens")
        assert config.overlap_unit == "tokens"

        # Invalid: characters unit
        with pytest.raises(
            ValueError, match="The 'semantic' strategy does not support 'characters'"
        ):
            ChunkConfig(strategy="semantic", overlap_unit="characters")

    def test_semantic_backend_warning_no_config(self):
        """Test warning when semantic strategy has no backend config."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            ChunkConfig(strategy="semantic")
            assert len(w) == 1
            assert "Semantic splitter: no `azure_openai_deployment`" in str(
                w[0].message
            )
            assert "falling back to the public OpenAI endpoint" in str(w[0].message)

    def test_semantic_no_warning_with_azure_deployment(self):
        """Test no warning when azure_openai_deployment is provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            config = ChunkConfig(
                strategy="semantic", azure_openai_deployment="test-deployment"
            )
            assert len(w) == 0
            assert config.azure_openai_deployment == "test-deployment"

    def test_semantic_no_warning_with_embed_model(self):
        """Test no warning when embed_model is provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            config = ChunkConfig(strategy="semantic", embed_model="test-model")
            assert len(w) == 0
            assert config.embed_model == "test-model"


class TestSemanticThresholdValidation:
    """Test semantic_threshold_percentile validation."""

    def test_semantic_threshold_range(self):
        """Test semantic_threshold_percentile range validation."""
        # Below minimum
        with pytest.raises(ValidationError):
            ChunkConfig(semantic_threshold_percentile=-1)

        # Above maximum
        with pytest.raises(ValidationError):
            ChunkConfig(semantic_threshold_percentile=101)

        # Valid range
        config = ChunkConfig(semantic_threshold_percentile=0)
        assert config.semantic_threshold_percentile == 0

        config = ChunkConfig(semantic_threshold_percentile=100)
        assert config.semantic_threshold_percentile == 100

        config = ChunkConfig(semantic_threshold_percentile=95)
        assert config.semantic_threshold_percentile == 95


class TestComplexConfigurations:
    """Test complex configuration scenarios."""

    def test_recursive_with_custom_separators(self):
        """Test recursive strategy with custom separators."""
        separators = ["\n\n", "\n", " "]
        config = ChunkConfig(
            strategy="recursive",
            separators=separators,
            chunk_size=512,
            chunk_overlap=64,
        )

        assert config.separators == separators
        assert config.strategy == "recursive"

    def test_markdown_configuration(self):
        """Test markdown strategy configuration."""
        config = ChunkConfig(
            strategy="markdown",
            chunk_size=800,
            chunk_overlap=100,
            overlap_unit="characters",
        )

        assert config.strategy == "markdown"
        assert config.chunk_size == 800
        assert config.overlap_unit == "characters"

    def test_token_configuration(self):
        """Test token strategy configuration."""
        config = ChunkConfig(
            strategy="token", encoding_name="gpt2", chunk_size=256, chunk_overlap=32
        )

        assert config.strategy == "token"
        assert config.encoding_name == "gpt2"
        assert config.chunk_size == 256

    def test_semantic_full_configuration(self):
        """Test semantic strategy with full configuration."""
        config = ChunkConfig(
            strategy="semantic",
            embed_model="text-embedding-ada-002",
            azure_openai_deployment="embedding-deployment",
            semantic_threshold_percentile=90,
            overlap_unit="tokens",
        )

        assert config.strategy == "semantic"
        assert config.embed_model == "text-embedding-ada-002"
        assert config.azure_openai_deployment == "embedding-deployment"
        assert config.semantic_threshold_percentile == 90


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_chunk_size_and_overlap(self):
        """Test minimal valid chunk_size and chunk_overlap."""
        config = ChunkConfig(chunk_size=1, chunk_overlap=0)
        assert config.chunk_size == 1
        assert config.chunk_overlap == 0

    def test_maximum_hash_bits(self):
        """Test maximum valid id_hash_bits."""
        config = ChunkConfig(id_hash_bits=256)
        assert config.id_hash_bits == 256

    def test_empty_separators_list(self):
        """Test empty separators list."""
        config = ChunkConfig(separators=[])
        assert config.separators == []

    def test_single_separator(self):
        """Test single separator in list."""
        config = ChunkConfig(separators=["\n"])
        assert config.separators == ["\n"]

    def test_multiple_warnings_combined(self):
        """Test multiple warnings can be triggered together."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            ChunkConfig(
                strategy="semantic",
                id_hash_bits=32,  # Should trigger hash bits warning
                # No azure_openai_deployment or embed_model - should trigger semantic warning
            )

            # Should have both warnings
            assert len(w) == 2
            warning_messages = [str(warning.message) for warning in w]
            assert any("id_hash_bits < 48" in msg for msg in warning_messages)
            assert any(
                "Semantic splitter: no `azure_openai_deployment`" in msg
                for msg in warning_messages
            )
