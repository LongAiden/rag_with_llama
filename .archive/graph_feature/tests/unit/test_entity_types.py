"""
Unit tests for graph_processing/entity_types.py.

Covers:
- EntityType enum values and string behavior
- RelationshipType enum values and string behavior
- ENTITY_TYPE_DESCRIPTIONS completeness
- RELATIONSHIP_TYPE_DESCRIPTIONS completeness
"""
import pytest

from graph_processing.entity_types import (
    EntityType,
    RelationshipType,
    ENTITY_TYPE_DESCRIPTIONS,
    RELATIONSHIP_TYPE_DESCRIPTIONS,
)


class TestEntityType:
    def test_is_string_enum(self):
        assert isinstance(EntityType.MODEL, str)

    def test_value_matches_name(self):
        assert EntityType.MODEL.value == "MODEL"
        assert EntityType.ALGORITHM.value == "ALGORITHM"
        assert EntityType.DATASET.value == "DATASET"

    def test_core_types_exist(self):
        core_types = [
            "ALGORITHM", "MODEL", "ARCHITECTURE", "TECHNIQUE",
            "CONCEPT", "LAYER", "ACTIVATION", "LOSS_FUNCTION",
            "OPTIMIZER", "DATASET", "METRIC", "TASK", "FRAMEWORK",
        ]
        for t in core_types:
            assert hasattr(EntityType, t)

    def test_nlp_types_exist(self):
        nlp_types = [
            "TOKENIZER", "EMBEDDING", "CORPUS", "LANGUAGE",
            "LINGUISTIC_FEATURE", "NLP_COMPONENT",
        ]
        for t in nlp_types:
            assert hasattr(EntityType, t)

    def test_llm_types_exist(self):
        llm_types = [
            "PROMPT_TEMPLATE", "FINE_TUNING_METHOD", "QUANTIZATION",
            "ALIGNMENT_METHOD", "DECODING_STRATEGY", "CONTEXT_WINDOW",
            "ATTENTION_MECHANISM", "POSITION_ENCODING", "INFERENCE_ENGINE",
            "SAFETY_TECHNIQUE",
        ]
        for t in llm_types:
            assert hasattr(EntityType, t)


class TestRelationshipType:
    def test_is_string_enum(self):
        assert isinstance(RelationshipType.IS_A, str)

    def test_value_matches_name(self):
        assert RelationshipType.IS_A.value == "IS_A"
        assert RelationshipType.USES.value == "USES"

    def test_core_types_exist(self):
        core_types = [
            "IS_A", "PART_OF", "USES", "IMPROVES", "OUTPERFORMS",
            "TRAINED_ON", "SOLVES", "BASED_ON", "EXTENDS",
        ]
        for t in core_types:
            assert hasattr(RelationshipType, t)

    def test_nlp_types_exist(self):
        nlp_types = [
            "TOKENIZED_BY", "EMBEDDED_BY", "PRETRAINED_ON",
            "FINE_TUNED_WITH", "QUANTIZED_TO", "ALIGNED_WITH",
            "SUPPORTS_LANGUAGE", "GENERATES", "PROMPTED_WITH",
            "DECODED_WITH", "SERVED_BY",
        ]
        for t in nlp_types:
            assert hasattr(RelationshipType, t)


class TestEntityTypeDescriptions:
    def test_all_entity_types_have_descriptions(self):
        for entity_type in EntityType:
            assert entity_type in ENTITY_TYPE_DESCRIPTIONS, \
                f"Missing description for {entity_type}"

    def test_descriptions_are_non_empty_strings(self):
        for entity_type, desc in ENTITY_TYPE_DESCRIPTIONS.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestRelationshipTypeDescriptions:
    def test_all_relationship_types_have_descriptions(self):
        for rel_type in RelationshipType:
            assert rel_type in RELATIONSHIP_TYPE_DESCRIPTIONS, \
                f"Missing description for {rel_type}"

    def test_descriptions_are_non_empty_strings(self):
        for rel_type, desc in RELATIONSHIP_TYPE_DESCRIPTIONS.items():
            assert isinstance(desc, str)
            assert len(desc) > 0
