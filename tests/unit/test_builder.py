# tests/unit/test_builder.py
import pytest
from hypothesis import given, strategies as st
from gecko.core.builder import AgentBuilder
from gecko.core.exceptions import AgentError

def test_builder_chain(mock_model):
    agent = AgentBuilder()\
        .with_model(mock_model)\
        .build()
    assert agent.model == mock_model

def test_builder_missing_model():
    with pytest.raises(ValueError, match="Model is required. Call .with_model(...) first."):
        AgentBuilder().build()