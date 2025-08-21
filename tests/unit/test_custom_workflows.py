import json
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from ingenious.api.routes.custom_workflows import (
    get_custom_workflow_agents,
    get_custom_workflow_schema,
)


class TestGetCustomWorkflowAgents:
    """Test suite for get_custom_workflow_agents function."""

    @pytest.fixture
    def mock_agent_file_content(self):
        """Sample agent.py file content for testing."""
        return """
class ProjectAgents:
    def Get_Project_Agents(self):
        agents = [
            Agent(
                agent_name="test_agent",
                agent_model_name="gpt-4",
                agent_display_name="Test Agent",
                agent_description="A test agent for unit testing",
                agent_type="classification"
            ),
            Agent(
                agent_name="analysis_agent",
                agent_model_name="gpt-3.5-turbo",
                agent_display_name="Analysis Agent",
                agent_description="Agent for data analysis",
                agent_type="analysis"
            )
        ]
        return agents
"""

    @pytest.mark.asyncio
    @patch("ingenious.api.routes.custom_workflows.normalize_workflow_name")
    @patch(
        "ingenious.api.routes.custom_workflows.get_path_from_namespace_with_fallback"
    )
    async def test_get_custom_workflow_agents_success(
        self, mock_get_path, mock_normalize, mock_agent_file_content
    ):
        """Test successful retrieval of custom workflow agents."""
        # Setup mocks
        mock_normalize.return_value = "test_workflow"
        mock_path = Mock()
        mock_path.is_dir.return_value = True
        mock_agent_file = Mock()
        mock_agent_file.is_file.return_value = True
        mock_agent_file.read_text.return_value = mock_agent_file_content
        mock_path.__truediv__ = Mock(return_value=mock_agent_file)
        mock_get_path.return_value = mock_path

        # Execute
        result = await get_custom_workflow_agents("test-workflow")

        # Assertions
        assert result["workflow_name"] == "test-workflow"
        assert result["normalized_workflow_name"] == "test_workflow"
        assert result["discovered_from"] == "ast_parsing"
        assert result["agent_count"] == 2
        assert len(result["agents"]) == 2

        # Check first agent
        first_agent = result["agents"][0]
        assert first_agent["agent_name"] == "test_agent"
        assert first_agent["agent_model_name"] == "gpt-4"
        assert first_agent["agent_display_name"] == "Test Agent"
        assert first_agent["agent_description"] == "A test agent for unit testing"
        assert first_agent["agent_type"] == "classification"

        # Check second agent
        second_agent = result["agents"][1]
        assert second_agent["agent_name"] == "analysis_agent"
        assert second_agent["agent_model_name"] == "gpt-3.5-turbo"
        assert second_agent["agent_display_name"] == "Analysis Agent"


class TestGetCustomWorkflowSchema:
    """Test suite for get_custom_workflow_schema function."""

    @pytest.fixture
    def mock_pydantic_model(self) -> type[BaseModel]:
        """Create a mock Pydantic model for testing."""

        class TestModel(BaseModel):
            name: str
            age: int
            is_active: bool = True

            class Config:
                title = "Test Model"
                description = "A test Pydantic model"

        return TestModel

    @pytest.fixture
    def mock_request(self) -> Mock:
        """Create a mock FastAPI request object."""
        from fastapi import Request

        return Mock(spec=Request)

    @pytest.mark.asyncio
    @patch("ingenious.api.routes.custom_workflows.normalize_workflow_name")
    @patch(
        "ingenious.api.routes.custom_workflows.get_path_from_namespace_with_fallback"
    )
    @patch("ingenious.api.routes.custom_workflows.pkgutil.iter_modules")
    @patch("ingenious.api.routes.custom_workflows.import_module_with_fallback")
    @patch("ingenious.api.routes.custom_workflows.inspect.getmembers")
    async def test_get_custom_workflow_schema_success(
        self,
        mock_getmembers: Mock,
        mock_import: Mock,
        mock_iter_modules: Mock,
        mock_get_path: Mock,
        mock_normalize: Mock,
        mock_request: Mock,
        mock_pydantic_model: type[BaseModel],
    ) -> None:
        """Test successful schema retrieval."""
        # Setup mocks
        mock_normalize.return_value = "test_workflow"
        mock_path = Mock()
        mock_path.is_dir.return_value = True
        mock_get_path.return_value = mock_path

        mock_module_info = Mock()
        mock_module_info.ispkg = False
        mock_module_info.name = "test_models"
        mock_iter_modules.return_value = [mock_module_info]

        mock_module = Mock()
        mock_import.return_value = mock_module
        mock_getmembers.return_value = [("TestModel", mock_pydantic_model)]
        result = await get_custom_workflow_schema("test-workflow", mock_request)
        # Assertions
        assert result.status_code == 200
        content = json.loads(result.body)
        assert content["workflow_name"] == "test-workflow"
        assert "schemas" in content
        assert "metadata" in content
        assert content["metadata"]["total_models"] >= 1
        assert content["metadata"]["features"]["validation"] is True
        assert "generated_at" in content["metadata"]

    @pytest.mark.asyncio
    @patch("ingenious.api.routes.custom_workflows.normalize_workflow_name")
    @patch(
        "ingenious.api.routes.custom_workflows.get_path_from_namespace_with_fallback"
    )
    @patch("ingenious.api.routes.custom_workflows.pkgutil.iter_modules")
    @patch("ingenious.api.routes.custom_workflows.import_module_with_fallback")
    @patch("ingenious.api.routes.custom_workflows.inspect.getmembers")
    async def test_get_custom_workflow_schema_multiple_models(
        self,
        mock_getmembers: Mock,
        mock_import: Mock,
        mock_iter_modules: Mock,
        mock_get_path: Mock,
        mock_normalize: Mock,
        mock_request: Mock,
    ) -> None:
        """Test schema retrieval with multiple Pydantic models."""
        mock_normalize.return_value = "test_workflow"
        mock_path = Mock()
        mock_path.is_dir.return_value = True
        mock_get_path.return_value = mock_path

        # Create multiple module infos
        module_info1 = Mock()
        module_info1.ispkg = False
        module_info1.name = "models1"

        module_info2 = Mock()
        module_info2.ispkg = False
        module_info2.name = "models2"

        mock_iter_modules.return_value = [module_info1, module_info2]

        # Create multiple mock models
        class Model1(BaseModel):
            field1: str

        class Model2(BaseModel):
            field2: int

        mock_module = Mock()
        mock_import.return_value = mock_module
        mock_getmembers.return_value = [("Model1", Model1), ("Model2", Model2)]
        result = await get_custom_workflow_schema("test-workflow", mock_request)
        # Check that both models are included
        content = json.loads(result.body)
        assert content["metadata"]["total_models"] == 2
        assert "Model1" in content["schemas"] or "Model2" in content["schemas"]
