import inspect
import pkgutil
import re
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.status import HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR

from ingenious.core.structured_logging import get_logger
from ingenious.utils.imports import import_module_with_fallback
from ingenious.utils.namespace_utils import (
    discover_custom_workflows,
    get_path_from_namespace_with_fallback,
    normalize_workflow_name,
)

router = APIRouter()
logger = get_logger(__name__)


@router.get("/custom-workflows/list", response_model=Dict[str, Any])
async def list_available_custom_workflows() -> Dict[str, Any]:
    """
    Lists all available custom workflows using the dedicated discovery function
    that validates custom workflows.
    """
    try:
        discovery_result = discover_custom_workflows()

        return {
            "custom_workflows": discovery_result["workflows"],
            "count": len(discovery_result["workflows"]),
            "discovered_from": discovery_result["discovered_from"],  # Moved to root
        }
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during workflow discovery: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while listing custom workflows.",
        )


@router.get(
    "/custom-workflows/schema/{custom_workflow_name}/", response_model=Dict[str, Any]
)
async def get_custom_workflow_schema(
    custom_workflow_name: str, request: Request
) -> Dict[str, Any]:
    """
    Retrieves Pydantic model schemas optimized for Alpine.js dynamic UI generation.
    Returns a structured schema with UI metadata and field ordering.
    """
    try:
        normalized_workflow_name = normalize_workflow_name(custom_workflow_name)
        models_dir_rel_path = f"models/{normalized_workflow_name}"
        models_path = get_path_from_namespace_with_fallback(models_dir_rel_path)

        if not models_path or not models_path.is_dir():
            # Include additional detail about the absolute path where the app is looking
            attempted_absolute_path = (
                str(models_path.resolve())
                if models_path
                else f"Unable to resolve path for '{models_dir_rel_path}'"
            )
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=(
                    f"Models directory for workflow '{custom_workflow_name}' not found. "
                    f"Attempted relative path: '{models_dir_rel_path}', "
                    f"Resolved absolute path: {attempted_absolute_path}."
                ),
            )

        # Collect all Pydantic models first
        pydantic_models = {}
        model_classes = {}

        for module_info in pkgutil.iter_modules([str(models_path)]):
            if module_info.ispkg or module_info.name == "agent":
                continue

            module_import_path = f"models.{normalized_workflow_name}.{module_info.name}"
            try:
                module = import_module_with_fallback(module_import_path)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseModel) and obj is not BaseModel:
                        pydantic_models[name] = obj.model_json_schema()
                        model_classes[name] = obj
            except (ImportError, AttributeError) as e:
                logger.error(
                    f"Error processing schema module {module_import_path}: {e}"
                )
                continue

        if not pydantic_models:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"No Pydantic models found for workflow '{normalized_workflow_name}'.",
            )

        # Transform schemas for Alpine.js
        alpine_schema = transform_schemas_for_alpine(pydantic_models, model_classes)

        response_data = {
            "workflow_name": custom_workflow_name,
            "schemas": alpine_schema,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_models": len(pydantic_models),
                "alpine_version": "3.x",
                "features": {
                    "validation": True,
                    "nested_objects": True,
                    "arrays": True,
                    "unions": True,
                    "conditional_fields": True,
                },
            },
        }

        # Return JSONResponse (headers handled by middleware)
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while retrieving Alpine schema for '{custom_workflow_name}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the workflow schema.",
        )


def transform_schemas_for_alpine(
    schemas: Dict[str, Any], model_classes: Dict[str, type[BaseModel]]
) -> Dict[str, Any]:
    """
    Transform Pydantic JSON schemas into Alpine.js-friendly format with UI metadata.
    """
    alpine_schemas = {}

    for model_name, schema in schemas.items():
        model_class = model_classes.get(model_name)

        alpine_schema = {
            "model_name": model_name,
            "title": schema.get("title", model_name),
            "description": schema.get("description", ""),
            "type": schema.get("type", "object"),
            "properties": {},
            "required": schema.get("required", []),
            "ui_metadata": {
                "display_order": [],
                "field_groups": {},
                "conditional_fields": {},
                "validation_rules": {},
            },
        }

        # Process properties with Alpine.js enhancements
        if "properties" in schema:
            for field_name, field_schema in schema["properties"].items():
                alpine_field = transform_field_for_alpine(
                    field_name, field_schema, model_class
                )
                alpine_schema["properties"][field_name] = alpine_field
                alpine_schema["ui_metadata"]["display_order"].append(field_name)

        # Handle discriminated unions (like your bike types)
        if "$defs" in schema:
            alpine_schema["definitions"] = {}
            for def_name, def_schema in schema["$defs"].items():
                alpine_schema["definitions"][def_name] = (
                    transform_definition_for_alpine(def_name, def_schema)
                )

        # Add form initialization data
        alpine_schema["default_values"] = generate_default_values(alpine_schema)

        alpine_schemas[model_name] = alpine_schema

    return alpine_schemas


def transform_field_for_alpine(
    field_name: str,
    field_schema: Dict[str, Any],
    model_class: type[BaseModel] | None = None,
) -> Dict[str, Any]:
    """
    Transform individual field schema for Alpine.js with UI hints.
    """
    alpine_field = {
        **field_schema,
        "ui_component": determine_ui_component(field_schema),
        "validation": extract_validation_rules(field_schema),
        "alpine_model": f"formData.{field_name}",
        "display_name": field_schema.get("title", field_name.replace("_", " ").title()),
    }

    # Handle special field types
    field_type = field_schema.get("type")
    field_format = field_schema.get("format")

    # Add Alpine.js specific attributes
    if field_type == "array":
        alpine_field["ui_component"] = "array"
        alpine_field["array_config"] = {
            "min_items": field_schema.get("minItems", 0),
            "max_items": field_schema.get("maxItems"),
            "item_schema": field_schema.get("items", {}),
            "add_button_text": f"Add {field_name.replace('_', ' ').title()}",
            "remove_button_text": "Remove",
        }

    elif field_type == "object":
        alpine_field["ui_component"] = "nested_object"
        alpine_field["nested_properties"] = field_schema.get("properties", {})

    elif "anyOf" in field_schema or "oneOf" in field_schema:
        alpine_field["ui_component"] = "union_select"
        alpine_field["union_options"] = extract_union_options(field_schema)

    elif field_format == "date":
        alpine_field["ui_component"] = "date_input"

    elif field_format == "email":
        alpine_field["ui_component"] = "email_input"

    elif field_type == "boolean":
        alpine_field["ui_component"] = "checkbox"

    elif field_type in ["integer", "number"]:
        alpine_field["ui_component"] = "number_input"
        alpine_field["number_config"] = {
            "min": field_schema.get("minimum"),
            "max": field_schema.get("maximum"),
            "step": 1 if field_type == "integer" else 0.01,
        }

    elif field_schema.get("enum"):
        alpine_field["ui_component"] = "select"
        alpine_field["options"] = [
            {"value": opt, "label": str(opt)} for opt in field_schema["enum"]
        ]

    else:
        alpine_field["ui_component"] = "text_input"

    return alpine_field


def determine_ui_component(field_schema: Dict[str, Any]) -> str:
    """Determine the appropriate UI component for Alpine.js rendering."""
    field_type = field_schema.get("type")
    field_format = field_schema.get("format")

    if field_schema.get("enum"):
        return "select"
    elif field_type == "boolean":
        return "checkbox"
    elif field_type == "array":
        return "array"
    elif field_type == "object":
        return "nested_object"
    elif "anyOf" in field_schema or "oneOf" in field_schema:
        return "union_select"
    elif field_format == "date":
        return "date_input"
    elif field_format == "email":
        return "email_input"
    elif field_format == "password":
        return "password_input"
    elif field_type in ["integer", "number"]:
        return "number_input"
    elif field_type == "string" and field_schema.get("maxLength", 0) > 100:
        return "textarea"
    else:
        return "text_input"


def extract_validation_rules(field_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extract validation rules for Alpine.js client-side validation."""
    rules = {}

    if field_schema.get("minLength"):
        rules["minLength"] = field_schema["minLength"]
    if field_schema.get("maxLength"):
        rules["maxLength"] = field_schema["maxLength"]
    if field_schema.get("minimum"):
        rules["min"] = field_schema["minimum"]
    if field_schema.get("maximum"):
        rules["max"] = field_schema["maximum"]
    if field_schema.get("pattern"):
        rules["pattern"] = field_schema["pattern"]
    if field_schema.get("format"):
        rules["format"] = field_schema["format"]

    return rules


def extract_union_options(field_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract union type options for discriminated unions."""
    options = []

    union_types = field_schema.get("anyOf", field_schema.get("oneOf", []))

    for i, union_type in enumerate(union_types):
        if "$ref" in union_type:
            # Handle reference types
            ref_name = union_type["$ref"].split("/")[-1]
            options.append(
                {
                    "value": ref_name.lower(),
                    "label": ref_name,
                    "schema_ref": union_type["$ref"],
                    "discriminator": ref_name,
                }
            )
        else:
            # Handle inline types
            title = union_type.get("title", f"Option {i + 1}")
            options.append(
                {"value": title.lower(), "label": title, "schema": union_type}
            )

    return options


def generate_default_values(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Generate default form values for Alpine.js initialization."""
    defaults = {}

    for field_name, field_schema in schema.get("properties", {}).items():
        field_type = field_schema.get("type")
        default_value = field_schema.get("default")

        if default_value is not None:
            defaults[field_name] = default_value
        elif field_type == "array":
            defaults[field_name] = []
        elif field_type == "object":
            defaults[field_name] = {}
        elif field_type == "boolean":
            defaults[field_name] = False
        elif field_type in ["integer", "number"]:
            defaults[field_name] = 0
        elif field_type == "string":
            defaults[field_name] = ""
        else:
            defaults[field_name] = None

    return defaults


def transform_definition_for_alpine(
    def_name: str, def_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """Transform schema definitions for Alpine.js discriminated unions."""
    return {
        "name": def_name,
        "title": def_schema.get("title", def_name),
        "type": def_schema.get("type", "object"),
        "properties": def_schema.get("properties", {}),
        "required": def_schema.get("required", []),
        "discriminator": extract_discriminator_info(def_schema),
    }


def extract_discriminator_info(schema: Dict[str, Any]) -> Dict[str, Any] | None:
    """Extract discriminator information for union types."""
    if "discriminator" in schema:
        return schema["discriminator"]

    # Try to infer discriminator from class hierarchy
    title = schema.get("title", "")
    if title:
        return {"property_name": "type", "mapping": {title.lower(): title}}

    return None


@router.get(
    "/custom-workflows/agents/{custom_workflow_name}/", response_model=Dict[str, Any]
)
async def get_custom_workflow_agents(custom_workflow_name: str) -> Dict[str, Any]:
    """
    Retrieves agent information by parsing the agent.py file of the specified custom workflow.
    This approach uses source code inspection to avoid runtime execution.
    """
    try:
        normalized_workflow_name = normalize_workflow_name(custom_workflow_name)
        agent_module_path = f"models.{normalized_workflow_name}.agent"
        models_dir_rel_path = f"models/{normalized_workflow_name}"
        models_path = get_path_from_namespace_with_fallback(models_dir_rel_path)

        if not models_path or not models_path.is_dir():
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Models directory for workflow '{custom_workflow_name}' not found.",
            )

        agent_file_path = models_path / "agent.py"
        if not agent_file_path.is_file():
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Agent file not found for workflow '{custom_workflow_name}'.",
            )

        try:
            # Import the module to get access to the class structure
            agent_module = import_module_with_fallback(agent_module_path)

            if not hasattr(agent_module, "ProjectAgents"):
                raise HTTPException(
                    status_code=HTTP_404_NOT_FOUND,
                    detail=f"ProjectAgents class not found in agent.py for workflow '{custom_workflow_name}'.",
                )

            project_agents_class = getattr(agent_module, "ProjectAgents")

            # Instantiation is needed to access the method for inspection,
            # but the method itself is not called.
            project_agents_instance = project_agents_class()

            agents_list = []
            discovery_method = "unknown"

            # Directly parse the source code of the Get_Project_Agents method
            source = inspect.getsource(project_agents_instance.Get_Project_Agents)

            # Parse agent definitions from source code using a detailed regex
            agent_pattern = r'Agent\s*\(\s*agent_name\s*=\s*["\']([^"\']+)["\'].*?agent_model_name\s*=\s*["\']([^"\']+)["\'].*?agent_display_name\s*=\s*["\']([^"\']+)["\'].*?agent_description\s*=\s*["\']([^"\']+)["\'].*?agent_type\s*=\s*["\']([^"\']+)["\'].*?log_to_prompt_tuner\s*=\s*([^,\s)]+).*?return_in_response\s*=\s*([^,\s)]+)'
            matches = re.findall(agent_pattern, source, re.DOTALL)

            if matches:
                discovery_method = "source_code_parsing"
                for match in matches:
                    (
                        agent_name,
                        model_name,
                        display_name,
                        description,
                        agent_type,
                        _,
                        _,
                    ) = match

                    agents_list.append(
                        {
                            "agent_name": agent_name.strip(),
                            "agent_model_name": model_name.strip(),
                            "agent_display_name": display_name.strip(),
                            "agent_description": description.strip(),
                            "agent_type": agent_type.strip(),
                        }
                    )
            else:
                # Final fallback if no agents can be parsed
                discovery_method = "unparsable"
                agents_list.append(
                    {
                        "agent_name": "unknown",
                        "agent_description": "Agent definitions exist but could not be parsed.",
                    }
                )

            return {
                "workflow_name": custom_workflow_name,
                "normalized_workflow_name": normalized_workflow_name,
                "discovered_from": discovery_method,
                "agent_count": len(agents_list),
                "agents": agents_list,
            }

        except ImportError as e:
            logger.error(f"Failed to import agent module {agent_module_path}: {e}")
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Could not import agent module for workflow '{custom_workflow_name}'. {str(e)}",
            )
        except Exception as e:
            logger.error(
                f"Error processing agents for workflow '{custom_workflow_name}': {e}"
            )
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing agents for workflow '{custom_workflow_name}': {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while retrieving agents for '{custom_workflow_name}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the workflow agents.",
        )
