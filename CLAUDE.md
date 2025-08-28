# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

This project uses **uv** for Python package and environment management. Python 3.13+ is required.

### Common Commands

```bash
# Install/sync dependencies
uv sync

# Add dependencies
uv add <package>                 # Regular dependency
uv add <package> --dev           # Development dependency

# Remove dependencies
uv remove <package>
uv remove <package> --group dev

# Run commands in environment
uv run <command>                 # Execute any command
uv run pytest                    # Run all tests
uv run pytest tests/unit/test_specific.py  # Run single test file
uv run pytest tests/unit/test_specific.py::TestClass::test_method  # Run specific test

# Type checking and linting
uv run mypy .                    # Type checking
uv run pre-commit run --all-files  # Run all linting checks

# View dependency tree
uv tree
```

## CLI Commands (ingen)

The `ingen` CLI provides these commands:
- `ingen init` - Initialize a new project
- `ingen validate` - Validate configuration
- `ingen serve` - Start API server (defaults to port 8000)
- `ingen workflows` - List available workflows
- `ingen prompt-tuner` - Tune prompts
- `ingen test` - Run tests
- `ingen azure-search <query>` - Query Azure search
- `ingen run-rest-api-server` - Start with custom host/port

## High-Level Architecture

### Core Components

**FastAPI Application** (`ingenious/main/app_factory.py`)
- Main API application factory using dependency injection
- RESTful endpoints with OpenAPI documentation
- JWT/Basic auth middleware (`ingenious/auth/`)

**Multi-Agent System** (`ingenious/services/chat_services/multi_agent/`)
- AutoGen-based agent orchestration
- Sequential and parallel conversation patterns
- Token tracking with configurable limits

**Conversation Flows** (`conversation_flows/`)
- Pluggable workflow patterns implementing `IConversationFlow` interface
- Each flow must have a `ConversationFlow` class with `get_conversation_response` method
- Auto-discovered by name match with `conversation_flow` parameter

**Configuration System**
- Environment variables with `INGENIOUS_` prefix
- Pydantic-settings based validation (`ingenious/config/`)
- Migration script for legacy YAML: `scripts/migrate_config.py`

**Storage Layer**
- Repository pattern for data access (`ingenious/db/`)
- Supports SQLite (default), Azure SQL, Cosmos DB
- Connection pooling via `ingenious/db/connection_pool.py`

### Key Patterns

- **Dependency Injection**: Uses dependency-injector for IoC container (`ingenious/services/container.py`)
- **Service Layer**: Business logic in `ingenious/services/`
- **Structured Logging**: Correlation IDs for tracing (`ingenious/core/structured_logging.py`)
- **Azure Builders**: Factory pattern for Azure service clients (`ingenious/client/azure/`)

## Testing

```bash
# Run all tests with coverage
uv run pytest --cov=ingenious --cov-report=html

# Run specific test categories
uv run pytest tests/unit/           # Unit tests only
uv run pytest -m "not azure_integration"  # Skip Azure integration tests

# Type checking (strict mode enabled)
uv run mypy .

# Pre-commit hooks (install first time)
uv run pre-commit install
uv run pre-commit run --all-files
```

## Configuration

Required environment variables for basic setup:

```bash
# Azure OpenAI (required)
INGENIOUS_MODELS__0__API_KEY=your-key
INGENIOUS_MODELS__0__BASE_URL=https://your-resource.openai.azure.com/
INGENIOUS_MODELS__0__MODEL=gpt-4o
INGENIOUS_MODELS__0__API_VERSION=2024-12-01-preview
INGENIOUS_MODELS__0__DEPLOYMENT=your-deployment

# Chat service
INGENIOUS_CHAT_SERVICE__TYPE=multi_agent
INGENIOUS_CHAT_HISTORY__DATABASE_TYPE=sqlite
INGENIOUS_CHAT_HISTORY__DATABASE_PATH=./.tmp/chat_history.db

# Port (if default 8000 is in use)
INGENIOUS_WEB_CONFIGURATION__PORT=8081

# Authentication (optional)
INGENIOUS_WEB_CONFIGURATION__ENABLE_AUTHENTICATION=false
```

## Adding New Conversation Flows

1. Create flow module in `services/chat_services/multi_agent/conversation_flows/your_flow/`
2. Implement `ConversationFlow` class with `get_conversation_response` method
3. Define agents inline or import from `ingenious/models/agent.py`
4. Create Jinja2 prompt templates in `templates/prompts/`
5. Flow is auto-discovered by name match with `conversation_flow` parameter

## API Testing

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Chat endpoint (with auth disabled)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Your question here",
    "conversation_flow": "classification_agent",
    "thread_id": "test123"
  }'
```

## Core Workflows (Built-in)

- `classification_agent` - Text classification
- `knowledge_base_agent` - Knowledge base search with Azure AI Search
- `sql_manipulation_agent` - Natural language SQL queries

## Development Tips

- Enable debug logging: Set `INGENIOUS_LOG_LEVEL=DEBUG`
- Check correlation IDs in structured JSON logs
- API docs available at `/docs` endpoint
- Always validate config before starting: `uv run ingen validate`
- Test payloads saved in `test_payloads/` directory
