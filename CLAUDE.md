# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Context

This is the **ingenious** package - a core AI agent framework library (v0.2.5) that's part of a monorepo. The sister project `mrwa-defect-chat` (in `../mrwa-defect-chat/`) demonstrates production usage with Azure integrations.

## Package Management

Uses **uv** for Python package and environment management. Python 3.13+ is required.

### Common Commands

```bash
# Install/sync dependencies
uv sync

# Add/remove dependencies
uv add <package>                 # Regular dependency
uv add <package> --dev           # Development dependency
uv remove <package>

# Run commands in environment
uv run pytest                    # Run all tests
uv run pytest tests/unit/test_specific.py  # Run single test file
uv run pytest tests/unit/test_specific.py::TestClass::test_method  # Run specific test

# Type checking and linting
uv run mypy .                    # Type checking (strict mode enabled)
uv run pre-commit run --all-files  # All linting checks

# View dependency tree
uv tree
```

## Building and Testing

```bash
# Run tests with coverage
uv run pytest --cov=ingenious

# Type checking (strict mode enabled)
uv run mypy .

# Linting (uses ruff)
uv run pre-commit run --all-files

# Build package
uv build
```

## High-Level Architecture

### Core Components

- **FastAPI Server** (`ingenious/main/app_factory.py`) - Main API application factory using dependency injection
- **Multi-Agent System** (`ingenious/services/chat_services/multi_agent/`) - AutoGen-based agent orchestration
- **Conversation Flows** (`services/chat_services/multi_agent/conversation_flows/`) - Pluggable workflow patterns
- **Dependency Injection** (`ingenious/services/container.py`) - Uses dependency-injector for IoC container
- **Configuration** - Pydantic-settings based (`ingenious/config/`) with `INGENIOUS_*` environment variables

### Built-in Conversation Flows

Located in `ingenious/services/chat_services/multi_agent/conversation_flows/`:
- `classification_agent` - Text classification and sentiment analysis
- `knowledge_base_agent` - Knowledge base search with Azure AI Search/ChromaDB
- `sql_manipulation_agent` - Natural language SQL queries
- `product_recommendation` - Legacy product recommendation flow
- `product_recommendation_v2` - Updated recommendation flow

### Key Architectural Patterns

- Repository pattern for data access (`ingenious/db/`)
- Service layer for business logic (`ingenious/services/`)
- Structured logging with correlation IDs (`ingenious/core/structured_logging.py`)
- JWT/Basic auth middleware (`ingenious/auth/`)
- Azure service builders with authentication (`ingenious/client/azure/`)

## CLI Commands

The `ingen` CLI (`ingenious/cli/`) provides:
- `ingen init` - Initialize a new project with templates
- `ingen validate` - Validate configuration
- `ingen serve` - Start API server
- `ingen run-rest-api-server` - Start with custom host/port
- `ingen test` - Run tests

## Configuration

Environment variables with `INGENIOUS_` prefix (using pydantic-settings):

```bash
# Required Azure OpenAI
INGENIOUS_MODELS__0__API_KEY=your-key
INGENIOUS_MODELS__0__BASE_URL=https://your-resource.openai.azure.com/
INGENIOUS_MODELS__0__MODEL=gpt-4o
INGENIOUS_MODELS__0__API_VERSION=2024-12-01-preview
INGENIOUS_MODELS__0__DEPLOYMENT=your-deployment

# Chat service
INGENIOUS_CHAT_SERVICE__TYPE=multi_agent
INGENIOUS_CHAT_HISTORY__DATABASE_TYPE=sqlite  # or azuresql
INGENIOUS_CHAT_HISTORY__DATABASE_PATH=./.tmp/chat_history.db

# Authentication (optional)
INGENIOUS_WEB_CONFIGURATION__ENABLE_AUTHENTICATION=true
INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__USERNAME=admin
INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__PASSWORD=secure_password
```

Legacy YAML migration: `uv run python scripts/migrate_config.py --yaml-file config.yml --output .env`

## Development Workflow

### Adding New Conversation Flows

1. Create flow module in `ingenious_extensions/services/chat_services/multi_agent/conversation_flows/your_flow/`
2. Implement `IConversationFlow` interface with `get_conversation_response` method
3. Define agents inline or import from `ingenious/models/ag_agents/`
4. Create Jinja2 prompt templates in `templates/prompts/`
5. Flow is auto-discovered by name match with `conversation_flow` parameter

### Project Template

Running `ingen init` creates:
- `ingenious_extensions/` directory with example `bike-insights` workflow
- Templates directory with Jinja2 prompts
- Sample configuration files

### Testing

```bash
# Unit tests only
uv run pytest tests/unit/

# Integration tests (requires Azure credentials)
uv run pytest tests/integration/

# Specific test markers
uv run pytest -m "not azure_integration"  # Skip Azure tests
uv run pytest -m unit  # Unit tests only

# With verbose output
uv run pytest -v

# Coverage report
uv run pytest --cov=ingenious --cov-report=html
```

## Database Integration

Supports multiple backends via repository pattern:
- **SQLite** (default) - Local development
- **Azure SQL** - Production deployment via pyodbc
- **Cosmos DB** - Document storage
- **ChromaDB** - Vector database for embeddings
- Connection pooling via `ingenious/db/connection_pool.py`

## Azure Service Integration

### Azure Search (`ingenious/services/azure_search/`)
- Classic retrieval with BM25
- Document fusion (DAT)
- Semantic reranking
- Generative answering (RAG)
- CLI tool: `uv run azure-search "query"`

### Azure Blob Storage (`ingenious/files/azure/`)
- Prompt template storage
- Document upload/download
- File summarization support

### Azure Authentication (`ingenious/client/azure/`)
- Service principal support
- Managed identity
- Interactive browser auth
- Connection string auth

## Type Safety

Strict mypy configuration with:
- `strict = true` for core library
- Relaxed rules for tests and certain legacy modules
- Run `uv run mypy .` before submitting changes

## Package Features

Install groups (via `uv add "ingenious[group]"`):
- `minimal` - Basic API functionality only
- `core` - Common production functionality
- `auth` - Authentication and security
- `azure` - Azure cloud integrations
- `ai` - AI and agent functionality
- `database` - Database connectivity
- `ui` - Web UI with Flask
- `standard` - Core + auth + AI + database
- `azure-full` - Full Azure integration (recommended)
- `full` - All features including document processing and ML
