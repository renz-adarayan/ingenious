---
title: "Complete Azure Deployment Guide"
layout: single
permalink: /guides/complete-azure-deployment/
sidebar:
  nav: "docs"
toc: true
toc_label: "Setup Steps"
toc_icon: "database"
---

This guide provides step-by-step instructions for deploying **Ingenious** with full
Azure integration: Azure OpenAI (chat + embeddings), Azure AI Search, Azure SQL for
chat history, and Azure Blob Storage for prompt/data management. It also covers
validating your configuration and testing the included **bike-insights** template.

---

## Overview

This deployment includes:

- **Bike-Insights workflow** (created by `uv run ingen init`)
- **Azure OpenAI**: at least one **chat** model deployment and one **embedding**
  model deployment
- **Azure AI Search**: knowledge retrieval (semantic optional)
- **Azure SQL Database**: persistent chat history
- **Azure Blob Storage**: prompt + data storage (token-based auth)
- **REST API**: `/api/v1/chat` and `/api/v1/prompts/*`

---

## Prerequisites

### Required Azure resources

- **Azure OpenAI** with:
  - One chat model (e.g., `gpt-4.1-mini`)
  - One embeddings model (e.g., `text-embedding-3-small`)
- **Azure AI Search** service (required for Azure-backed KB mode)
- **Azure SQL Database** (ODBC connectivity)
- **Azure Storage Account** with **Blob** service enabled

### Local requirements

- **Python 3.13+** (earlier versions are not supported)
- **[uv package manager](https://docs.astral.sh/uv/)**
- **ODBC Driver 18 for SQL Server**
- **curl**

---

## Step-by-Step Deployment

### Step 1: Install Ingenious

```bash
# Initialize a new uv project (or use an existing directory)
uv init

# Full Azure integration (core, auth, azure, ai, database, ui)
uv add "ingenious[azure-full]"
```

### Step 2: Initialize Project

```bash
# Generates a template project with the bike-insights workflow
uv run ingen init
```

This creates, among other files:

- `ingenious_extensions/` — includes the **bike-insights** workflow
- (You will create your own `.env` in the next step)

---

### Step 3: Configure Credentials (`.env`)

Create a `.env` file in the **project root** and paste the example below. Then
replace all placeholder values with your real credentials.

> **Tip:** Quoting is recommended for safety (values with spaces/special chars).

#### 📄 Environment example (`.env`)

```bash
# ---------- Azure OpenAI ----------
AZURE_OPENAI_ENDPOINT="https://your-aoai-resource.openai.azure.com/"
AZURE_OPENAI_KEY="YOUR_AZURE_OPENAI_KEY"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
AZURE_OPENAI_GENERATION_DEPLOYMENT="gpt-4.1-mini"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"

# Ingenious model slots (must be token auth)
INGENIOUS_MODELS__0__MODEL="${AZURE_OPENAI_GENERATION_DEPLOYMENT}"
INGENIOUS_MODELS__0__API_TYPE="rest"
INGENIOUS_MODELS__0__API_VERSION="${AZURE_OPENAI_API_VERSION}"
INGENIOUS_MODELS__0__DEPLOYMENT="${AZURE_OPENAI_GENERATION_DEPLOYMENT}"
INGENIOUS_MODELS__0__API_KEY="${AZURE_OPENAI_KEY}"
INGENIOUS_MODELS__0__BASE_URL="${AZURE_OPENAI_ENDPOINT}"
INGENIOUS_MODELS__0__AUTHENTICATION_METHOD="token"

INGENIOUS_MODELS__1__MODEL="${AZURE_OPENAI_EMBEDDING_DEPLOYMENT}"
INGENIOUS_MODELS__1__API_TYPE="rest"
INGENIOUS_MODELS__1__API_VERSION="${AZURE_OPENAI_API_VERSION}"
INGENIOUS_MODELS__1__DEPLOYMENT="${AZURE_OPENAI_EMBEDDING_DEPLOYMENT}"
INGENIOUS_MODELS__1__API_KEY="${AZURE_OPENAI_KEY}"
INGENIOUS_MODELS__1__BASE_URL="${AZURE_OPENAI_ENDPOINT}"
INGENIOUS_MODELS__1__AUTHENTICATION_METHOD="token"

# (optional local cache for chat memory when SQL isn't configured)
INGENIOUS_CHAT_HISTORY__MEMORY_PATH="./.tmp"

# ---------- Azure AI Search ----------
AZURE_SEARCH_ENDPOINT="https://your-search-service.search.windows.net"
AZURE_SEARCH_KEY="YOUR_AZURE_SEARCH_API_KEY"
AZURE_SEARCH_INDEX_NAME="your-index-name"
AZURE_SEARCH_SEMANTIC_CONFIG="your-semantic-config"

INGENIOUS_AZURE_SEARCH_SERVICES__0__SERVICE="default"
INGENIOUS_AZURE_SEARCH_SERVICES__0__ENDPOINT="${AZURE_SEARCH_ENDPOINT}"
INGENIOUS_AZURE_SEARCH_SERVICES__0__KEY="${AZURE_SEARCH_KEY}"
INGENIOUS_AZURE_SEARCH_SERVICES__0__INDEX_NAME="${AZURE_SEARCH_INDEX_NAME}"

# (optional tuning; keep if you use semantic)
INGENIOUS_AZURE_SEARCH_SERVICES__0__USE_SEMANTIC_RANKING="1"
INGENIOUS_AZURE_SEARCH_SERVICES__0__SEMANTIC_CONFIGURATION_NAME="${AZURE_SEARCH_SEMANTIC_CONFIG}"
INGENIOUS_AZURE_SEARCH_SERVICES__0__ID_FIELD="id"
INGENIOUS_AZURE_SEARCH_SERVICES__0__CONTENT_FIELD="content"
INGENIOUS_AZURE_SEARCH_SERVICES__0__VECTOR_FIELD="vector"
INGENIOUS_AZURE_SEARCH_SERVICES__0__TOP_K_RETRIEVAL="5"
INGENIOUS_AZURE_SEARCH_SERVICES__0__TOP_N_FINAL="10"

# (optional; harmless if present)
INGENIOUS_SEARCH__PROVIDER="azure"
INGENIOUS_SEARCH__POLICY="azure_only"

# ---------- KB tuning ----------
KB_MODE="direct"
KB_TOP_K="5"
KB_USE_SEMANTIC_RANKING="1"
KB_POLICY="azure_only"
KB_WRITE_CONFIG_SNAPSHOT="1"

# ---------- Chat history (Azure SQL) ----------
# Keep on ONE line; quoting protects special chars
AZURE_SQL_CONNECTION_STRING="Driver={ODBC Driver 18 for SQL Server};Server=tcp:yourserver.database.windows.net,1433;Database=yourdatabase;Uid=your-username;Pwd=YOUR_STRONG_PASSWORD;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
INGENIOUS_CHAT_HISTORY__DATABASE_TYPE="azuresql"
INGENIOUS_CHAT_HISTORY__DATABASE_NAME="yourdatabase"
INGENIOUS_CHAT_HISTORY__DATABASE_CONNECTION_STRING="${AZURE_SQL_CONNECTION_STRING}"

# ---------- Azure Blob ----------
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=yourstorageaccount;AccountKey=YOUR_STORAGE_ACCOUNT_KEY;EndpointSuffix=core.windows.net"

AZURE_STORAGE_REVISIONS_URL="https://yourstorageaccount.blob.core.windows.net/prompts/"
INGENIOUS_FILE_STORAGE__REVISIONS__ENABLE="true"
INGENIOUS_FILE_STORAGE__REVISIONS__STORAGE_TYPE="azure"
INGENIOUS_FILE_STORAGE__REVISIONS__CONTAINER_NAME="prompts"
INGENIOUS_FILE_STORAGE__REVISIONS__PATH="ingenious-files"
INGENIOUS_FILE_STORAGE__REVISIONS__ADD_SUB_FOLDERS="true"
INGENIOUS_FILE_STORAGE__REVISIONS__URL="${AZURE_STORAGE_REVISIONS_URL}"
INGENIOUS_FILE_STORAGE__REVISIONS__TOKEN="${AZURE_STORAGE_CONNECTION_STRING}"
INGENIOUS_FILE_STORAGE__REVISIONS__AUTHENTICATION_METHOD="token"

AZURE_STORAGE_DATA_URL="https://yourstorageaccount.blob.core.windows.net/data/"
INGENIOUS_FILE_STORAGE__DATA__ENABLE="true"
INGENIOUS_FILE_STORAGE__DATA__STORAGE_TYPE="azure"
INGENIOUS_FILE_STORAGE__DATA__CONTAINER_NAME="data"
INGENIOUS_FILE_STORAGE__DATA__PATH="ingenious-files"
INGENIOUS_FILE_STORAGE__DATA__ADD_SUB_FOLDERS="true"
INGENIOUS_FILE_STORAGE__DATA__URL="${AZURE_STORAGE_DATA_URL}"
INGENIOUS_FILE_STORAGE__DATA__TOKEN="${AZURE_STORAGE_CONNECTION_STRING}"
INGENIOUS_FILE_STORAGE__DATA__AUTHENTICATION_METHOD="token"

# ---------- Web ----------
INGENIOUS_WEB_CONFIGURATION__IP_ADDRESS="0.0.0.0"
INGENIOUS_WEB_CONFIGURATION__PORT="8080"

# ---------- Debug (optional) ----------
DEBUG_AZURE_CONFIG="1"
```

> **Important (SQL connection strings):** Use the value from the **ODBC** tab in the Azure
> Portal. ADO.NET/JDBC formats are incompatible and will cause connection errors.

> **Gotcha (Azure Blob auth):** When `AUTHENTICATION_METHOD=token`, the **`TOKEN`** value
> must be your full Azure Storage **connection string**. Having only
> `AZURE_STORAGE_CONNECTION_STRING` defined is **not** sufficient.

---

### Step 4: Configure Azure Blob Storage Integration

If you skipped the Blob section above, ensure you add both **REVISIONS** and **DATA**
blocks (token auth) to your `.env`. These back your prompts and any workflow data.

---

### Step 5: Install ODBC Driver (if not already installed)

#### macOS

```bash
brew tap microsoft/mssql-release
brew install msodbcsql18

# Verify installation
odbcinst -q -d | grep "ODBC Driver 18"
```

#### Ubuntu/Debian

```bash
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list \
  > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install msodbcsql18
```

---

### Step 6: Upload Prompt Templates to Azure Blob Storage

For the **bike-insights** workflow, use the provided script:

```bash
# Ensure server is running first
uv run ingen serve --port 8000 &

# Upload bike-insights templates
uv run python scripts/upload_bike_templates.py
```

---

### Step 7: Validate Configuration

```bash
uv run ingen validate
```

**If validation fails with port conflicts:** change
`INGENIOUS_WEB_CONFIGURATION__PORT` in your `.env`, then re-run:

```bash
uv run ingen validate
```

> **⚠️ BREAKING CHANGE**
> Ingenious now uses **pydantic-settings** for configuration via environment
> variables. Legacy YAML files (`config.yml`, `profiles.yml`) are **no longer
> supported**. Migrate with:
>
> ```bash
> uv run python scripts/migrate_config.py --yaml-file config.yml --output .env
> uv run python scripts/migrate_config.py --yaml-file profiles.yml --output .env.profiles
> ```

---

### Step 8: Start the Server

```bash
# Recommended for development
uv run ingen serve --port 8000

# Additional options:
# --host 0.0.0.0         # Bind host (default: 0.0.0.0)
# --port                 # Port to bind (default: 80 or $WEB_PORT env var)
# --config config.yml    # Legacy config (deprecated — use environment variables)
# --profile production   # Legacy profile (deprecated — use environment variables)
```

---

### Step 9: Test the Deployment

#### Server health

```bash
curl http://localhost:8000/api/v1/health
```

#### Test **bike-insights** workflow (recommended: use `parameters`)

```bash
cat > test_bike_insights.json <<'JSON'
{
  "user_prompt": "Analyze these store sales and summarize insights.",
  "conversation_flow": "bike-insights",
  "parameters": {
    "revision_id": "test-v1",
    "identifier": "test-001",
    "stores": [
      {
        "name": "Test Store",
        "location": "NSW",
        "bike_sales": [
          {
            "product_code": "MB-TREK-2021-XC",
            "quantity_sold": 2,
            "sale_date": "2023-04-01",
            "year": 2023,
            "month": "April",
            "customer_review": { "rating": 4.5, "comment": "Great bike" }
          }
        ],
        "bike_stock": []
      }
    ]
  }
}
JSON

curl -sS -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d @test_bike_insights.json
```

#### Prompts API sanity checks

```bash
# List prompt templates
curl "http://localhost:8000/api/v1/prompts/list/quickstart-1"

# View a prompt template
curl "http://localhost:8000/api/v1/prompts/view/quickstart-1/bike_lookup_agent_prompt.jinja"

# Update a prompt template
curl -X POST "http://localhost:8000/api/v1/prompts/update/quickstart-1/bike_lookup_agent_prompt.jinja" \
  -H "Content-Type: application/json" \
  -d '{"content": "### UPDATED ROLE\nYou are an updated bike lookup agent...\n"}'
```

---

## Troubleshooting

- **Azure AI Search policy errors**
  `PreflightError: [azure_search] policy: Azure Search is required…`
  → Ensure `INGENIOUS_AZURE_SEARCH_SERVICES__0__...` variables are set and
  `INGENIOUS_SEARCH__POLICY="azure_only"` (or set KB policy accordingly).

- **404/401/403 from Azure Search**
  → Check `INDEX_NAME`, `ENDPOINT`, and `KEY`.

- **Sanity check your Search index** (replace placeholders):

  ```bash
  curl -sD- -H "api-key: <your-search-key>" \
    "https://<your-service>.search.windows.net/indexes/<your-index>?api-version=2023-11-01"
  # Expect: HTTP/1.1 200 OK
  ```

- **Blob auth issues**
  → Confirm each storage block sets `AUTHENTICATION_METHOD="token"` and
  `TOKEN="${AZURE_STORAGE_CONNECTION_STRING}".`

- **Port conflicts**
  → Change `INGENIOUS_WEB_CONFIGURATION__PORT` or pass `--port` to `serve`.

---

## Security Notes

- Do **NOT** commit `.env` files to source control.
- Redact secrets in logs. Avoid printing `api_key`, `key`, `Authorization`,
  `password`, `token`, etc.
- Rotate keys immediately if they are exposed.

---

## Documentation

For detailed documentation, see the
[docs](https://insight-services-apac.github.io/ingenious/).

## Contributing

Contributions are welcome! Please see
[CONTRIBUTING.md](https://github.com/Insight-Services-APAC/ingenious/blob/main/CONTRIBUTING.md).

## License

This project is licensed under the terms specified in the
[LICENSE](https://github.com/Insight-Services-APAC/ingenious/blob/main/LICENSE)
file.
