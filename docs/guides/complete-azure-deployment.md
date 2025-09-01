# Complete Azure Deployment Guide

This guide provides step-by-step instructions for moving from local development to production Azure deployment with **Azure SQL** and **Azure Blob Storage**.

## Prerequisites

- Azure CLI installed and authenticated
- Azure OpenAI resource (required)
- Azure subscription with appropriate permissions

## Minimal Azure Provisioning

### 1. Create Resource Group

```bash
az group create --name your-rg-name --location eastus
```

### 2. Provision Azure SQL (Basic - 5 DTUs)

```bash
# Create SQL Server
SQL_PASSWORD=$(openssl rand -base64 32)
az sql server create \
  --name your-sql-server \
  --resource-group your-rg-name \
  --location westus2 \
  --admin-user ingenious \
  --admin-password "$SQL_PASSWORD"

# Create Database (Basic SKU - cheapest option)
az sql db create \
  --resource-group your-rg-name \
  --server your-sql-server \
  --name ingenious-db \
  --service-objective Basic

# Allow Azure services access
az sql server firewall-rule create \
  --resource-group your-rg-name \
  --server your-sql-server \
  --name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

# Allow your IP (replace with your actual IP)
MY_IP=$(curl -s ipinfo.io/ip)
az sql server firewall-rule create \
  --resource-group your-rg-name \
  --server your-sql-server \
  --name AllowMyIP \
  --start-ip-address "$MY_IP" \
  --end-ip-address "$MY_IP"
```

### 3. Provision Azure Blob Storage

```bash
# Create storage account (Standard_LRS - cheapest option)
az storage account create \
  --name yourblobstorage \
  --resource-group your-rg-name \
  --location westus2 \
  --sku Standard_LRS \
  --kind StorageV2

# Create prompts container
az storage container create \
  --account-name yourblobstorage \
  --name prompts \
  --auth-mode login
```

## Environment Configuration

### Update .env File

Add these environment variables to your `.env` file:

| Variable | Description | Example |
|----------|-------------|---------|
| `INGENIOUS_CHAT_HISTORY__DATABASE_TYPE` | Database type | `azuresql` |
| `INGENIOUS_AZURE_SQL_SERVICES__DATABASE_CONNECTION_STRING` | SQL connection string | `Driver={ODBC Driver 18 for SQL Server};Server=tcp:your-sql-server.database.windows.net,1433;Database=ingenious-db;Uid=ingenious;Pwd=YOUR_PASSWORD;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;` |
| `INGENIOUS_AZURE_SQL_SERVICES__DATABASE_NAME` | Database name | `ingenious-db` |
| `INGENIOUS_AZURE_SQL_SERVICES__TABLE_NAME` | Chat history table | `chat_history` |
| `INGENIOUS_CHAT_SERVICE__ENABLE_BUILTIN_WORKFLOWS` | **Production Security**: Disable built-in workflows | `false` |
| `INGENIOUS_FILE_STORAGE__REVISIONS__ENABLE` | Enable file storage | `true` |
| `INGENIOUS_FILE_STORAGE__REVISIONS__STORAGE_TYPE` | Storage type | `azure` |
| `INGENIOUS_FILE_STORAGE__REVISIONS__CONTAINER_NAME` | Container name | `prompts` |
| `INGENIOUS_FILE_STORAGE__REVISIONS__PATH` | Path prefix | `./` |
| `INGENIOUS_FILE_STORAGE__REVISIONS__URL` | Storage account URL | `https://yourblobstorage.blob.core.windows.net` |
| `INGENIOUS_FILE_STORAGE__REVISIONS__TOKEN` | Connection string | `DefaultEndpointsProtocol=https;AccountName=yourblobstorage;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net` |

### Get Storage Account Key

```bash
az storage account keys list \
  --account-name yourblobstorage \
  --resource-group your-rg-name \
  --output table
```

## Upload Prompt Templates

Upload your prompt templates to the correct blob path:

```bash
# Upload templates for revision "test-v1"
for file in templates/prompts/quickstart-1/*.jinja; do
  filename=$(basename "$file")
  az storage blob upload \
    --account-name yourblobstorage \
    --container-name prompts \
    --name "templates/prompts/test-v1/$filename" \
    --file "$file" \
    --auth-mode key \
    --overwrite
done
```

## Production Security Configuration

### Disable Built-in Workflows (Recommended for Production)

For production deployments, you can disable the built-in workflows (`classification-agent`, `knowledge-base-agent`, `sql-manipulation-agent`) to expose only your custom workflows from `ingenious_extensions`:

```bash
# Add to your .env file for production security
INGENIOUS_CHAT_SERVICE__ENABLE_BUILTIN_WORKFLOWS=false
```

**Testing the Security Setting:**

```bash
# Test that built-in workflows are blocked (should return error)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $(echo -n 'username:password' | base64)" \
  -d '{"user_prompt": "Test", "conversation_flow": "classification-agent", "thread_id": "test"}'

# Expected error response:
# {"detail":"Built-in workflow 'classification-agent' is disabled. Set INGENIOUS_CHAT_SERVICE__ENABLE_BUILTIN_WORKFLOWS=true to enable built-in workflows, or use a custom workflow from ingenious_extensions."}

# Custom workflows still work:
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $(echo -n 'username:password' | base64)" \
  -d '{"user_prompt": "Test", "conversation_flow": "your-custom-workflow", "thread_id": "test"}'
```

## Verification

### 1. Validate Configuration

```bash
uv run ingen validate
```

Expected output: `All validations passed! Your Ingenious setup is ready.`

### 2. Start Server

```bash
uv run ingen serve --port 8000
```

### 3. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-29T06:06:02.916027",
  "response_time_ms": 2.06,
  "components": {
    "configuration": "ok",
    "profile": "ok"
  },
  "version": "1.0.0",
  "uptime": "available"
}
```

### 4. Test Workflow with Azure Integrations

```bash
# Test bike-insights workflow (requires prompt templates uploaded)
echo '{
  "user_prompt": "{\"revision_id\": \"test-v1\", \"identifier\": \"test-001\", \"stores\": [{\"name\": \"Test Store\", \"location\": \"NSW\", \"bike_sales\": [{\"product_code\": \"MB-TREK-2021-XC\", \"quantity_sold\": 2, \"sale_date\": \"2023-04-01\", \"year\": 2023, \"month\": \"April\", \"customer_review\": {\"rating\": 4.5, \"comment\": \"Great bike\"}}], \"bike_stock\": []}]}",
  "conversation_flow": "bike-insights"
}' > test_azure.json

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d @test_azure.json
```

Successful response indicates:
- Azure SQL connection for chat history persistence
- Azure Blob prompt template loading
- Multi-agent workflow execution

## Troubleshooting

### SQL Connection Issues
- Verify firewall rules allow your IP
- Check connection string format
- Ensure database exists

### Blob Storage Issues
- Verify storage account key is correct
- Ensure prompt templates uploaded to correct path: `templates/prompts/{revision_id}/`
- Check container permissions

### Rate Limiting
Azure OpenAI free tier (S0) has token limits. Consider upgrading to Pay-as-you-go for production use.

## Cost Optimization

- **Azure SQL Basic (5 DTUs)**: ~$5/month
- **Storage Account (Standard_LRS)**: ~$0.02/GB/month
- **Azure OpenAI**: Pay per token usage

Total minimal cost: ~$5-10/month for light usage.

For detailed local setup, see the [Getting Started Guide](../getting-started.md).
