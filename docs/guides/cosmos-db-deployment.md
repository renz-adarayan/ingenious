# Cosmos DB Deployment Guide

This guide provides step-by-step instructions for transitioning from local SQLite to Azure Cosmos DB for chat history persistence in **Ingenious**.

## Prerequisites

- Azure CLI installed and authenticated
- Azure OpenAI resource (required)
- Azure subscription with appropriate permissions

## Minimal Azure Provisioning

### 1. Create Cosmos DB Account (with Free Tier)

```bash
# Create Cosmos DB account with free tier (cheapest option)
az cosmosdb create \
  --name your-cosmos-account \
  --resource-group your-rg-name \
  --default-consistency-level Session \
  --locations regionName=westus2 failoverPriority=0 isZoneRedundant=false \
  --kind GlobalDocumentDB \
  --enable-free-tier true
```

### 2. Create Database

```bash
# Create SQL API database
az cosmosdb sql database create \
  --account-name your-cosmos-account \
  --resource-group your-rg-name \
  --name ingenious-db
```

### 3. Get Connection Information

```bash
# Get Cosmos DB keys
az cosmosdb keys list \
  --name your-cosmos-account \
  --resource-group your-rg-name

# Get endpoint URL
az cosmosdb show \
  --name your-cosmos-account \
  --resource-group your-rg-name \
  --query "documentEndpoint" \
  --output tsv
```

## Environment Configuration

### Update .env File

Change your chat history database configuration from SQLite to Cosmos DB:

| Variable | Description | Example |
|----------|-------------|---------|
| `INGENIOUS_CHAT_HISTORY__DATABASE_TYPE` | Database type | `cosmos` |
| `INGENIOUS_COSMOS_SERVICE__URI` | Cosmos DB endpoint | `https://your-cosmos-account.documents.azure.com:443/` |
| `INGENIOUS_COSMOS_SERVICE__DATABASE_NAME` | Database name | `ingenious-db` |
| `INGENIOUS_COSMOS_SERVICE__API_KEY` | Primary master key | `your-primary-master-key` |
| `INGENIOUS_COSMOS_SERVICE__AUTHENTICATION_METHOD` | Auth method | `token` |

### Complete .env Configuration

```bash
# Change database type from sqlite to cosmos
INGENIOUS_CHAT_HISTORY__DATABASE_TYPE=cosmos

# Add Cosmos DB configuration
INGENIOUS_COSMOS_SERVICE__URI=https://your-cosmos-account.documents.azure.com:443/
INGENIOUS_COSMOS_SERVICE__DATABASE_NAME=ingenious-db
INGENIOUS_COSMOS_SERVICE__API_KEY=your-primary-master-key-here
INGENIOUS_COSMOS_SERVICE__AUTHENTICATION_METHOD=token

# Optional: Keep memory path for temporary files
INGENIOUS_CHAT_HISTORY__MEMORY_PATH=./.tmp
```

## Automatic Container Creation

Ingenious automatically creates the following containers in Cosmos DB:

- `chat_history` - Main chat messages
- `chat_history_summary` - Memory summaries
- `users` - User information
- `threads` - Thread metadata
- `steps` - Workflow steps
- `elements` - UI elements
- `feedbacks` - User feedback

No manual container creation is required.

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
  "timestamp": "2025-08-29T07:14:46.522450",
  "response_time_ms": 2.7,
  "components": {
    "configuration": "ok",
    "profile": "ok"
  },
  "version": "1.0.0",
  "uptime": "available"
}
```

### 4. Test Workflow with Cosmos DB

```bash
# Test bike-insights workflow
echo '{
  "user_prompt": "{\"revision_id\": \"test-v1\", \"identifier\": \"test-001\", \"stores\": [{\"name\": \"Test Store\", \"location\": \"NSW\", \"bike_sales\": [{\"product_code\": \"MB-TREK-2021-XC\", \"quantity_sold\": 2, \"sale_date\": \"2023-04-01\", \"year\": 2023, \"month\": \"April\", \"customer_review\": {\"rating\": 4.5, \"comment\": \"Great bike\"}}], \"bike_stock\": []}]}",
  "conversation_flow": "bike-insights"
}' > test_cosmos.json

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d @test_cosmos.json
```

Successful response indicates:
- Cosmos DB connection for chat history persistence
- Automatic container creation and data storage
- Multi-agent workflow execution

## Migration from SQLite

To migrate existing chat history from SQLite to Cosmos DB:

1. **Backup existing data** (optional):
   ```bash
   cp ./.tmp/chat_history.db ./.tmp/chat_history_backup.db
   ```

2. **Update configuration** as described above

3. **Restart application** - Cosmos DB containers will be created automatically

4. **Verify migration** by testing workflows

Note: Direct data migration tools are not currently provided. For production migrations, consider implementing custom migration scripts.

## Troubleshooting

### Connection Issues
- Verify Cosmos DB account is provisioned and accessible
- Check API key is correct (primary master key)
- Ensure endpoint URL format is correct
- Verify network connectivity to Azure

### Authentication Issues
- Confirm `AUTHENTICATION_METHOD=token` for API key auth
- For production, consider using Azure AD authentication instead of API keys
- Check that API key has read/write permissions

### Container Creation Issues
- Ensure Cosmos DB account has sufficient permissions
- Verify free tier limits are not exceeded
- Check that database exists before running workflows

### Performance Considerations
- Cosmos DB free tier includes 1000 RU/s and 25GB storage
- Monitor RU consumption in Azure portal
- Consider upgrading to provisioned throughput for production workloads

## Cost Optimization

- **Cosmos DB Free Tier**: First 1000 RU/s and 25GB free per month
- **Additional usage**: ~$0.008 per 100 RU/s provisioned per hour
- **Storage**: ~$0.25/GB/month

Total cost for minimal usage: $0-5/month depending on throughput requirements.

For local development setup, see the [Getting Started Guide](../getting-started.md).
