# Azure AI Search Deployment Guide

This guide provides step-by-step instructions for migrating from local ChromaDB to Azure AI Search for knowledge-base-agent workflows in **Ingenious**.

## Prerequisites

- Azure CLI installed and authenticated
- Azure OpenAI resource with separate embedding and chat deployments
- Azure subscription with appropriate permissions

## Minimal Azure Provisioning

### 1. Create Azure AI Search Service

```bash
# Create AI Search service (Basic tier - cheapest option after free tier)
# Note: Free tier is limited to one per subscription
az search service create \
  --name your-search-service \
  --resource-group your-rg-name \
  --sku basic \
  --location eastus
```

### 2. Get Search Service Keys

```bash
# Get admin keys for the search service
az search admin-key show \
  --service-name your-search-service \
  --resource-group your-rg-name
```

### 3. Create Required Azure OpenAI Deployments

Azure AI Search requires separate deployments for embeddings and generation:

```bash
# Create embedding deployment (required for vector search)
az cognitiveservices account deployment create \
  --name your-openai-resource \
  --resource-group your-rg-name \
  --deployment-name text-embedding-3-small-deployment \
  --model-name text-embedding-3-small \
  --model-version "1" \
  --model-format OpenAI \
  --sku-capacity 1 \
  --sku-name "Standard"

# Create or verify chat deployment exists
az cognitiveservices account deployment create \
  --name your-openai-resource \
  --resource-group your-rg-name \
  --deployment-name gpt-4o-mini-deployment \
  --model-name gpt-4o-mini \
  --model-version "2024-07-18" \
  --model-format OpenAI \
  --sku-capacity 1 \
  --sku-name "Standard"
```

## Environment Configuration

### Update .env File

Migrate from local ChromaDB to Azure AI Search:

| Variable | Description | Example |
|----------|-------------|---------|
| `KB_POLICY` | Backend selection policy | `azure_only` or `prefer_azure` |
| `INGENIOUS_AZURE_SEARCH_SERVICES__0__ENDPOINT` | Search service endpoint | `https://your-search-service.search.windows.net` |
| `INGENIOUS_AZURE_SEARCH_SERVICES__0__KEY` | Search admin key | `your-search-admin-key` |
| `INGENIOUS_AZURE_SEARCH_SERVICES__0__INDEX_NAME` | Search index name | `knowledge-base` |
| `INGENIOUS_AZURE_SEARCH_SERVICES__0__EMBEDDING_DEPLOYMENT_NAME` | Embedding model deployment | `text-embedding-3-small-deployment` |
| `INGENIOUS_AZURE_SEARCH_SERVICES__0__GENERATION_DEPLOYMENT_NAME` | Generation model deployment | `gpt-4o-mini-deployment` |
| `INGENIOUS_AZURE_SEARCH_SERVICES__0__OPENAI_ENDPOINT` | Azure OpenAI endpoint | `https://eastus.api.cognitive.microsoft.com/` |
| `INGENIOUS_AZURE_SEARCH_SERVICES__0__OPENAI_KEY` | Azure OpenAI key | `your-openai-key` |
| `INGENIOUS_AZURE_SEARCH_SERVICES__0__OPENAI_VERSION` | API version | `2024-12-01-preview` |

### Complete .env Configuration

```bash
# Knowledge base policy - choose one:
KB_POLICY=azure_only          # Use only Azure AI Search (strict)
# KB_POLICY=prefer_azure      # Prefer Azure, fallback to local ChromaDB
# KB_POLICY=prefer_local      # Prefer local ChromaDB, fallback to Azure
# KB_POLICY=local_only        # Use only local ChromaDB

# Optional: Enable fallback when Azure returns empty results
KB_FALLBACK_ON_EMPTY=true

# Knowledge base search parameters
KB_TOPK_DIRECT=3              # Number of results for direct mode
KB_TOPK_ASSIST=5              # Number of results for assist mode
KB_MODE=direct                # Search mode: direct or assist

# Azure AI Search Service Configuration
INGENIOUS_AZURE_SEARCH_SERVICES__0__ENDPOINT=https://your-search-service.search.windows.net
INGENIOUS_AZURE_SEARCH_SERVICES__0__KEY=your-search-admin-key
INGENIOUS_AZURE_SEARCH_SERVICES__0__INDEX_NAME=knowledge-base

# Azure OpenAI Deployments (MUST be different)
INGENIOUS_AZURE_SEARCH_SERVICES__0__EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small-deployment
INGENIOUS_AZURE_SEARCH_SERVICES__0__GENERATION_DEPLOYMENT_NAME=gpt-4o-mini-deployment

# Azure OpenAI Configuration for Search Service
INGENIOUS_AZURE_SEARCH_SERVICES__0__OPENAI_ENDPOINT=https://your-region.api.cognitive.microsoft.com/
INGENIOUS_AZURE_SEARCH_SERVICES__0__OPENAI_KEY=your-openai-key
INGENIOUS_AZURE_SEARCH_SERVICES__0__OPENAI_VERSION=2024-12-01-preview
```

## Create Search Index and Upload Documents

### 1. Create Search Index Schema

Create a basic search index with required fields:

```bash
# Create index schema file
cat > index_schema.json << 'EOF'
{
  "name": "knowledge-base",
  "fields": [
    {
      "name": "id",
      "type": "Edm.String",
      "key": true,
      "searchable": false,
      "filterable": true,
      "retrievable": true,
      "sortable": true,
      "facetable": false
    },
    {
      "name": "content",
      "type": "Edm.String",
      "searchable": true,
      "filterable": false,
      "retrievable": true,
      "sortable": false,
      "facetable": false,
      "analyzer": "standard.lucene"
    },
    {
      "name": "title",
      "type": "Edm.String",
      "searchable": true,
      "filterable": false,
      "retrievable": true,
      "sortable": false,
      "facetable": false,
      "analyzer": "standard.lucene"
    }
  ]
}
EOF

# Create the index
curl -X POST "https://your-search-service.search.windows.net/indexes?api-version=2023-11-01" \
  -H "Content-Type: application/json" \
  -H "api-key: your-search-admin-key" \
  -d @index_schema.json
```

### 2. Upload Sample Documents

```bash
# Create sample documents
cat > sample_documents.json << 'EOF'
{
  "value": [
    {
      "id": "1",
      "title": "Ingenious Setup Guide",
      "content": "Ingenious is a multi-agent AI framework that allows you to quickly set up APIs for AI agents. Prerequisites include Python 3.13+, OpenAI API key or Azure OpenAI credentials, and UV package manager. Installation steps: 1. Initialize UV project with uv init, 2. Install Ingenious with uv add ingenious[azure-full], 3. Initialize project with uv run ingen init, 4. Configure environment variables in .env file, 5. Start server with uv run ingen serve --port 8000."
    },
    {
      "id": "2",
      "title": "Azure Integration Guide",
      "content": "Ingenious supports Azure SQL, Cosmos DB, Azure Blob Storage, and Azure AI Search. For chat history persistence, you can use SQLite for local development or Azure SQL/Cosmos DB for production. Azure Blob Storage can be used for prompt template storage. Azure AI Search enables advanced knowledge base search capabilities with semantic ranking and vector search."
    }
  ]
}
EOF

# Upload documents to the index
curl -X POST "https://your-search-service.search.windows.net/indexes/knowledge-base/docs/index?api-version=2023-11-01" \
  -H "Content-Type: application/json" \
  -H "api-key: your-search-admin-key" \
  -d @sample_documents.json
```

## Policy Configuration

Azure AI Search supports flexible backend selection policies:

### KB_POLICY Options

- **azure_only**: Use only Azure AI Search (strict mode)
  - Fails if Azure is misconfigured or unavailable
  - Recommended for production environments

- **prefer_azure**: Prefer Azure AI Search, fallback to local ChromaDB
  - Attempts Azure first, falls back to local on failure
  - Good for development/testing environments

- **prefer_local**: Prefer local ChromaDB, fallback to Azure AI Search
  - Uses local ChromaDB first, Azure as backup
  - Useful during development phase

- **local_only**: Use only local ChromaDB
  - Ignores Azure configuration entirely
  - Development mode

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

### 4. Test Knowledge Base Agent

```bash
# Test knowledge-base-agent workflow
echo '{
  "user_prompt": "Search for documentation about setup",
  "conversation_flow": "knowledge-base-agent"
}' > test_knowledge_azure.json

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d @test_knowledge_azure.json
```

Expected successful response with Azure AI Search results from your uploaded documents.

## Migration Strategies

### Strategy 1: Direct Migration (azure_only)

For immediate production deployment:

```bash
# Set strict Azure-only policy
KB_POLICY=azure_only

# Ensure all Azure AI Search configuration is complete
# Upload your existing knowledge base documents to Azure AI Search
# Test thoroughly before going live
```

### Strategy 2: Gradual Migration (prefer_azure)

For safer migration with fallback:

```bash
# Set prefer Azure with fallback
KB_POLICY=prefer_azure
KB_FALLBACK_ON_EMPTY=true

# Keep existing .tmp/knowledge_base/ directory as backup
# Azure AI Search will be tried first, local ChromaDB as fallback
# Gradually move content to Azure AI Search
```

### Strategy 3: Hybrid Approach (prefer_local)

For development with Azure testing:

```bash
# Set prefer local with Azure backup
KB_POLICY=prefer_local

# Keep developing with local ChromaDB
# Use Azure AI Search for testing production scenarios
```

## Troubleshooting

### Deployment Configuration Issues

**Error**: "Embedding and chat deployments must not be the same"

**Solution**: Ensure you have separate Azure OpenAI deployments:
- Embedding deployment: `text-embedding-3-small-deployment`
- Generation deployment: `gpt-4o-mini-deployment` (or different model)

```bash
# Verify deployments are different
az cognitiveservices account deployment list \
  --name your-openai-resource \
  --resource-group your-rg-name \
  --output table
```

### Index Not Found Issues

**Error**: "The index 'your-index-name' was not found"

**Solution**:
1. Create the index using the REST API as shown above
2. Verify index exists:
   ```bash
   curl "https://your-search-service.search.windows.net/indexes/knowledge-base?api-version=2023-11-01" \
     -H "api-key: your-search-admin-key"
   ```

### Connection Issues

- Verify Azure AI Search service is running and accessible
- Check admin key is correct (not query key)
- Ensure search service endpoint format is correct
- Confirm network connectivity to Azure

### Policy Fallback Issues

If Azure fails but local fallback doesn't work:
- Check `.tmp/knowledge_base/` directory exists with documents
- Verify ChromaDB dependencies are installed
- Ensure KB_FALLBACK_ON_EMPTY=true for prefer_azure policy

## Cost Optimization

### Azure AI Search Pricing

- **Free Tier**: One per subscription, basic search capabilities
- **Basic Tier**: ~$15/month, 2GB storage, includes semantic search
- **Standard S1**: ~$250/month, production-grade with high availability

### Azure OpenAI Deployments

- **text-embedding-3-small**: ~$0.00002 per 1K tokens
- **gpt-4o-mini**: ~$0.000075 per 1K tokens (generation)

Total minimal cost for light usage: ~$15-20/month with Basic search tier.

## Advanced Features

When properly configured, Azure AI Search provides:

- **Semantic Search**: Natural language understanding for better relevance
- **Vector Search**: Embedding-based similarity search
- **Hybrid Search**: Combines keyword and vector search
- **Dynamic Alpha Tuning (DAT)**: Optimizes search result fusion
- **Reranking**: Improves result ordering with semantic models

For local development setup, see the [Getting Started Guide](../getting-started.md).
