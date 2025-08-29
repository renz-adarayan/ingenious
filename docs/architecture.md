# Architecture Overview

This document describes the high-level architecture of Insight Ingenious, an enterprise-grade Python library designed for quickly setting up APIs to interact with AI Agents with comprehensive Azure service integrations and debugging capabilities.

## System Architecture

Insight Ingenious is architected as a production-ready library with enterprise-grade features including seamless Azure service integrations, robust debugging tools, and extensive customization capabilities. The system consists of the following main components:

```mermaid
graph TB
    subgraph ClientLayer ["Client Layer"]
        API_CLIENT["API Clients<br/>External Applications"]
        DOCS["API Documentation<br/>Swagger/OpenAPI"]
    end

    subgraph APIGateway ["API Gateway"]
        API["FastAPI<br/>REST Endpoints"]
        AUTH["Authentication<br/>& Authorization"]
    end

    subgraph CoreEngine ["Core Engine"]
        AGENT_SERVICE["Agent Service<br/>Conversation Manager"]
        CONVERSATION_FLOWS["Conversation Flows<br/>Workflow Orchestrator"]
        LLM_SERVICE["LLM Service<br/>Azure OpenAI Integration"]
    end

    subgraph ExtensionLayer ["Extension Layer"]
        CUSTOM_AGENTS["Custom Agents<br/>Domain Specialists"]
        PATTERNS["Conversation Patterns<br/>Workflow Templates"]
        TOOLS["Custom Tools<br/>External Integrations"]
    end

    subgraph StorageLayer ["Storage Layer"]
        CONFIG["Configuration<br/>Environment Variables"]
        HISTORY["Chat History<br/>SQLite/Azure SQL"]
        FILES["File Storage<br/>Local/Azure Blob"]
    end

    subgraph ExternalServices ["External Services"]
        AZURE["Azure OpenAI<br/>GPT Models"]
        EXTERNAL_API["External APIs<br/>Data Sources"]
    end

    API_CLIENT --> API
    API_CLIENT --> DOCS
    API --> AUTH
    AUTH --> AGENT_SERVICE
    AGENT_SERVICE --> CONVERSATION_FLOWS
    CONVERSATION_FLOWS --> LLM_SERVICE
    AGENT_SERVICE --> CUSTOM_AGENTS
    CUSTOM_AGENTS --> PATTERNS
    PATTERNS --> TOOLS
    AGENT_SERVICE --> CONFIG
    AGENT_SERVICE --> HISTORY
    AGENT_SERVICE --> FILES
    LLM_SERVICE --> AZURE
    TOOLS --> EXTERNAL_API

    classDef clientLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef apiLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef coreLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef extensionLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef storageLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef externalLayer fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class API_CLIENT,DOCS clientLayer
    class API,AUTH apiLayer
    class AGENT_SERVICE,CONVERSATION_FLOWS,LLM_SERVICE coreLayer
    class CUSTOM_AGENTS,PATTERNS,TOOLS extensionLayer
    class CONFIG,HISTORY,FILES storageLayer
    class AZURE,EXTERNAL_API externalLayer
```

## Core Components

### API Layer

**FastAPI Application**
- RESTful API endpoints for chat interactions
- OpenAPI/Swagger documentation
- Request/response validation with Pydantic models
- Error handling and standardized responses

**Authentication & Authorization**
- JWT token-based authentication
- Basic authentication support
- Role-based access control
- Secure credential management

### Core Engine

**Agent Service**
- Conversation management and routing
- Agent lifecycle management
- Context preservation across conversations
- Thread-safe execution

**Conversation Flows Service**
- Conversation flow orchestration
- Multi-agent coordination
- Workflow execution engine
- State management

**LLM Service**
- Azure OpenAI integration
- Model configuration and management
- Token usage tracking
- Response streaming support

### Extension Layer

**Custom Agents**
- Domain-specific AI agents
- Specialized task handlers
- Configurable behavior patterns
- Integration with external tools

**Conversation Patterns**
- Reusable workflow templates
- Multi-agent coordination patterns
- Sequential and parallel execution modes
- Error handling and retry logic

**Custom Tools**
- External API integrations
- Data source connectors
- Business logic plugins
- Utility functions

### Storage Layer

**Configuration Management**
- Environment-based configuration
- Pydantic settings validation
- Dynamic reconfiguration support
- Secure credential storage

**Chat History**
- Conversation persistence
- Flexible storage backends: SQLite for development, Azure SQL/Cosmos DB for production
- Query and retrieval capabilities
- Data retention policies

**File Storage**
- Dual storage support: Local filesystem for development, Azure Blob for production
- Version control for templates
- Binary file handling
- Secure access management

**Knowledge Base**
- Flexible search backends: ChromaDB for development, Azure AI Search for production
- Document indexing and retrieval
- Semantic search capabilities
- Content management

## Data Flow

### Request Processing Flow

```mermaid
sequenceDiagram
    participant Client as Client
    participant API as API
    participant Auth as Auth
    participant AgentService as Agent Service
    participant ConversationFlows as Conversation Flows
    participant LLM as LLM
    participant Storage as Storage

    Client->>+API: POST /api/v1/chat
    API->>+Auth: Validate credentials
    Auth-->>-API: Authentication result

    API->>+AgentService: Process conversation request
    AgentService->>+Storage: Load conversation history
    Storage-->>-AgentService: Historical context

    AgentService->>+ConversationFlows: Execute conversation flow
    ConversationFlows->>+LLM: Generate AI response
    LLM-->>-ConversationFlows: AI-generated content

    ConversationFlows-->>-AgentService: Flow execution result
    AgentService->>+Storage: Save conversation state
    Storage-->>-AgentService: Save confirmation
    AgentService-->>-API: Conversation response

    API-->>-Client: JSON response
```

### Configuration Loading

```mermaid
flowchart LR
    ENV["Environment Variables"] --> PYDANTIC["Pydantic Settings"]
    PYDANTIC --> VALIDATION["Configuration Validation"]
    VALIDATION --> SERVICES["Service Initialization"]
    SERVICES --> READY["System Ready"]

    VALIDATION -->|"Validation Error"| ERROR["Configuration Error"]
    ERROR --> EXIT["System Exit"]

    classDef success fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef error fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#2196f3,stroke-width:2px

    class ENV,PYDANTIC,VALIDATION,SERVICES process
    class READY success
    class ERROR,EXIT error
```

## Security Architecture

### Authentication Flow

```mermaid
flowchart TB
    CLIENT["Client Request"] --> AUTH_CHECK{"Authentication<br/>Required?"}
    AUTH_CHECK -->|"No"| PROCESS["Process Request"]
    AUTH_CHECK -->|"Yes"| VALIDATE{"Validate<br/>Credentials"}

    VALIDATE -->|"Invalid"| REJECT["401 Unauthorized"]
    VALIDATE -->|"Valid"| AUTHORIZE{"Check<br/>Authorization"}

    AUTHORIZE -->|"Denied"| FORBIDDEN["403 Forbidden"]
    AUTHORIZE -->|"Allowed"| PROCESS

    PROCESS --> RESPONSE["Return Response"]

    classDef success fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef error fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#2196f3,stroke-width:2px

    class CLIENT,PROCESS,RESPONSE process
    class AUTH_CHECK,VALIDATE,AUTHORIZE decision
    class REJECT,FORBIDDEN error
```

### Security Features

- **Credential Management**: Environment-based secrets management
- **Input Validation**: Pydantic model validation for all inputs
- **Rate Limiting**: Request throttling and abuse prevention
- **Audit Logging**: Comprehensive request and response logging
- **Error Handling**: Secure error responses without information leakage

## Scalability Considerations

### Horizontal Scaling

- **Stateless Design**: Request processing without server-side state
- **Database Connection Pooling**: Efficient database resource management
- **Async Processing**: Non-blocking I/O for high throughput
- **Load Balancing**: Multiple instance deployment support

### Performance Optimization

- **Caching**: Response caching and template caching
- **Connection Reuse**: HTTP client connection pooling
- **Lazy Loading**: On-demand resource initialization
- **Memory Management**: Efficient resource cleanup and garbage collection

## Extension Architecture

### Extension Discovery

```mermaid
flowchart TB
    DISCOVERY["Extension Discovery"] --> LOCAL{"Local Extensions<br/>Found?"}
    LOCAL -->|"Yes"| LOAD_LOCAL["Load Local Extensions"]
    LOCAL -->|"No"| TEMPLATE{"Template Extensions<br/>Found?"}

    TEMPLATE -->|"Yes"| LOAD_TEMPLATE["Load Template Extensions"]
    TEMPLATE -->|"No"| CORE["Load Core Extensions"]

    LOAD_LOCAL --> VALIDATE["Validate Extensions"]
    LOAD_TEMPLATE --> VALIDATE
    CORE --> VALIDATE

    VALIDATE --> REGISTER["Register Extensions"]
    REGISTER --> READY["Extensions Ready"]

    classDef discovery fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef success fill:#e8f5e8,stroke:#4caf50,stroke-width:2px

    class DISCOVERY discovery
    class LOCAL,TEMPLATE decision
    class LOAD_LOCAL,LOAD_TEMPLATE,CORE,VALIDATE,REGISTER process
    class READY success
```

### Extension Interface

Extensions implement the standardized interface:

- **IConversationFlow**: Conversation workflow interface (defined in `ingenious/services/chat_services/multi_agent/service.py`)
  - Required method: `get_conversation_response()`
  - Auto-discovered by name match with `conversation_flow` parameter
  - Can leverage agents, tools, and storage backends as needed

## Monitoring and Observability

### Logging Architecture

```mermaid
flowchart LR
    APP["Application"] --> STRUCT_LOG["Structured Logging"]
    STRUCT_LOG --> FORMAT["JSON Formatting"]
    FORMAT --> OUTPUT["Log Output"]

    OUTPUT --> FILE["File Logs"]
    OUTPUT --> CONSOLE["Console Logs"]
    OUTPUT --> REMOTE["Remote Logging Service"]

    classDef app fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#4caf50,stroke-width:2px

    class APP app
    class STRUCT_LOG,FORMAT,OUTPUT process
    class FILE,CONSOLE,REMOTE output
```

### Metrics and Monitoring

- **Request Metrics**: Response times, error rates, throughput
- **Resource Metrics**: Memory usage, CPU utilization, database connections
- **Business Metrics**: Conversation counts, token usage, agent performance
- **Health Checks**: System health and dependency status

## Deployment Architecture

### Local Development Deployment

```mermaid
flowchart TB
    CLIENT["API Client"] --> APP_INSTANCE["Ingenious Instance<br/>:8000"]
    APP_INSTANCE --> LOCAL_DB["SQLite Database<br/>.tmp/chat_history.db"]
    APP_INSTANCE --> LOCAL_FILES["Local File Storage<br/>.tmp/"]
    APP_INSTANCE --> CHROMADB["ChromaDB<br/>Knowledge Base"]
    APP_INSTANCE --> AZURE_OPENAI["Azure OpenAI<br/>LLM Service"]

    classDef client fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef app fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef localServices fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef azureServices fill:#e1f5fe,stroke:#01579b,stroke-width:2px

    class CLIENT client
    class APP_INSTANCE app
    class LOCAL_DB,LOCAL_FILES,CHROMADB localServices
    class AZURE_OPENAI azureServices
```

### Production Azure Deployment

```mermaid
flowchart TB
    LOAD_BALANCER["Azure Load Balancer"] --> APP1["Ingenious Instance 1<br/>Container Apps"]
    LOAD_BALANCER --> APP2["Ingenious Instance 2<br/>Container Apps"]
    LOAD_BALANCER --> APPN["Ingenious Instance N<br/>Container Apps"]

    APP1 --> AZURE_SQL["Azure SQL Database<br/>Chat History"]
    APP2 --> AZURE_SQL
    APPN --> AZURE_SQL

    APP1 --> COSMOS_DB["Cosmos DB<br/>Document Storage"]
    APP2 --> COSMOS_DB
    APPN --> COSMOS_DB

    APP1 --> BLOB_STORAGE["Azure Blob Storage<br/>File Storage"]
    APP2 --> BLOB_STORAGE
    APPN --> BLOB_STORAGE

    APP1 --> AI_SEARCH["Azure AI Search<br/>Knowledge Base"]
    APP2 --> AI_SEARCH
    APPN --> AI_SEARCH

    APP1 --> AZURE_OPENAI["Azure OpenAI<br/>LLM Service"]
    APP2 --> AZURE_OPENAI
    APPN --> AZURE_OPENAI

    classDef loadBalancer fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef appInstance fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef database fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef storage fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef aiService fill:#fce4ec,stroke:#e91e63,stroke-width:2px

    class LOAD_BALANCER loadBalancer
    class APP1,APP2,APPN appInstance
    class AZURE_SQL,COSMOS_DB database
    class BLOB_STORAGE,AI_SEARCH storage
    class AZURE_OPENAI aiService
```

This flexible architecture enables developers to start with local development using minimal dependencies (SQLite, ChromaDB) and seamlessly scale to production Azure deployments (Azure SQL, Cosmos DB, Azure AI Search, Azure Blob, Container Apps) as needed. The dual-configuration approach provides a smooth development-to-production pathway while maintaining the same API interface and conversation flows.
