# Custom Workflow Development Guide

This guide walks you through creating custom conversation flows in Ingenious, based on real-world implementation experience.

## Prerequisites

Before creating custom workflows, ensure you have:

1. **Working Ingenious installation** - Complete the [Quick Start](../getting-started.md) setup
2. **Tested core workflows** - Verify `classification-agent`, `knowledge-base-agent`, and `sql-manipulation-agent` work
3. **Development environment** - Python 3.13+, uv package manager

## Step 1: Create the Directory Structure

Custom workflows follow a specific directory structure:

```bash
# Navigate to your project directory
cd /path/to/your/project

# Create the directory structure
mkdir -p ingenious_extensions/services/chat_services/multi_agent/conversation_flows/your_workflow_name

# Create __init__.py files for Python module discovery
touch ingenious_extensions/__init__.py
touch ingenious_extensions/services/__init__.py
touch ingenious_extensions/services/chat_services/__init__.py
touch ingenious_extensions/services/chat_services/multi_agent/__init__.py
touch ingenious_extensions/services/chat_services/multi_agent/conversation_flows/__init__.py
touch ingenious_extensions/services/chat_services/multi_agent/conversation_flows/your_workflow_name/__init__.py
```

## Step 2: Implement the Conversation Flow

Create your workflow file with the correct structure:

```python
# ingenious_extensions/services/chat_services/multi_agent/conversation_flows/task_manager/task_manager.py

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import EVENT_LOGGER_NAME, CancellationToken

import ingenious.config.config as config
from ingenious.client.azure import AzureClientFactory
from ingenious.models.agent import LLMUsageTracker
from ingenious.models.chat import ChatRequest, ChatResponse
from ingenious.services.chat_services.multi_agent.service import IConversationFlow


class ConversationFlow(IConversationFlow):
    """Custom task manager conversation flow."""

    async def get_conversation_response(self, chat_request: ChatRequest) -> ChatResponse:
        """Process task management requests."""

        # Get configuration and model setup
        _config = config.get_config()
        model_config = _config.models[0]

        # Initialize LLM usage tracking
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        logger.setLevel(logging.INFO)

        revision_id = str(uuid.uuid4())
        identifier = str(uuid.uuid4())

        llm_logger = LLMUsageTracker(
            agents=["task_manager_agent"],
            config=_config,
            chat_history_repository=self._chat_service.chat_history_repository,
            revision_id=revision_id,
            identifier=identifier,
            event_type="task_management",
        )

        logger.handlers = [llm_logger]

        # Create the Azure OpenAI client
        model_client = AzureClientFactory.create_openai_chat_completion_client(model_config)

        # Create system prompt
        system_prompt = f\"\"\"You are a helpful task management assistant.

        You can help users:
        1. Add new tasks
        2. List existing tasks
        3. Complete tasks
        4. Delete tasks

        User request: {chat_request.user_prompt}
        \"\"\"

        # Create the assistant agent
        task_agent = AssistantAgent(
            name="task_manager_agent",
            system_message=system_prompt,
            model_client=model_client,
        )

        # Create cancellation token
        cancellation_token = CancellationToken()

        try:
            # Process the user request
            response = await task_agent.on_messages(
                messages=[TextMessage(content=chat_request.user_prompt, source="user")],
                cancellation_token=cancellation_token,
            )

            result = response.chat_message.content

            # Add any custom business logic here
            if "add task" in chat_request.user_prompt.lower():
                result += "\\n\\n✅ Task functionality would be implemented here!"

        except Exception as e:
            result = f"I'm here to help you manage tasks! Error: {str(e)}"

        finally:
            # Close the model client connection
            await model_client.close()

        # Return ChatResponse object
        return ChatResponse(
            thread_id=chat_request.thread_id,
            message_id=identifier,
            agent_response=result,
            token_count=llm_logger.prompt_tokens if hasattr(llm_logger, 'prompt_tokens') else 0,
            max_token_count=0,
            memory_summary=f"Task management interaction: {chat_request.user_prompt[:50]}..."
        )
```

## Step 3: Critical Setup Requirements

### Set PYTHONPATH Environment Variable

**This step is critical** - without it, the server cannot discover your custom workflow:

```bash
# Set PYTHONPATH to include your project directory
export PYTHONPATH=/path/to/your/project:$PYTHONPATH

# For persistent setup, add to your shell configuration:
echo "export PYTHONPATH=$(pwd):$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc

# Or add to your .env file for the project:
echo "PYTHONPATH=$(pwd)" >> .env
```

### Restart the Server

The server must be restarted to discover new workflows:

```bash
# Stop existing server (Ctrl+C or kill process)
# Then restart with PYTHONPATH:
export PYTHONPATH=/path/to/your/project:$PYTHONPATH
KB_POLICY=local_only uv run ingen serve --port 8000
```

## Step 4: Test Your Custom Workflow

Create a test file to avoid JSON escaping issues:

```bash
# Create test file
cat > test_task_manager.json << EOF
{
    "user_prompt": "Add task: Review documentation",
    "conversation_flow": "task_manager"
}
EOF

# Test your custom workflow
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d @test_task_manager.json
```

**Expected Response:**
```json
{
  "thread_id": "uuid-here",
  "message_id": "uuid-here",
  "agent_response": "I can help you add a task to review documentation...\n\nTask functionality would be implemented here!",
  "token_count": 0,
  "memory_summary": "Task management interaction: Add task: Review documentation..."
}
```

## Step 5: Testing with Authentication

Custom workflows work seamlessly with both Basic Authentication and JWT authentication. Here's how to test your workflow with authentication enabled.

### Enable Authentication

First, update your `.env` file:

```bash
# Enable authentication
INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__ENABLE=true
INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__USERNAME=admin
INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__PASSWORD=secure_password
```

Restart the server for authentication to take effect:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
uv run ingen serve --port 8000
```

### Basic Authentication Testing

```bash
# Test with correct credentials (should succeed)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $(echo -n 'admin:secure_password' | base64)" \
  -d @test_task_manager.json

# Test with wrong credentials (should return 401)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $(echo -n 'wrong:wrong' | base64)" \
  -d @test_task_manager.json
```

**Expected Results:**
- Correct credentials: Full workflow response
- Wrong credentials: `{"detail":"Incorrect username or password"}`

### JWT Authentication Testing

```bash
# 1. Login to get JWT tokens
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "secure_password"}'

# Extract access token from response (manually or with jq)
TOKEN="your-access-token-here"

# 2. Test custom workflow with JWT token
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d @test_task_manager.json

# 3. Verify token (optional)
curl -X GET http://localhost:8000/api/v1/auth/verify \
  -H "Authorization: Bearer $TOKEN"

# 4. Test with invalid token (should return 401)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer invalid-token" \
  -d @test_task_manager.json
```

**Expected Results:**
- Valid JWT token: Full workflow response
- Invalid JWT token: `{"detail":"Authentication required"}`
- Token verification: `{"username":"admin","valid":true}`

For more authentication details, see the [Authentication Guide](../auth.md).

## Common Issues and Troubleshooting

### 1. Import Error: "Failed to import module"

**Error**: `Failed to import module 'services.chat_services.multi_agent.conversation_flows.your_workflow.your_workflow'`

**Solutions**:
- ✅ Check import path: `from ingenious.services.chat_services.multi_agent.service import IConversationFlow`
- ✅ Verify class name is exactly `ConversationFlow`
- ✅ Ensure PYTHONPATH includes your project directory
- ✅ Verify all `__init__.py` files exist in the directory tree

### 2. Module Not Discovered

**Error**: Workflow not found when testing

**Solutions**:
- ✅ Restart the server after creating new workflows
- ✅ Check directory naming matches: `conversation_flow` parameter = directory name
- ✅ Verify PYTHONPATH is set correctly: `echo $PYTHONPATH`
- ✅ Test Python import directly: `python -c "import ingenious_extensions.services.chat_services.multi_agent.conversation_flows.your_workflow.your_workflow"`

### 3. Server Import Issues

**Error**: Direct import failed with various module errors

**Solutions**:
- ✅ Ensure you're using the correct imports for your Ingenious version
- ✅ Check that all required dependencies are available
- ✅ Verify the directory structure matches exactly: `ingenious_extensions/services/chat_services/multi_agent/conversation_flows/workflow_name/workflow_name.py`

## Advanced Patterns

### Multi-Agent Workflows

For complex workflows with multiple agents, see the `bike-insights` template created by `ingen init` as a reference implementation.

### State Management

For workflows requiring persistent state (like the task manager example), you can:
- Use in-memory storage for development
- Integrate with databases for production
- Implement custom storage layers

### Error Handling

Always implement proper error handling:
```python
try:
    # Your workflow logic
    response = await agent.process(request)
    result = response.content
except Exception as e:
    result = f"Workflow error: {str(e)}"
    # Log error for debugging
    self._logger.error(f"Workflow failed: {e}")
```

## Production Considerations

1. **Performance**: Cache model clients and reuse connections
2. **Security**: Validate all user inputs and sanitize responses
3. **Monitoring**: Implement proper logging and metrics
4. **Testing**: Create comprehensive test suites for your workflows
5. **Documentation**: Document your custom workflows for team members

## Next Steps

- Explore the `bike-insights` workflow template for advanced multi-agent patterns
- Review the [Architecture Guide](../architecture.md) for deeper system understanding
- Check the main documentation for integration patterns
- See the [Complete Azure Deployment](complete-azure-deployment.md) guide for production setup
