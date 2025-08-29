# Troubleshooting Guide

This guide covers common issues encountered when setting up and running Ingenious.

## Installation Issues

### Python Version Errors

**Issue**: `Python 3.13+ is required`

**Solution**:
```bash
# Check current Python version
python --version

# Install Python 3.13 via pyenv (recommended)
pyenv install 3.13.0
pyenv local 3.13.0

# Or update system Python version
```

### UV Package Manager Issues

**Issue**: `uv: command not found`

**Solution**:
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart terminal or source shell config
source ~/.bashrc  # or ~/.zshrc
```

## Configuration Issues

### Azure OpenAI Connection Errors

**Issue**: `Authentication failed` or `API key invalid`

**Solutions**:
1. **Verify API key format**: Ensure no extra spaces or quotes
2. **Check endpoint URL**: Use Cognitive Services format:
   - ✅ Correct: `https://eastus.api.cognitive.microsoft.com/`
   - ❌ Incorrect: `https://your-resource.openai.azure.com/`
3. **Verify deployment name**: Must match your Azure deployment exactly
4. **Check API version**: Use `2024-12-01-preview`

### Port Conflicts

**Issue**: `Port 8000 already in use`

**Solutions**:
```bash
# Find and kill process using port
lsof -ti:8000 | xargs kill -9

# Or use different port
INGENIOUS_WEB_CONFIGURATION__PORT=8001 uv run ingen serve

# Update .env file permanently
echo "INGENIOUS_WEB_CONFIGURATION__PORT=8001" >> .env
```

### Database Connection Issues

**Issue**: SQLite database errors or permission denied

**Solutions**:
```bash
# Ensure .tmp directory exists and is writable
mkdir -p ./.tmp
chmod 755 ./.tmp

# Check database path in .env
INGENIOUS_CHAT_HISTORY__DATABASE_PATH=./.tmp/chat_history.db
```

## Workflow Issues

### Workflow Not Found

**Issue**: `Workflow 'custom_workflow' not found`

**Solutions**:
1. **Check PYTHONPATH**: `export PYTHONPATH=$(pwd):$PYTHONPATH`
2. **Restart server**: New workflows require server restart
3. **Verify directory structure**: Must match exactly
4. **Test import**: `python -c "import ingenious_extensions.services.chat_services.multi_agent.conversation_flows.workflow_name.workflow_name"`

### Custom Workflow Import Errors

**Issue**: `Failed to import module` for custom workflows

**Solutions**:
1. **Create all __init__.py files**: Required for Python module discovery
2. **Check class name**: Must be exactly `ConversationFlow`
3. **Verify imports**: Use correct import paths for your Ingenious version
4. **Restart server**: Always restart after creating new workflows

## Authentication Issues

### Basic Authentication Fails

**Issue**: `401 Unauthorized` with correct credentials

**Solutions**:
1. **Check environment variables**:
   ```bash
   INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__ENABLE=true
   INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__USERNAME=admin
   INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__PASSWORD=secure_password
   ```
2. **Restart server**: Authentication changes require restart
3. **Test encoding**: `echo -n 'admin:password' | base64`

### JWT Token Issues

**Issue**: JWT tokens not working or expiring

**Solutions**:
1. **Login first**: Get fresh token via `/api/v1/auth/login`
2. **Check token format**: Use `Bearer token_here` format
3. **Verify token**: Use `/api/v1/auth/verify` endpoint

## Performance Issues

### Slow Response Times

**Issue**: API responses taking too long

**Solutions**:
1. **Check model selection**: Use faster models like `gpt-4o-mini`
2. **Monitor token usage**: Large prompts slow down responses
3. **Enable caching**: Consider prompt caching for repeated requests
4. **Check Azure region**: Use closest Azure region

### Memory Issues

**Issue**: High memory usage or out-of-memory errors

**Solutions**:
1. **Limit conversation history**: Clear chat history periodically
2. **Monitor token tracking**: Large conversations consume memory
3. **Restart server**: Clear accumulated memory usage

## Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Enable debug logging
export INGENIOUS_LOGGING__LOG_LEVEL=debug
export INGENIOUS_LOGGING__ROOT_LOG_LEVEL=debug

# Start server with debug mode
uv run ingen serve --port 8000
```

## Getting Help

If issues persist:

1. **Check server logs**: Look for error messages in terminal output
2. **Verify configuration**: Run `uv run ingen validate`
3. **Test core workflows**: Ensure basic functionality works
4. **Review documentation**: Check this guide and main documentation
5. **Create minimal test**: Isolate the issue with simple test cases

## Common Error Messages

### Configuration Errors
- `Configuration validation failed`: Check .env file format and required fields
- `Model configuration missing`: Verify Azure OpenAI settings

### Runtime Errors
- `Connection timeout`: Check network connectivity and Azure endpoint
- `Token limit exceeded`: Reduce conversation length or prompt size
- `Workflow failed`: Check custom workflow implementation

### Import Errors
- `Module not found`: Verify PYTHONPATH and directory structure
- `Class not found`: Ensure ConversationFlow class exists and is named correctly
