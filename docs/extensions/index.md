# Extensions and Customization

This section covers how to extend Insight Ingenious with custom components and configurations.

## Quick Start: Create Your First Custom Workflow

🚀 **New to custom workflows?** Start here:

**[Complete Custom Workflow Guide →](../guides/custom-workflows.md)**

This comprehensive guide includes:
- Step-by-step implementation with working examples
- Real-world task management system walkthrough
- Critical setup requirements (PYTHONPATH, server restart)
- Common issues and troubleshooting solutions
- Production deployment considerations

## Available Extension Guides

### Core Extensions

- **[Custom Workflow Development](../guides/custom-workflows.md)** - **START HERE** - Complete guide with working examples
- **[Custom Agents](custom-agents.md)** - Create specialized AI agents for specific tasks
- **[Conversation Patterns](conversation-patterns.md)** - Design custom multi-agent conversation flows
- **[Flow Implementation](flow-implementation.md)** - Implement custom workflow logic
- **[Custom Templates](custom-templates.md)** - Create and modify prompt templates

## Getting Started with Extensions

Before creating custom extensions, ensure you have:

1. **A working Insight Ingenious installation** - See [Getting Started](../getting-started.md)
2. **Understanding of core workflows** - See [Architecture](../architecture.md)
3. **Development environment setup** - See [Development Guide](../development.md)

## Extension Architecture

Ingenious uses a modular architecture that supports both external and internal extensions. The **recommended approach is external extensions** to avoid modifying library code.

**Critical Requirements for Custom Workflows:**
- Set `PYTHONPATH` environment variable to include your project directory
- Use exact class name `ConversationFlow` (not `MyFlow` or other names)
- Import from `ingenious.services.chat_services.multi_agent.service import IConversationFlow`
- Restart server after creating new workflows

For detailed implementation guidance, see the [Development Guide](../development.md#extending-ingenious).
