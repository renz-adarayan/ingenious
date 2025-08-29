# Authentication

Ingenious supports both Basic Authentication and JWT (Bearer) token authentication for protecting API endpoints.

## Configuration

Enable authentication by setting these environment variables:

```bash
# Enable authentication
INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__ENABLE=true

# Set Basic Auth credentials
INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__USERNAME=<username>
INGENIOUS_WEB_CONFIGURATION__AUTHENTICATION__PASSWORD=<password>
```

## Basic Authentication

Use HTTP Basic Authentication with your configured username and password.

### Example curl commands:

```bash
# Health check (no auth required)
curl http://localhost:<port>/api/v1/health

# Chat endpoint with Basic Auth
curl -X POST http://localhost:<port>/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $(echo -n '<username>:<password>' | base64)" \
  -d '{
    "user_prompt": "Hello",
    "conversation_flow": "classification_agent",
    "thread_id": "test123"
  }'

# Test with wrong credentials (should return 401)
curl -X POST http://localhost:<port>/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $(echo -n 'wrong:wrong' | base64)" \
  -d '{
    "user_prompt": "Hello",
    "conversation_flow": "classification_agent",
    "thread_id": "test123"
  }'
```

## JWT Authentication

Use JWT tokens for stateless authentication. First obtain tokens via login, then use the access token.

### Example curl commands:

```bash
# 1. Login to get JWT tokens
curl -X POST http://localhost:<port>/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "<username>", "password": "<password>"}'

# Response:
# {
#   "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "token_type": "bearer"
# }

# 2. Use access token for API calls
TOKEN="<your-access-token>"

curl -X POST http://localhost:<port>/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "user_prompt": "Hello with JWT",
    "conversation_flow": "classification_agent",
    "thread_id": "test-jwt"
  }'

# 3. Verify token validity
curl -X GET http://localhost:<port>/api/v1/auth/verify \
  -H "Authorization: Bearer $TOKEN"

# 4. Refresh access token using refresh token
curl -X POST http://localhost:<port>/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<your-refresh-token>"}'
```

## Authentication Testing

### Test Basic Auth:
```bash
# Correct credentials - should return 200
curl -X POST http://localhost:<port>/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $(echo -n '<username>:<password>' | base64)" \
  -d '{"user_prompt": "Test", "conversation_flow": "classification_agent", "thread_id": "test"}'

# Wrong credentials - should return 401
curl -X POST http://localhost:<port>/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $(echo -n 'wrong:wrong' | base64)" \
  -d '{"user_prompt": "Test", "conversation_flow": "classification_agent", "thread_id": "test"}'
```

### Test JWT Auth:
```bash
# Get token
TOKEN=$(curl -s -X POST http://localhost:<port>/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "<username>", "password": "<password>"}' | \
  python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

# Use token - should return 200
curl -X POST http://localhost:<port>/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"user_prompt": "Test JWT", "conversation_flow": "classification_agent", "thread_id": "test-jwt"}'

# Verify token
curl -X GET http://localhost:<port>/api/v1/auth/verify \
  -H "Authorization: Bearer $TOKEN"
```

## Security Notes

- **Production**: Always use strong passwords and secure JWT secret keys
- **HTTPS**: Use HTTPS in production to protect credentials in transit
- **Token Expiry**: Access tokens expire in 24 hours by default, refresh tokens in 7 days
- **Environment Variables**: Store credentials in environment variables, not in code
