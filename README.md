# N8N Pipe for Open-WebUI

This module implements a robust, asynchronous pipe for Open-WebUI that serves as an interface between the application and n8n workflows. It allows redirecting chat messages to n8n workflows and incorporating responses back into the conversation.

Based on/forked from https://openwebui.com/f/coleam/n8n_pipe. Many thanks to the original author.

## Features

- ✅ Fully asynchronous interface compatible with Open-WebUI
- ✅ Enhanced HTTP and general error handling with retry mechanism
- ✅ Verification of expected response fields
- ✅ Uses `httpx` for asynchronous HTTP requests
- ✅ Configurable timeout to avoid request blocking
- ✅ Message history management
- ✅ Comprehensive input validation
- ✅ Detailed status updates during processing
- ✅ Robust error handling for edge cases

## Installation

1. Ensure you have the required dependencies installed:
   ```bash
   pip install httpx pydantic
   ```

2. Add the pipe file to your Open-WebUI installation.

## Configuration

The `Pipe` class uses a "valves" system to configure its behavior:

| Parameter | Description | Default Value |
|-----------|-------------|------------------|
| `n8n_host` | Base URL for n8n server (without trailing slash) | http://localhost:5678 |
| `n8n_webhook_id` | Webhook ID from n8n | your-webhook-id-here |
| `n8n_test_mode` | Whether to use test mode URLs for n8n webhooks | False |
| `n8n_bearer_token` | Bearer authentication token | your-token-here |
| `input_field` | Input field name in the request | chatInput |
| `response_field` | Response field name in the JSON response | output |
| `emit_interval` | Interval between status emissions (seconds) | 2.0 |
| `enable_status_indicator` | Enable or disable status indicators | True |
| `timeout` | HTTP request timeout (seconds) | 30.0 |
| `max_retries` | Maximum number of retries for failed requests | 2 |
| `history_limit` | Maximum number of messages to keep in history | 10 |

### Handling Test Mode vs Production

The pipe automatically handles the difference between n8n test mode and production:

1. When `n8n_test_mode` is set to `False` (default), it uses the URL format:
   ```
   {n8n_host}/webhook/{n8n_webhook_id}
   ```

2. When `n8n_test_mode` is set to `True`, it uses the URL format:
   ```
   {n8n_host}/webhook-test/{n8n_webhook_id}
   ```

This allows you to easily switch between testing and production environments without changing the webhook URL manually.

## Data Format

### Data Sent to n8n

The data format sent to the n8n webhook is as follows:

```json
{
  "sessionId": "user_id - content_of_first_message",
  "chatInput": "question_extracted_from_last_message"
}
```

Where:
- `sessionId` is a string combining the user ID and the content of the first message (limited to 100 characters)
- The input field (default `chatInput`) is configured via `valves.input_field`

### Data Received from n8n

The expected response from n8n must be in JSON format and contain a field specified by `valves.response_field` (default "output"):

```json
{
  "output": "Text response from the n8n workflow"
}
```

### Status Emission Format

When an event emitter is provided, the module emits events in the following format:

```json
{
  "type": "status",
  "data": {
    "status": "in_progress|complete",
    "level": "info|warning|error",
    "description": "Message describing the current status",
    "done": false|true
  }
}
```

## Usage Flow

1. The user sends a message in the Open-WebUI interface
2. The pipe intercepts this message and sends it to the configured n8n webhook
3. n8n processes the message according to its configured workflow
4. The pipe receives the response from n8n and adds it to the conversation
5. The interface displays the response as if it came directly from the language model

## Error Handling

The pipe includes comprehensive error handling:

1. **HTTP Errors**: All HTTP errors are caught, logged, and reported back with appropriate status codes.
2. **Retry Mechanism**: Failed requests can be automatically retried based on the `max_retries` setting.
3. **Validation**: Input validation ensures proper URLs and field structures.
4. **Message Validation**: Checks for empty messages, non-user messages, etc.
5. **Response Field Validation**: Ensures the expected response field exists in the n8n response.

## Configuring n8n

To configure n8n to work with this pipe:

1. Create a workflow in n8n
2. Add a Webhook node as an entry point (trigger)
3. Configure this webhook to accept POST requests with Bearer authentication
4. Process the input data from the field configured by `input_field` (default: "chatInput")
5. Return the response in a JSON object containing the field configured by `response_field` (default: "output")

## Troubleshooting

Common issues and solutions:

1. **Error "No messages found in the request body"**
   - Check that the request body contains a non-empty `messages` array

2. **Error "Last message is not from user"**
   - Ensure the last message in the array has `role: "user"`

3. **Error "Empty question received"**
   - The last user message has empty content

4. **HTTP error when calling n8n**
   - Verify that the webhook URL is correct
   - Check that the Bearer token is valid
   - Ensure that the n8n webhook is accessible from Open-WebUI

5. **Missing response field**
   - Verify that your n8n workflow returns a JSON with the field configured in `response_field`

6. **Retry failed**
   - If all retries fail, check your n8n server availability
   - Consider increasing the `max_retries` or `timeout` values

## Testing

Comprehensive tests are provided to ensure code reliability. To run them:

```bash
python -m pytest -xvs
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
