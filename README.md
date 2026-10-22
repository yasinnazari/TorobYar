# first_agent

A simple AI agent that uses OpenRouter chat completions API with native tool calling.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenRouter API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

3. Run the agent:
   ```bash
   python -m first_agent.main
   ```

## Usage

The agent can handle queries that require the `calculate_sum` tool. The model will decide when to call the tool and will use the result to generate its final response.

## Architecture

- Uses OpenAI-compatible SDK with OpenRouter
- Native tool calling with `calculate_sum` function
- Maximum 5 model/tool interaction rounds
- API key from environment variable only
- Error handling and logging