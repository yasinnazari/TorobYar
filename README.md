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
   or
   uvicorn app.main:app --reload
   ```


**1. Create datasets folder and copy torob-datasets file and paste in datasets folder**
