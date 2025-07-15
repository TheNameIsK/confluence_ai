# Confluence AI

Confluence AI is a lightweight Flask application that integrates with Atlassian Confluence. It indexes your Confluence pages and provides a simple chat interface for searching or managing content via the Confluence API.

## Installation

1. Create and activate a Python virtual environment (optional but recommended).
2. Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

1. Set up environment variables for your Confluence instance and OpenAI credentials. A `.env` file can be used to store these values.
2. Run the application:

```bash
python app.py
```

The web interface will be available at `http://localhost:5000` by default.
