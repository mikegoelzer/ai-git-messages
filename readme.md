# ai-git-messages

CLI tool for generating git messages using AI.

## Installation

```bash
# Option 1: using `pipx`
pipx install ai-git-messages

# Option 2: using plain `pip`
python -m pip install --upgrade pip
python -m pip install ai-git-messages

# Option 3: using `uv`
uv pip install ai-git-messages
```

## Usage

```bash
ai-git-messages --help
```

## Configuration

Ollama modes talk to the server named by `OLLAMA_HOST` (default `127.0.0.1:11434`, same as Ollama itself). To set it for this program only, copy `.env.sample` to `.env` in the repo root and edit it; `.env` overrides the environment.


## Contributing

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for development setup and contribution guidelines.
