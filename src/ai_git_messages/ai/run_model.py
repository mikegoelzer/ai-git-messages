from ..ai.prompt import get_prompt
from ..types import (
    OutputType,
    AiSource,
    PRFromBranchDescription,
    ChangesOnMainDescription,
)
from ollama import Client
from dotenv import load_dotenv
from pathlib import Path
import subprocess
import os
import sys
import json
from pydantic import ValidationError
from ..types import AiSource, OllamaModel
from ..util.log_console import log_console

# Per-invocation only; does not change the user's Claude Code default model.
CLAUDE_CODE_MODEL = "claude-sonnet-5"

def cursor_generate(output_type: OutputType, verbosity: int = 0) -> str:
    prompt = get_prompt(output_type, verbosity)
    if verbosity >= 2:
        log_console.log("Prompt:", style="bold")
        log_console.log(prompt, highlight=True, markup=False, end="\n\n")
        # time.sleep(1) # this is for the logger to print a new time stamp
        log_console.log(f"Using cursor-agent to generate {output_type.desc}...", end="\\n\\n")

    p = subprocess.run(
        ["cursor-agent", "-p", "--output-format", "json", "--approve-mcps", "--trust"],
        input=prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd(),
    )
    if p.returncode not in (0, None):
        log_console.log(f"Error: {p.stderr}", style="red bold", end="\n\n")
        sys.exit(p.returncode)
        #raise subprocess.CalledProcessError(p.returncode, p.args, p.stderr)
    response_json = json.loads(p.stdout.strip())
    s = response_json["result"]
    if verbosity >= 2:
        log_console.log("Response:", style="bold")
        log_console.log(s, highlight=True, markup=False, end="\n\n")
    if "```json" in s:
        # slice anything preceding the first "```json"
        s = s.split("```json")[1]
        # slice anything following the last "```"
        s = s.split("```")[0]
    return s

USER_ENV_FILE = Path.home() / ".config" / "ai-git-messages" / ".env"

def ollama_host() -> str:
    """Resolve the Ollama server: USER_ENV_FILE overrides the environment, which overrides ollama's own default."""
    load_dotenv(USER_ENV_FILE, override=True)
    return os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")

def ollama_generate(output_type: OutputType, *, ollama_model: OllamaModel = OllamaModel.QWEN2_5_CODER_7B_LOCAL, verbosity: int = 0) -> str:
    prompt = get_prompt(output_type)
    if verbosity >= 2:
        log_console.log("Prompt:", style="bold")
        log_console.log(prompt, highlight=True, markup=False, end="\n\n")
        # time.sleep(1) # this is for the logger to print a new time stamp
        log_console.log(f"Using ollama ({ollama_model.value}) to generate {output_type.desc}...", end="\\n\\n")

    chat_args = dict(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model=ollama_model.value,
        format=PRFromBranchDescription.model_json_schema() if output_type == OutputType.PR_DESCRIPTION else ChangesOnMainDescription.model_json_schema(),
    )
    response = Client(host=ollama_host()).chat(**chat_args)
    if verbosity >= 2:
        log_console.log("Response:", style="bold")
        log_console.log(response.message.content, highlight=True, markup=False, end="\\n\\n")
    resp = response.message.content

    # Handle markdown code blocks if present
    if "```json" in resp:
        # slice anything preceding the first "```json"
        s = resp.split("```json")[1]
        # slice anything following the last "```"
        s = s.split("```")[0]
    else:
        s = resp

    return s

def claude_generate(output_type: OutputType, verbosity: int = 0) -> str:
    prompt = get_prompt(output_type)
    if verbosity >= 2:
        log_console.log("Prompt:", style="bold")
        log_console.log(prompt, highlight=True, markup=False, end="\n\n")
        # time.sleep(1) # this is for the logger to print a new time stamp
        log_console.log(f"Using Claude Code ({CLAUDE_CODE_MODEL}) to generate {output_type.desc}...", end="\\n\\n")

    schema = (PRFromBranchDescription if output_type == OutputType.PR_DESCRIPTION else ChangesOnMainDescription).model_json_schema()
    p = subprocess.run(
        ["claude", "-p", "--output-format", "json", "--model", CLAUDE_CODE_MODEL, "--json-schema", json.dumps(schema)],
        input=prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd(),
    )
    if p.returncode != 0:
        log_console.log(f"Error: {p.stderr}", style="red bold", end="\n\n")
        sys.exit(p.returncode)
    result = json.loads(p.stdout)
    if result.get("is_error") or "structured_output" not in result:
        log_console.log(f"Error: claude returned no structured output: {result.get('result')}", style="red bold", end="\n\n")
        sys.exit(1)
    s = json.dumps(result["structured_output"])

    if verbosity >= 2:
        log_console.log("Response:", style="bold")
        log_console.log(s, highlight=True, markup=False, end="\\n\\n")

    return s

def validate_resp_str_and_return_json_str(resp_str: str, output_type: OutputType, verbosity: int = 0) -> str:
    """
    Converts the response from the model into a JSON string.

    Args:
        resp_str: the response from the model as a string
        output_type: the type of output to convert the response to a JSON string for

    Returns:
        the JSON string
    """
    s: str | None = None
    try:
        if output_type == OutputType.PR_DESCRIPTION:
            pr_desc = PRFromBranchDescription.model_validate_json(resp_str)
            if verbosity >= 2:
                log_console.log("Pull Request Description:", style="bold")
                log_console.log(pr_desc, highlight=True, markup=False, end="\\n\\n")
            s = pr_desc.to_json()
        elif output_type == OutputType.BRANCH_OFF_FROM_MAIN_ARGUMENTS:
            changes_on_main = ChangesOnMainDescription.model_validate_json(resp_str)
            if verbosity >= 2:
                log_console.log("Branch off from main arguments:", style="bold")
                log_console.log(changes_on_main, highlight=True, markup=False, end="\\n\\n")
            s = changes_on_main.to_json()
        else:
            raise ValueError(f"Invalid output type: {output_type}")
            s = None
    except ValidationError as e:
        log_console.log(f"Validation error: {e}", style="red bold", end="")
        s = None
    return s


def run_model(ai_source: AiSource, output_type: OutputType, verbosity: int = 0) -> str:
    if verbosity >= 2:
        log_console.log(f"run_model:\n  ai_source='{ai_source}'\n  output type='{output_type}'\n  verbosity='{verbosity}'", end="\\n\\n")

    if ai_source == AiSource.OLLAMA_LOCAL:
        ollama_model = OllamaModel.QWEN2_5_CODER_7B_LOCAL
        if verbosity >= 2:
            log_console.log(f"Using ollama ({ollama_model.value}) to generate {output_type.desc}...", end="\\n\\n")
        resp_str = ollama_generate(output_type, ollama_model=ollama_model, verbosity=verbosity)
    elif ai_source == AiSource.OLLAMA_CLOUD:
        ollama_model = OllamaModel.KIMI_K2_6_CLOUD
        if verbosity >= 2:
            log_console.log(f"Using ollama ({ollama_model.value}) to generate {output_type.desc}...", end="\\n\\n")
        resp_str = ollama_generate(output_type, ollama_model=ollama_model, verbosity=verbosity)
    # elif ai_source == AiSource.CURSOR:
    #     console.log(f"Using cursor-agent to generate {output_type.desc}...", end="\\n\\n")
    #     resp_str = cursor_generate(output_type, verbosity)
    elif ai_source == AiSource.CLAUDE:
        if verbosity >= 2:
            log_console.log(f"Using Claude Code ({CLAUDE_CODE_MODEL}) to generate {output_type.desc}...", end="\\n\\n")
        resp_str = claude_generate(output_type, verbosity)
    elif ai_source == AiSource.DEBUG:
        if output_type == OutputType.PR_DESCRIPTION:
            if verbosity >= 2:
                log_console.log(f"Using hardcoded {output_type.desc}...", end="\\n\\n")
            resp_obj = {
                "title":"Add git-branch script and clean Makefile output",
                "body":"- Replaced emoticons in Makefile fetch-latest-tags and publish status messages with plain text symbols.\n- Extracted the git-branch, commit, and push workflow into a new script `scripts/git-branch-add-commit-push.sh`.\n- Removed the temporary `push` rule from the Makefile.\n- Updated the script to validate arguments, ensure `git-extras` is installed, and provide a help message.\n- Added prompts for commit message and optional push confirmation.",
            }
        elif output_type == OutputType.BRANCH_OFF_FROM_MAIN_ARGUMENTS:
            if verbosity >= 2:
                log_console.log(f"Using hardcoded {output_type.desc}...", end="\\n\\n")
            resp_obj = {
                "feat_or_fix":"feat",
                "branch_name":"add-auth-tokens",
                "commit_message":"Add auth tokens to the Makefile",
            }
        else:
            raise ValueError(f"Invalid output type: {output_type}")
        resp_str = json.dumps(resp_obj)
    else:
        raise ValueError(f"Invalid AI source: {ai_source}")

    # validate the response
    s = validate_resp_str_and_return_json_str(resp_str, output_type, verbosity)
    if s is None:
        log_console.log("Validation failed", style="red bold", end="\n\n")
        return None
    return s
