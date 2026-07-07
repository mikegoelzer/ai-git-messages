from ..ai.prompt import get_prompt
from ..types import (
    OutputType,
    AiSource,
    PRFromBranchDescription,
    ChangesOnMainDescription,
)
from ollama import chat
from anthropic import Anthropic
import subprocess
import os
import sys
import json
from pydantic import ValidationError
from rich.console import Console
from ..types import AiSource

console = Console()

def cursor_generate(output_type: OutputType, verbose: bool = False) -> str:
    prompt = get_prompt(output_type, verbose)
    if verbose:
        console.log("Prompt:", style="bold")
        console.log(prompt, highlight=True, end="\n\n")
        # time.sleep(1) # this is for the logger to print a new time stamp
        console.log(f"Using cursor-agent to generate {output_type.desc}...", end="\\n\\n")

    p = subprocess.run(
        ["cursor-agent", "-p", "--output-format", "json", "--approve-mcps", "--trust"],
        input=prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd(),
    )
    if p.returncode not in (0, None):
        console.log(f"Error: {p.stderr}", style="red bold", end="\n\n")
        sys.exit(p.returncode)
        #raise subprocess.CalledProcessError(p.returncode, p.args, p.stderr)
    response_json = json.loads(p.stdout.strip())
    s = response_json["result"]
    if "```json" in s:
        # slice anything preceding the first "```json"
        s = s.split("```json")[1]
        # slice anything following the last "```"
        s = s.split("```")[0]
    return s

def ollama_generate(output_type: OutputType, verbose: bool = False) -> str:
    prompt = get_prompt(output_type)
    if verbose:
        console.log("Prompt:", style="bold")
        console.log(prompt, highlight=True, end="\n\n")
        # time.sleep(1) # this is for the logger to print a new time stamp
        console.log(f"Using ollama (kimi-k2.6:cloud) to generate {output_type.desc}...", end="\\n\\n")

    response = chat(
        messages=[
        {
            'role': 'user',
            'content': prompt,
        }
        ],
        model='kimi-k2.6:cloud',
        format=PRFromBranchDescription.model_json_schema() if output_type == OutputType.PR_DESCRIPTION else ChangesOnMainDescription.model_json_schema(),
    )
    if verbose:
        console.log("Response:", style="bold")
        console.log(response.message.content, highlight=True, end="\\n\\n")
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

def claude_generate(output_type: OutputType, verbose: bool = False) -> str:
    prompt = get_prompt(output_type)
    if verbose:
        console.log("Prompt:", style="bold")
        console.log(prompt, highlight=True, end="\n\n")
        # time.sleep(1) # this is for the logger to print a new time stamp
        console.log(f"Using Claude to generate {output_type.desc}...", end="\\n\\n")

    client = Anthropic()

    # Determine which schema to use
    schema = PRFromBranchDescription.model_json_schema() if output_type == OutputType.PR_DESCRIPTION else ChangesOnMainDescription.model_json_schema()

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=8192,
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        temperature=0.0,
    )

    if verbose:
        console.log("Response:", style="bold")
        console.log(response.content[0].text, highlight=True, end="\\n\\n")

    resp = response.content[0].text

    # Handle markdown code blocks if present
    if "```json" in resp:
        # slice anything preceding the first "```json"
        s = resp.split("```json")[1]
        # slice anything following the last "```"
        s = s.split("```")[0]
    else:
        s = resp

    return s

def validate_resp_str_and_return_json_str(resp_str: str, output_type: OutputType, verbose: bool = False) -> str:
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
            if verbose:
                console.log("Pull Request Description:", style="bold")
                console.log(pr_desc, highlight=True, end="\\n\\n")
            s = pr_desc.to_json()
        elif output_type == OutputType.BRANCH_OFF_FROM_MAIN_ARGUMENTS:
            changes_on_main = ChangesOnMainDescription.model_validate_json(resp_str)
            if verbose:
                console.log("Branch off from main arguments:", style="bold")
                console.log(changes_on_main, highlight=True, end="\\n\\n")
            s = changes_on_main.to_json()
        else:
            raise ValueError(f"Invalid output type: {output_type}")
            s = None
    except ValidationError as e:
        console.log(f"Validation error: {e}", style="red bold", end="")
        s = None
    return s


def run_model(ai_source: AiSource, output_type: OutputType, verbose: bool = False) -> str:
    if verbose:
        console.log(f"run_model:\n  ai_source='{ai_source}'\n  output type='{output_type}'\n  verbose='{verbose}'", end="\\n\\n")

    if ai_source == AiSource.OLLAMA:
        console.log(f"Using ollama (kimi-k2.6:cloud) to generate {output_type.desc}...", end="\\n\\n")
        resp_str = ollama_generate(output_type, verbose)
    # elif ai_source == AiSource.CURSOR:
    #     console.log(f"Using cursor-agent to generate {output_type.desc}...", end="\\n\\n")
    #     resp_str = cursor_generate(output_type, verbose)
    elif ai_source == AiSource.CLAUDE:
        console.log(f"Using Claude to generate {output_type.desc}...", end="\\n\\n")
        resp_str = claude_generate(output_type, verbose)
    elif ai_source == AiSource.DEBUG:
        if output_type == OutputType.PR_DESCRIPTION:
            console.log(f"Using hardcoded {output_type.desc}...", end="\\n\\n")
            resp_obj = {
                "title":"Add git-branch script and clean Makefile output",
                "body":"- Replaced emoticons in Makefile fetch-latest-tags and publish status messages with plain text symbols.\n- Extracted the git-branch, commit, and push workflow into a new script `scripts/git-branch-add-commit-push.sh`.\n- Removed the temporary `push` rule from the Makefile.\n- Updated the script to validate arguments, ensure `git-extras` is installed, and provide a help message.\n- Added prompts for commit message and optional push confirmation.",
            }
        elif output_type == OutputType.BRANCH_OFF_FROM_MAIN_ARGUMENTS:
            console.log(f"Using hardcoded {output_type.desc}...", end="\\n\\n")
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
    s = validate_resp_str_and_return_json_str(resp_str, output_type, verbose)
    if s is None:
        console.log("Validation failed", style="red bold", end="\n\n")
        return None
    return s
