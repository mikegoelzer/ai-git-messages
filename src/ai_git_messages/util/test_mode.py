import argparse
from rich import print as rprint
import logging
from ai_git_messages.types import TestMode
from ai_git_messages.ai.run_model import run_model
from ai_git_messages.ai.prompt import get_prompt
import tiktoken

log = logging.getLogger(__name__)

prompt_prefix      = "prompt>   "; prompt_prefix = f"[green4]{prompt_prefix}[/]"
response_prefix    = "response> "; response_prefix = f"[sky_blue3]{response_prefix}[/]"

def _indent_str(s: str, indent: str = "    ") -> str:
    return "\n".join([f"{indent}{line}" for line in s.split("\n")])

def _print_prompt_and_response(prompt: str, model_response: str, verbosity: int = 0) -> None:
    if verbosity >= 1:
        rprint(f"{_indent_str(prompt, prompt_prefix)}")
        rprint(f"--------------------------------")
        rprint(f"{_indent_str(model_response, response_prefix)}")
        rprint(f"--------------------------------")

def test_mode(args: argparse.Namespace) -> int:
    prompt = get_prompt(args.output_type, verbosity=0)
    # prompt_len = len(prompt.encode('utf-8'))
    enc = tiktoken.get_encoding("cl100k_base")
    prompt_token_count = len(enc.encode(prompt))

    model_response = run_model(args.ai_source, args.output_type, verbosity=0)
    # model_response_len = len(s.encode('utf-8'))
    enc = tiktoken.get_encoding("cl100k_base")
    model_response_token_count = len(enc.encode(model_response))
    # rprint(f"Model response length: [bold]{model_response_len} bytes[/]")
    _print_prompt_and_response(prompt, model_response, args.verbosity)
    rprint(f"[chartreuse2]Token utilization:[/]")
    rprint(f"  input:  [bold]{prompt_token_count}[/] tokens")
    rprint(f"  output: [bold]{model_response_token_count}[/] tokens")
    return 0
