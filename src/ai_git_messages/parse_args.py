import argparse
from curvpyutils.cli_util import VerbosityActionGroupFactory
from .types import AiSource, OutputType, TestMode
from importlib.metadata import version, PackageNotFoundError

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a pull request description or `git branch-off` arguments based on analysis of the current branches changes.")

    # Add version argument
    try:
        pkg_version = version("ai-git-messages")
    except PackageNotFoundError:
        pkg_version = "unknown (not installed)"

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {pkg_version}"
    )

    ai_source_group = parser.add_argument_group("engine choices")
    ai_source_mutex_group = ai_source_group.add_mutually_exclusive_group()
    ai_source_mutex_group.add_argument("--ollama", "-o", dest="ai_source", action="store_const", const=AiSource.OLLAMA, help="use the Ollama AI agent")
    # ai_source_mutex_group.add_argument("--cursor", "-c", dest="ai_source", action="store_const", const=AiSource.CURSOR, help="use the Cursor AI agent (default)")
    ai_source_mutex_group.add_argument("--claude", "-k", dest="ai_source", action="store_const", const=AiSource.CLAUDE, help="use the Claude AI agent")
    ai_source_mutex_group.add_argument("--debug-mode", "-D", dest="ai_source", action="store_const", const=AiSource.DEBUG, help="use the debug mode")
    parser.set_defaults(ai_source=AiSource.CLAUDE)

    parser.add_argument("--editable", '-e', action="store_true", default=False, help="allow the user to edit the generated response (default: %(default)s)")
    
    output_type_group = parser.add_argument_group("output type choices")
    output_type_mutex_group = output_type_group.add_mutually_exclusive_group()
    output_type_mutex_group.add_argument("--pr-description", "-p", dest="output_type", action="store_const", const=OutputType.PR_DESCRIPTION, help="generate a pull request description (default)")
    output_type_mutex_group.add_argument("--branch-off-main", "-b", dest="output_type", action="store_const", const=OutputType.BRANCH_OFF_FROM_MAIN_ARGUMENTS, help="generate `git branch-off` arguments")
    parser.set_defaults(output_type=OutputType.PR_DESCRIPTION)

    test_mode_group = parser.add_argument_group("test mode choices (for debugging)")
    test_mode_mutex_group = test_mode_group.add_mutually_exclusive_group()
    test_mode_mutex_group.add_argument("--test-mode", "-t", dest="test_mode", action="store_const", const=TestMode.TEST_MODE, help="show stats on prompt and response (add -v for full prompt and response)")
    parser.set_defaults(test_mode=TestMode.NONE)

    VerbosityActionGroupFactory(
        parser, 
        quiet_flags=['--quiet', '-q'],
        verbose_flags=['--verbose', '-v'], 
        debug_flags=['--debug', '-d'], 
        MAX_VERBOSITY=3
    ).add_verbosity_group()

    args = parser.parse_args()

    return args
