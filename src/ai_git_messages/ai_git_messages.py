#!/usr/bin/env python3

import sys
from rich.json import JSON
from rich import print as rprint
import logging
from curvpyutils.logging import configure_rich_root_logger
from .util.test_mode import test_mode
from .types import TestMode
from .util import get_edited_response
from .parse_args import parse_args
from .ai.run_model import run_model
from .util.log_console import log_console

log = logging.getLogger(__name__)

def main():
    args = parse_args()
    configure_rich_root_logger(args.verbosity)

    if args.test_mode != TestMode.NONE:
        sys.exit(test_mode(args))
    
    s = run_model(args.ai_source, args.output_type, args.verbosity)

    if args.editable:
        s = get_edited_response(s, args.verbosity)

    if args.verbosity >= 2:
        log_console.log(f"writing to stdout:", style="bold", end="")
        log_console.log(JSON(s), highlight=True, end="\\n\\n")

    if s is None:
        log_console.critical("Unable to generate valid output")
        sys.exit(1)

    # emit the JSON directly to stdout
    print(s)
    sys.exit(0)

if __name__ == "__main__":
    main()
