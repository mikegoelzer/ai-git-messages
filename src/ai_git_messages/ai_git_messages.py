#!/usr/bin/env python3

import sys
from typing import Literal
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.text import Text
from rich.json import JSON
import json
import logging
from importlib.metadata import version, PackageNotFoundError
from curvpyutils.logging import configure_rich_root_logger
from .types import AiSource, OutputType
from .util import get_edited_response
from .parse_args import parse_args
from .ai.run_model import run_model

log = logging.getLogger(__name__)







def main():
    args = parse_args()
    configure_rich_root_logger(args.verbosity)

    try:
        s = run_model(args.ai_source, args.output_type, args.verbose)
        if s is None:
            raise ValueError("run_model returned nothing")
    except Exception as e:
        log.exception(e, exc_info=True)
        sys.exit(1)

    if args.verbose:
        console.log(f"writing to stdout:", style="bold", end="")
        console.log(JSON(s), highlight=True, end="\\n\\n")
    
    if args.editable:
        s = get_edited_response(s, args.verbose)
    
    if s is None:
        log.critical("Unable to generate valid output")
        sys.exit(1)

    # emit the JSON directly to stdout
    print(s)
    sys.exit(0)

if __name__ == "__main__":
    main()
