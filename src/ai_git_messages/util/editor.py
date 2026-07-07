import subprocess
import os
import tempfile
import json
from typing import Optional
from rich.console import Console
import shlex
from rich.prompt import Confirm
from rich.json import JSON

console = Console()

def _run_editor(path: str) -> None:
    # VISUAL > EDITOR > fallback
    editor = (
        os.environ.get("VISUAL")
        or os.environ.get("EDITOR")
        or "vi"
    )

    # $EDITOR might be "vim -u NONE" etc.
    cmd = shlex.split(editor) + [path]

    # Attach editor I/O directly to the controlling terminal
    tty_fd = os.open("/dev/tty", os.O_RDWR)
    try:
        subprocess.run(
            cmd,
            check=True,
            stdin=tty_fd,
            stdout=tty_fd,
            stderr=tty_fd,
            # commented out b/c setting to False is a bit dubious here; 
            #   however, it would only matter if we had fd's besides 0/1/2
            #   in this program (sockets, pipes, files, etc.):
            # close_fds=False,
        )
    finally:
        os.close(tty_fd)

def _edit_json_str(json_str: str) -> Optional[str]:
    """
    Edits the given JSON string in a text editor.
    """
    updated_json_str: Optional[str] = None
    try:
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        f.write(json_str.encode())
        f.flush()
        _run_editor(f.name)
        with open(f.name, "r") as f:
            updated_json_str = f.read()
        if updated_json_str is None:
            console.log("Edit failed", style="red bold", end="\n\n")
            return None
        else:
            resp_obj = json.loads(updated_json_str)
            return json.dumps(resp_obj, indent=4)
    except Exception as e:
        console.log(f"Error: {e}", style="red bold", end="")
        return None
    finally:
        if f is not None:
            f.close()
        if os.path.exists(f.name):
            os.unlink(f.name)

def get_edited_response(resp_str: str, verbose: bool = False) -> str:
    s = resp_str
    while True:
        s = _edit_json_str(s)
        if s is None:
            again = Confirm.ask(
                "Try again?", 
                choices=["y", "N"], 
                default="n",
                show_choices=False,
                show_default=False,
                case_sensitive=False)
            if again:
                continue
            else:
                raise SystemExit(1)
        else:
            if verbose:
                console.log(f"post-edit response going to stdout:", style="bold", end="")
                console.log(JSON(s), highlight=True, end="\\n\\n")
            break
    return s
