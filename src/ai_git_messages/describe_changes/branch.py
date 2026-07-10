import subprocess
import sys
import os
from rich import console

def get_changes_on_branch() -> str:
    ret_str = ""
    for cmd in [
      ["git", "log", "main..HEAD"],
      ["git", "diff"],
    ]:
      p = subprocess.run(
          cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          cwd=os.getcwd(),
      )
      if p.returncode not in (0, None):
          console.log(f"Error: {p.stderr}", style="red bold", end="\n\n")
          sys.exit(p.returncode)
          #raise subprocess.CalledProcessError(p.returncode, p.args, p.stderr)
      ret_str += p.stdout.strip() + "\n"
    return ret_str
