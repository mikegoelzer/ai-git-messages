import subprocess
import os
from typing import Literal, Optional
from pydantic import BaseModel
from enum import Enum

def get_changes_on_main() -> str:
    ChangeType = Literal["added", "modified", "deleted", "renamed"]
    
    spacer_str = "=" * 80

    class Change(BaseModel):
        file: str
        action: ChangeType
        diff: Optional[str]
        file_contents: Optional[str]
    
        def get_prompt_fragment(self) -> str:
            """
            Returns a string that can be used as part of a prompt to describe the change.
            """
            class SpacerType(Enum):
                BEGIN_DIFF = "BEGIN DIFF"
                END_DIFF = "END DIFF"
                BEGIN_FILE_CONTENTS = "BEGIN FILE CONTENTS"
                END_FILE_CONTENTS = "END FILE CONTENTS"

            get_begin_spacer_line = lambda filename, spacer_type: ("-" * 20)+f" {spacer_type}: {filename} "+("-" * 20)+"\n" # noqa: E731
            get_end_spacer_line = lambda filename, spacer_type: ("-" * 20)+f" {spacer_type}: {filename} "+("-" * 20)+"\n" # noqa: E731
            
            diff_str = f"{get_begin_spacer_line(self.file, SpacerType.BEGIN_DIFF.value)}{self.diff if self.diff else '<diff unavailable>'}\n{get_end_spacer_line(self.file, SpacerType.END_DIFF.value)}\n"
            file_contents_str = f"{get_begin_spacer_line(self.file, SpacerType.BEGIN_FILE_CONTENTS.value)}{self.file_contents if self.file_contents else '<file contents unavailable>'}\n{get_end_spacer_line(self.file, SpacerType.END_FILE_CONTENTS.value)}\n"
            
            s = f"Change: '{self.file}' was {self.action}\n"
            if self.action in ["modified"]:
                s += f"Diff of the {self.action} file '{self.file}':\n"
                s += diff_str
                s += f"Complete contents of the {self.action} file '{self.file}':\n"
                s += file_contents_str
            elif self.action in ["deleted"]:
                s += f"Complete contents of the {self.action} file '{self.file}' in patch format:\n"
                s += diff_str
            elif self.action in ["renamed", "added"]:
                s += f"Complete contents of the {self.action} file '{self.file}':\n"
                s += file_contents_str
            return s
    
    def get_all_changes_on_current_branch() -> list[Change]:
        """
        Returns a list of Change objects that represent the changes on the current branch,
        including files added, modified, deleted, and renamed.
        """

        #
        # Helper function to generate a list of Change objects for a given action.
        #
        def mk_change_list(generate_list_cmd: list[str], diff_cmd: Optional[list[str]], action: ChangeType) -> list[Change]:
            """
            Helper function to generate a list of Change objects representing all changes on the current branch

            Args:
                generate_list_cmd: the command that will generate a list of file paths
                diff_cmd: the command that will generate a diff for each given file path, 
                    or None if the diff is not desired in the Change object
                action: what to set the action field to in the Change object

            Returns:
                a list of Change objects
            """
            ret_list = []
            p = subprocess.run(
                generate_list_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
            )
            file_names_list = p.stdout.strip().split("\n")
            for f in file_names_list:
                if diff_cmd:
                    p2 = subprocess.run(
                        diff_cmd + [f],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=os.getcwd(),
                    )
                    diff = p2.stdout.strip()
                else:
                    diff = None
                p3 = subprocess.run(
                    ["cat", f],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.getcwd(),
                )
                file_contents = p3.stdout.strip()
                ret_list.append(Change(file=f, action=action, diff=diff, file_contents=file_contents))
            return ret_list

        #
        # generate lists of changes of each type
        #
        untracked_files_added = mk_change_list(
            ["git", "ls-files", "--others", "--exclude-standard"],
            # each file name will be appended to the end of this command
            ["git", "diff", "--"],
            "added",
        )
        tracked_files_modified = mk_change_list(
            ["git", "diff", "--name-only"],
            # each file name will be appended to the end of this command
            ["git", "diff", "--"],
            "modified",
        )
        tracked_files_deleted = mk_change_list(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=D"],
            # each file name will be appended to the end of this command
            ["git", "diff", "--cached", "--diff-filter=D", "--"],
            "deleted",
        )
        tracked_files_renamed = mk_change_list(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=R"],
            None,
            "renamed",
        )
        return untracked_files_added + tracked_files_modified + tracked_files_deleted + tracked_files_renamed

    #
    # get the changes on the current branch which is presumed to be main
    #
    changes: list[Change] = get_all_changes_on_current_branch()
    ret_str = f"{spacer_str}\n"
    for c in changes:
        ret_str += c.get_prompt_fragment() + f"{spacer_str}\n"
    return ret_str
