from ai_git_messages.types import OutputType
from ai_git_messages.describe_changes import (
    get_changes_on_main,
    get_changes_on_branch,
)

def get_prompt(output_type: OutputType, verbosity: int = 0) -> str:
    if output_type == OutputType.BRANCH_OFF_FROM_MAIN_ARGUMENTS:
        prompt = """
You are a helpful assistant that generates git branch names and commit messages based on the provided 
changes that have been made to this branch.  Read through the changes and select the most appropriate 
values for these three fields:
 - "feat_or_fix": this field should be set to "feat" if the changes represent a new feaeture, 
 or "fix" if the changes represent a bug fix. Those are the only two valid values for this field.
 - "branch_name": a short, descriptive name for the new branch that I will checkout with the changes 
 described below. The branch name should be a hyphen-separated string of 5 or fewer words describing the 
 changes.
 - "commit_message": a short, descriptive commit message that describes the changes you'll see below.

Requirements:
- Your final response should be a JSON object with three string fields: "feat_or_fix", "branch_name", and "commit_message".
- Your final response should contain no other text.

Below is a list of every changed files and what has changed. Where possible, the entire contents of the
file plus the diffs are provided to help you understand what change was made.

{changes}
""".format(changes=get_changes_on_main())
    elif output_type == OutputType.PR_DESCRIPTION:
        prompt = """
You are a helpful assistant that generates a pull request description based on the provided changes.

Requirements:
- Your final response should be a JSON object with two string fields: "title" and "body".
- Your final response should contain no other text.
- The --body portion may contain ascii \n to indicate where newlines should go.
- The --body portion should be a bullet Markdown list of changes.
- The title should be succient and general, not a list of changes. Try to sum of all the changes in a single phrase like ("improved scripts" or "added feature X")

Here are the changes:

{changes}
""".format(changes=get_changes_on_branch())
    else:
        raise ValueError(f"Invalid output type: {output_type}")
    return prompt
