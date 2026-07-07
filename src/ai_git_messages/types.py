from enum import Enum
from pydantic import BaseModel
from typing import Literal
import json
from rich.text import Text

class AiSource(Enum):
    OLLAMA = "ollama"
    CURSOR = "cursor"
    CLAUDE = "claude"
    DEBUG = "debug"

class OutputType(Enum):
    BRANCH_OFF_FROM_MAIN_ARGUMENTS = "branch_off_main"
    PR_DESCRIPTION = "pr_description"

    @property
    def desc(self) -> str:
        return {
            OutputType.BRANCH_OFF_FROM_MAIN_ARGUMENTS: "branch off from main arguments",
            OutputType.PR_DESCRIPTION: "pull request description",
        }[self]

class PRFromBranchDescription(BaseModel):
    title: str
    body: str

    def __str__(self) -> str:
        return f"Title:\n{self.title}\nBody:\n{self.body}"

    def __rich__(self) -> Text:
        title_text = Text(self.title, style="bold")
        body_text: list[Text] = []
        for ln in self.body.split("\n"):
          body_text.append(Text('  '))
          body_text.append(Text(ln, style="bold"))
          body_text.append(Text("\n"))
        t = Text.assemble(
          "Title:",
          title_text, 
          "\n", 
          "Body:\n",
          *body_text,
        )
        return t

    def __repr__(self) -> str:
        return f"PRDescription(title={self.title}, body={self.body})"

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=4)

class ChangesOnMainDescription(BaseModel):
    feat_or_fix: Literal["feat", "fix"]
    branch_name: str
    commit_message: str

    def __str__(self) -> str:
        return f"Feature or fix:\n{self.feat_or_fix}\nBranch name:\n{self.branch_name}\nCommit message:\n{self.commit_message}"

    def __rich__(self) -> Text:
        return Text.assemble(
          "Feature or fix:",
          Text(self.feat_or_fix, style="bold"),
          "\n", 
          "Branch name:",
          Text(self.branch_name, style="bold"),
          "\n", 
          "Commit message:",
          Text(self.commit_message, style="bold"),
        )

    def __repr__(self) -> str:
        return f"ChangesOnMainDescription(feat_or_fix={self.feat_or_fix}, branch_name={self.branch_name}, commit_message={self.commit_message})"

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=4)
