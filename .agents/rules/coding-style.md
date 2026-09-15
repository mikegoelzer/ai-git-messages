---
description: Python coding conventions for this project
globs: ["*.py"]
alwaysApply: false
---

### Important Coding Rules

## Source Code Comments

Both too few and too many source code comments are equally harmful to code readability.  When the purpose or reasoning behind a given piece of code would be unclear to an intelligent person, even one with substantial experience in software development, absent a source code comment, a comment is often an excellent tool.  However, source-level comments have a cost too:  code becomes larger and more bloated, it is more time consuming to read through it, and they can become outdated quickly when an implementation changes but the comments around it are not updated.

When trying to decide how many and which source-level comments to add, the most important question to ask is "will this comment be valuable to a reader who is trying to understand the code?"  If the answer is yes, add the comment.  If the answer is no, do not add the comment.

Whenever an agent writes new code or modifies existing code, comments should adhere to the rules and principles described below.

- **Docstrings**:  When writing function docstrings, we require one of three formats:
  - **Omit the docstring entirely**: if the function is so trivial that no one could reasonably be expected to benefit from reading it.  (Exception:  public API methods that are used to generate developer documentation by automated means often warrant a docstring simply as a formality.)
  - **A single sentence, a short phrase, or at most 1-2 sentences**:  sometimes a docstring is needed but only to convey one thing about the function. For instance, if a function looks unncessary but removing it will lead to a non-obvious failure, then just noting that in 1 sentence or less may be both very helpful and fully sufficient.
  - **A Google-style docstring**:  if the function is not trivial and warrants more than just a single sentence, then we try to employ a standardized Google-style docstring format: (1) a short one-sentence summary at the top, (2) an `Args:` section listing each argument and what it does, and (3) a `Returns:` section.  If the function raises exceptions to indicate specific anomalous states, a `Raises:` section should be used as well.  

Good example of the Google-style docstring format:

  ```python
  def screen_friendly_value(value: int, formatter: Callable[[int], str], label: str = "") -> str:
      """Format value and return a string, possibly with a label prepended.
      Used by callers who need to control how a number is formatted
      and labeled based on the type of display screen they are printing to.

      Args:
          value: numeric value 
          formatter: function to convert value to a string appropriately for 
            caller's display hardware
          label (optional): label to prepend to the formatted value

      Returns:
          a string of the form ``label: formatted_value`` or ``formatted_value``
          (if label is supplied)
      """
  ```

The above example is good because:
  - The docstring follows the Google style guide for docstrings as mentioned above.
  - The docstring doesn't merely state what the function does, it also alludes
  to the purpose of the function which is apparently related to display hardware.
  - The docstring illustrates, by using two examples, what the behavior is when no label
  is provided -- in some contexts, this might be non-obvious to the reader.
  - The function omits a `Raises:` section even though pressing Ctrl+C during execution and many other such common things would actually raise. Discretion is appropriate with `Raises`; if a function does not explicitly use exceptions to alter control flow, they do not deserve special attention.

Example of a bad function docstring:

```python
def arithmetic_operation(a: int, b: int) -> int:
    """
    Performs integer addition of two numbers.

    Args:
        a: the first number to add
        b: the second number to add

    Returns:
        sum of a and b
    """
    return a + b
```

Why the above example is wrong:
  - The function's single line of code is too trivial to warrant a docstring.
  - Had self-documenting code principles been employed, the function would have just been renamed 
  `add()`, obviating the need for any docstring at all.
  - The docstring ultimately just wastes the reader's time.  Not only does it take longer to read and understand the trivial code it documents, it fails to answer the one question that almost any reader would actually be wondering, which is why such a simple algebraic expression needed to be factored ou in the first place.

Additional docstring/comment rules:
- Include an Examples section with >>> notation when the function's behavior is not obvious from the signature alone.
- For NDArray fields, document shape and axis semantics
- `__init__.py` files should be included in all packages even if they contain no code, and should have ~1 sentence docstring that states the module's purpose
- Methods on classes follow the same function docstring rules
- Add a blank line between the summary and the body if the docstring has multiple sections

### SQL Style

- **SQL Capitalization**: Always write SQL keywords like `SELECT` or `INSERT` and functions in UPPERCASE, but keep column names, table names, aliases, and other identifiers in lowercase: 
  - **UPPERCASE**: Keywords (SELECT, FROM, WHERE, JOIN, ORDER BY, GROUP BY, AS, AND, OR, IN, ON, etc.) and functions (COUNT, SUM, AVG, MAX, MIN, COALESCE, CURRENT_TIMESTAMP, etc.)
  - **lowercase**: Table names, column names, aliases, view names, trigger names, variable names and the like. This is illustrated below.

  Example:
  ```sql
  SELECT COUNT(*) AS count, AVG(ruin_probability) AS avg_ruin
  FROM simulation_case_run_status r
  JOIN simulation_result res ON r.id = res.id
  WHERE r.unavailable = TRUE OR r.paths_completed > 0
  ORDER BY res.ruin_probability DESC
  ```

Notice how the word "count" is capitalized when it is used as a SQL function (`COUNT(*)`), but lowercase when it is used as an alias or column name (`AS count`).

## Miscellaneous

### Python Style

- Use type hints on all function signatures
- Prefer dataclasses over plain dicts for structured data

### Important Communication and Teamwork Rules

- **Comments and docstrings should focus on current design facts, not the history of the work**: When writing a comment, docstring, readme.md file, etc., put yourself in the reader's shoes and focus on what is likely to be important to them.  Irrelevant context often just confuses the reader or wastes time. For example, saying "Implemented in accordance with Decision 72 of Appendix G of the plan" does little to help most readers who weren't involved in that planning process.  Focus instead on explaing information that is actually useful and important to the reader.  For example, if Decision 72 was that no function may take more than 1 microsecond to complete on certain hardware, it is far more useful to explain that the function has been designed to meet a strict performance deadline of 1 microsecond and not bother the reader with some bureacracy from the past.

  For example, a comment like "Despite being called `bubble_sort()`, this function actually deletes all tables in the database" is a good example of something that might really have arisen due to Appendix G, but where the surprising nature of the information is much more important than what particular design document created that odd requirement.

- **Plans and design docs**: Write implementation plans and design specs to the repository's `plans/` directory (e.g. `plans/YYYY-MM-DD-<topic>-design.md`) unless the human directs otherwise.
- **PR Naming**: The most important thing about a pull request title is that it succienctly captures enough about what changed in that PR that other engineers can quickly decide for themselves whether those changes are relevant to them without having to expend a great deal of time or effort.  Names like the following are completely useless:
  - "Sub-PR #1", "Sub-PR #2", etc.
  - "SP1", "SP2", etc.
  - "Part 1", "Part 2", etc.
  The problem with these names is that (1) terms like "sub" or "#3" do not mean much to anyone besides you, the author of that PR message, (2) they presuppose a lot of context that even diligent and well-informed readers might not have.  #1 and #2 of what?  Is it #2 out of 3, or 2 out of 30?  What does "SP" stand for -- service pack? Swedish pancakes?
  - Here are examples of good PR titles:
    - "docs: add CUDA example"
    - "feat: add resolve cmd"
    - "feat: transition to new db schema"
Those 3 are good because they are only a few words, can be glanced at quickly, and provide some idea of what the PR is probably about.  Note that these titles alone are not sufficient to determine, from the title alone, what the full details of the PR are -- and that's totally fine.  They are sufficient for someone to be able to decide whether to learn more or not.
- **Git commit messages**: Do NOT add `Co-Authored-By: Claude` or `Generated by Codex` (or any other AI co-author / "Generated with" attribution) lines to commit messages, issues or PR descriptions.  The goal of the project is not to provide free advertising to AI companies who have nothing to do with the project and are not supporting it.  While we do not seek to hide the involvement of AI collaborators, your specific affiliation or employer is not relevant to our work together.
- **Commit and PR message issues**:  Please do not mention humans using constructs like `@github_username` or a person's name.  For the same reason, commit messages, PR's and issues should not reference to "the human" or "the owner" or similar.  Instead, it is preferable to say "The plan was approved" or "The feature was implemented".
- **Merging PRs**: When an agent merges a PR, generally you should use **squash merge** (`gh pr merge <N> --squash`), though use your judgment to decide whether that maxim applies in your specific situation.
- **Do not discard working-tree changes you did not make until you have determined who did and whether they are important**: If `git status` shows a modified file and you're not sure where it came from, do NOT do `git restore` / `git checkout --` / `rm` without asking the human about it first. It may be the human's uncommitted, in-progress edit (often made outside the session).
