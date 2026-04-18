#!/usr/bin/env python3

"""
Regression tests for Rich markup handling.

When the prompt (built from `git diff` output) or the AI response contains
bracketed substrings like ``[/EXPANDED]`` or ``[bold]``, Rich's
``Console.log`` previously tried to parse them as markup tags and raised
``rich.errors.MarkupError``. These tests ensure that verbose logging of
untrusted content does not crash the process.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest

from ai_git_messages.ai_git_messages import (
    OutputType,
    cursor_generate,
    validate_resp_str_and_return_json_str,
)


MARKUP_LIKE_TEXT = (
    "Some content [/EXPANDED] with a stray closing tag "
    "and a lone [bold] open tag, plus [link=https://example.com] foo."
)


def _make_fake_cursor_response(result_text: str) -> SimpleNamespace:
    """
    Return an object shaped like ``subprocess.run``'s return value, whose
    ``stdout`` is the JSON envelope that ``cursor-agent -p --output-format
    json`` would produce.
    """
    payload = json.dumps({"result": result_text})
    return SimpleNamespace(returncode=0, stdout=payload, stderr="")


def test_cursor_generate_tolerates_markup_like_response(monkeypatch):
    """
    cursor_generate should not crash when both the prompt and the response
    contain substrings that look like Rich markup tags (e.g. ``[/EXPANDED]``).
    """
    # Make the prompt contain markup-like text via the changes-on-branch helper.
    monkeypatch.setattr(
        "ai_git_messages.ai_git_messages.get_changes_on_branch",
        lambda: f"diff content {MARKUP_LIKE_TEXT}",
    )

    # Valid PR description JSON, but include markup-like text inside the body.
    cursor_result = json.dumps(
        {
            "title": "Test title with [/EXPANDED]",
            "body": f"- a bullet {MARKUP_LIKE_TEXT}\n- another bullet",
        }
    )

    fake_completed = _make_fake_cursor_response(cursor_result)

    def fake_run(cmd, *args, **kwargs):
        assert cmd[0] == "cursor-agent", (
            f"cursor_generate should invoke cursor-agent, got {cmd!r}"
        )
        return fake_completed

    monkeypatch.setattr(subprocess, "run", fake_run)

    # verbose=True is the path that previously raised MarkupError while
    # logging the prompt / response via rich.console.Console.log.
    returned = cursor_generate(OutputType.PR_DESCRIPTION, verbose=True)

    # The function should have returned the model's raw result unchanged
    # (no ```json fence was used in our fake response).
    assert returned == cursor_result
    assert "[/EXPANDED]" in returned


def test_validate_resp_str_handles_markup_like_body(capsys):
    """
    The verbose log of the parsed PRDescription should not crash when the
    body contains Rich-markup-like brackets.
    """
    resp = json.dumps(
        {
            "title": "t",
            "body": f"body with {MARKUP_LIKE_TEXT}",
        }
    )
    out = validate_resp_str_and_return_json_str(
        resp, OutputType.PR_DESCRIPTION, verbose=True
    )
    assert out is not None
    parsed = json.loads(out)
    assert "[/EXPANDED]" in parsed["body"]


def test_validate_resp_str_handles_markup_like_branch_fields(capsys):
    """
    Same as above but for the branch-off-main output type.
    """
    resp = json.dumps(
        {
            "feat_or_fix": "feat",
            "branch_name": "add-expanded-tag",
            "commit_message": f"commit with {MARKUP_LIKE_TEXT}",
        }
    )
    out = validate_resp_str_and_return_json_str(
        resp, OutputType.BRANCH_OFF_FROM_MAIN_ARGUMENTS, verbose=True
    )
    assert out is not None
    parsed = json.loads(out)
    assert "[/EXPANDED]" in parsed["commit_message"]


@pytest.mark.parametrize(
    "markup_fragment",
    [
        "[/EXPANDED]",
        "[bold]",
        "[/bold]",
        "[link=https://example.com]click[/link]",
        "[not a tag",
        "nested [[/EXPANDED]] content",
    ],
)
def test_cursor_generate_tolerates_various_markup_fragments(
    monkeypatch, markup_fragment
):
    """
    Parametrized smoke test covering several markup-looking fragments.
    """
    monkeypatch.setattr(
        "ai_git_messages.ai_git_messages.get_changes_on_branch",
        lambda: f"diff {markup_fragment}",
    )

    cursor_result = json.dumps(
        {"title": f"t {markup_fragment}", "body": f"b {markup_fragment}"}
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _make_fake_cursor_response(cursor_result),
    )

    returned = cursor_generate(OutputType.PR_DESCRIPTION, verbose=True)
    assert markup_fragment in returned
