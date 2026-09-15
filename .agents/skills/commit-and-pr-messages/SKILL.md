---
name: commit-and-pr-messages
description: How to write commit messages and PR descriptions for this repository — what changed and why, written for any future reader; never personal notes to the repo owner or any individual.
---

# Commit and PR Messages Skill

Use this skill whenever you write a git commit message, a PR title, or a PR description
(or edit an existing one) in this repository.

## The audience rule (the core of this skill)

A commit or PR message is a record **for posterity**, addressed to **any reader of the
repository** — a future maintainer, an auditor, a new contributor, someone studying the
code years from now. It is **never** a channel for passing notes to the repo owner or to
any specific person.

- **Be extremely concise — more concise than feels comfortable** - Commit messages do not need to be full sentences and they do not need to describe everything that was changed.  They should be short and a high level statement of the overall set of changes.  PR descriptions do need to be written in full sentences, but should still focus on giving a high level overview of the changes at the beginning.  Additional sections can then be used to provide more details.  Still, do not feel that you have to explain every minor detail because the human reader is just looking to understand generally what the purpose of the PR is and how it changed the code at a high level. For exact details of what changed, they are just going to read the code itself.
- **Being brief helps the human.** - A major problem with agent-written documentation is that it tends to be so verbose that humans don't read it all.  At all times, strive to be as brief as possible while still conveying what is important.
- **No TMI (too much information)** - In documentation like `docs/` and markdown files, we generally try to avoid spending a lot of time on the "why we did it this way" part. PR descriptions are the one place where it is legitimate to explain the "why" part.  Still, please be concise in doing so.
- **You are not writing a personal email** - Words that address an individual — "you", "your", "as you asked", "for your review" — do not belong in these messages. If you need to tell the human something (status, caveats, follow-ups, questions), say it in your harness output (Claude Code, Codex, etc.), not in the commit or PR text.
  - Section headers must be reader-neutral: write `## Notes`, never `## Notes for @username` or similar.
  - Rewrite person-addressed sentences into descriptions of the codebase. Wrong: "Your interleaved commits on this branch ride along untouched, as requested." Right: "Some commits on this branch modify planning documents for future iterations of this effort."
  - The string `@username` (any Github username) is appropriate **only** as attribution when explaining the *why* behind a change — a substitute for "the owner requested…". Example: "The fill-down tolerance was removed because @github_user123 specified that a base_case source must cover every year through MAX_AGE." Do not use it as an addressee.

## What the message must contain

Record what the change did to the code base and why:

- How the code changed (what was added, removed, reshaped).
- Why those changes were made (the motivating problem, decision, or requirement — with
  attribution where the "why" is a person's decision).
- Where the detailed documentation lives (design docs, plans, module docstrings).
- What new gotchas, invariants, or fragile points were introduced, and where those are
  documented.
- Verification that was performed, stated factually (test counts, end-to-end runs).

Do not include: conversational asides, promises to the reader, status-report phrasing
("I'll follow up"), or anything whose only audience is one person.

## PR descriptions: clickable, verified file links

PR descriptions render in the GitHub/GitLab web UI, so file references must be clickable
links, not bare backticked paths.

- Wrap every referenced repo file in a markdown link. Prefer a **commit-pinned blob URL**
  so the link survives branch deletion after merge:
  `[plans/2026-06-19-inline-everything-roadmap.md](https://github.com/<owner>/<repo>/blob/<commit-sha>/plans/2026-06-19-inline-everything-roadmap.md)`
  where `<commit-sha>` is the PR head commit (or any commit containing the file).
- **Test every link before submitting.** Verify the file exists at that path in the
  referenced commit, e.g.:
  - `git cat-file -e <commit-sha>:<path>` (exits 0 iff the blob exists), and/or
  - `gh api repos/<owner>/<repo>/contents/<path>?ref=<commit-sha>` (returns 200 iff
    fetchable on GitHub).
  A link that 404s is worse than no link.
- Line-anchored links (`...#L42`) are welcome where a specific location matters; verify
  the line number is current at that commit.
- Make the link pass a **full-body sweep as the final step**, after all other edits:
  rewriting sentences (e.g. fixing audience-rule violations) often surfaces additional
  bare file paths that were easy to miss on a first pass. Grep the finished body for
  backticked path-like strings before submitting.

## Commit messages: short and plain-text friendly

Commit messages are read in terminals (`git log`), so keep them **plain text**:

- Short imperative subject line; a concise body only when the change needs explanation.
- No markdown links, no heavy formatting, no tables. Backticks for identifiers are fine.
- The audience rule above still applies in full.

## Other repository rules that apply to these messages

- No AI-attribution trailers (`Co-Authored-By: Claude`, "Generated with …") — see
  CLAUDE.md.
- SQL keywords UPPERCASE / identifiers lowercase when SQL appears in a message.
- Messages must not narrate the historical sequence of sub-projects as if the reader
  knows it; like code comments, they should make sense to someone who never saw the
  intermediate steps. (Naming the PR's place in a multi-PR effort with a link to the
  plan document is fine — that is documentation, not chat.)
