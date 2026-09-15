---
name: writing-docs
description: Conventions for writing documentation in this repository (docs/ and readmes). Deliberately minimal for now; grows as documentation work accumulates.
---

# Writing documentation in this repo

A starter checklist — woefully incomplete on purpose. Add to it when a documentation convention gets decided, so the decision is not re-litigated.

## IMPORTANT: Be Brief

- **Being brief helps the human.** - A major problem with agent-written documentation is that it tends to be too verbose, unnecessarily detailed, and not helpful to humans because it includes too much information that they don't need to be told.
- **Be more concise than feels comfortable** - Prefer a few words to a sentence; hang a short clause off an existing sentence rather than adding a new one.
- **No TMI (too much information)** - Omit history and justification in documentation. No one cares that a bug used to exist and fixing that bug is why the software now works the way that it does. What they care about is simply, how does the software work and how do I use it? 
- **Before writing any documentation text, ask yourself these questions** - Who is the intended reader? What are they looking for that has caused them to read this document that I am writing?  Users want to know how to use the software. Engineers want to know how the software works.
  - **For engineer-focused documentation** - Explain the high level and let the code speak for itself. Explain "why we did it this way" only when the choice is unusual enough to confuse a future reader; usually it is not.

## Line Breaks Rules for Markdown

- Never insert hard line breaks inside a paragraph — markdown viewers wrap automatically. Line breaks between sections, around code blocks, and the blank line ending a paragraph are fine.

## Documentation Conventions

- **Readme files are lowercase** `readme.md`, always — in `docs/`, in
subdirectories, everywhere (cf. `docs/readme.md`,
`docs/installation_methods/readme.md`). Never `README.md`.
- `docs/` **documents the system as it is;** `plans/` **is task lists.** Never
cite a `plans/` file as authority in documentation, and never write
documentation that only makes sense to someone who has read a plan — see
`plans/AGENTS.md`.
- **Describe current design facts, not the history of the work.** No "as of
PR #NN" narration, no decision-number references; state the fact itself.
- **Link between docs with relative paths** (`[text](other-doc.md)`), so
links survive both GitHub rendering and local viewing.
