---
name: multi-vendor-agent-files
description: Where agent-facing instructions, rules, and skills live in this repo so every vendor's agent (Claude, Cursor, Codex, ...) finds one canonical copy. Follow this when creating or moving any agent instruction file.
---

# Multi-vendor agent files

One canonical tree, per-vendor shims. Never fork the content per vendor.

- **Instructions**: the real text lives in `AGENTS.md` (repo root, and
  optionally per-directory, e.g. `docs/AGENTS.md`). Each `AGENTS.md` gets a
  sibling `CLAUDE.md` whose entire content is `@AGENTS.md` (or a symlink —
  either direction is fine as long as there is exactly one real file).
- **Skills**: one per directory under `.agents/skills/<name>/SKILL.md`, with
  YAML frontmatter (`name`, `description`). Vendor discovery goes through
  symlinks at the repo root: `.claude/skills → ../.agents/skills` and
  `.cursor/skills → ../.agents/skills`. Add a new skill by creating the
  directory under `.agents/skills/` — the symlinks pick it up; never create
  a real skill file under `.claude/` or `.cursor/`.
- **Rules**: `.agents/rules/` (e.g. `coding-style.md`, referenced from
  `AGENTS.md`).

Agents that do not auto-load a given location are pointed at it from the
nearest `AGENTS.md` — that pointer, not duplication, is the fallback.
