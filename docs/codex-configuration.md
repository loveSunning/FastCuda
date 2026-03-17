# Codex Configuration Layout

FastCuda now follows Codex's official configuration layout.

## Official Modules

- `AGENTS.md`
  - root project instructions and workflow guidance
- `.codex/config.toml`
  - project-scoped Codex runtime configuration
- `.codex/rules/*.rules`
  - command and approval rules
- `.codex/agents/*.md`
  - custom subagents stored as Markdown files with YAML frontmatter
- `.codex/skills/*/SKILL.md`
  - project skills stored as `SKILL.md`

## Project Assets That Are Not Codex Config

- `docs/prompts/`
  - reusable prompt briefs
- `scripts/`
  - environment, benchmark, profile, and hook wrappers
- `configs/`
  - benchmark presets and device profiles

## Migration Mapping

- old `.codex/project.toml` -> `.codex/config.toml`
- old `.codex/instructions/*.md` -> root `AGENTS.md`
- old `.codex/hooks/*.toml` -> `scripts/hooks/` scripts only
- old `.codex/tools/README.md` -> regular project docs and `scripts/`
- old `.codex/prompts/` -> `docs/prompts/`

The project keeps prompt files and helper scripts, but they are no longer
presented as first-class Codex configuration modules.
