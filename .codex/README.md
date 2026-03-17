# Codex Control Plane

This directory is the control plane for agent behavior in FastCuda.

## Order of Use

1. `instructions/`
   Load durable project rules first.
2. `prompts/`
   Select the task entrypoint.
3. `agents/`
   Choose the role that best matches the task.
4. `skills/`
   Add focused workflow modules.
5. `hooks/`
   Run guardrails before and after benchmark or profiling tasks.

## Practical Rule

If a task can be handled through an existing script in `scripts/`, prefer that
script over ad hoc command construction.
