---
name: prepare-conventional-commit
description: Prepare Git commits that follow the Conventional Commits specification. Use when the user asks to prepare, draft, review, or create a commit message; asks for a conventional commit; wants staged or unstaged changes summarized before commit; or wants help choosing commit type, scope, subject, body, footers, or breaking-change notation.
---

# Prepare Conventional Commit

## Workflow

1. Inspect repository state with `git status --short`.
2. Inspect staged changes with `git diff --cached --stat` and `git diff --cached`.
3. If nothing is staged, inspect unstaged changes with `git diff --stat` and `git diff`, then tell the user no commit is ready unless they asked you to stage files.
4. Identify the smallest honest commit intent from the diff. Do not include unrelated changes in the message.
5. Choose a Conventional Commit type and optional scope.
6. Draft the commit message and explain the choice briefly.
7. Run project tests or focused validation when the user asked you to prepare a real commit and validation is feasible.
8. Execute `git commit` only when the user explicitly asked to create the commit. Otherwise, output the proposed command/message.

## Message Format

Use this structure:

```text
<type>(<scope>): <subject>

<body>

<footer>
```

Use `type: subject` when no meaningful scope exists.

Subject rules:

- Use English unless the repository strongly uses another language.
- Use imperative mood when natural.
- Start lowercase after the colon unless the subject begins with a proper noun, acronym, or code identifier.
- Keep it concise, ideally 50-72 characters.
- Do not end with a period.

Body rules:

- Include a body when it clarifies why the change exists, what behavior changed, or which trade-off matters.
- Wrap lines near 72 characters when writing a multi-line commit message.
- Prefer bullets only when they make a multi-part change easier to read.

Footer rules:

- Use `BREAKING CHANGE: ...` for breaking changes.
- Include issue refs only when visible in the task, branch name, or diff.
- Preserve required project-specific trailers if the repository already uses them.

## Type Selection

- `feat`: user-facing feature or new capability.
- `fix`: bug fix or corrected behavior.
- `docs`: documentation-only change.
- `style`: formatting-only change with no behavior impact.
- `refactor`: code restructuring without behavior change.
- `perf`: performance improvement.
- `test`: tests only, or test infrastructure.
- `build`: build system, packaging, dependencies, lockfiles.
- `ci`: CI configuration or automation.
- `chore`: maintenance that does not fit other types.
- `revert`: revert a previous commit.

Prefer the most user-visible accurate type. If a change includes production code and tests, choose the production-code type rather than `test`.

## Scope Selection

Use a short scope when it improves scanning:

- package, module, command, route, feature, integration, or subsystem name;
- examples: `cli`, `auth`, `api`, `docs`, `deps`, `day5`.

Omit scope when it would be vague, such as `misc`, `changes`, or `update`.

## Breaking Changes

If the diff removes or changes a public API, CLI command, config key, database schema, response format, or behavior that callers may rely on, mark it as breaking:

```text
feat(api)!: require project id for deployments

BREAKING CHANGE: deployment creation now rejects requests without a
project id.
```

Use `!` in the header and a `BREAKING CHANGE:` footer.

## Output

When only preparing a commit, respond with:

```text
Suggested commit:

<commit message>

Why:
<brief reason for type/scope/subject>
```

When creating a commit, use a non-interactive command:

```bash
git commit -m "<type>(<scope>): <subject>" -m "<body>"
```

For multi-line bodies or footers, use repeated `-m` arguments instead of opening an editor.
