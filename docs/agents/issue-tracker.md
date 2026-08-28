# Issue tracker: GitHub

Issues and specs live in GitHub Issues at `assle/traffic-light`.

Use the `gh` CLI with `--repo assle/traffic-light` when the local checkout has no configured remote.

## Conventions

- Create: `gh issue create --repo assle/traffic-light --title "..." --body "..."`
- Read: `gh issue view <number> --repo assle/traffic-light --comments`
- List: `gh issue list --repo assle/traffic-light --state open`
- Comment: `gh issue comment <number> --repo assle/traffic-light --body "..."`
- Label: `gh issue edit <number> --repo assle/traffic-light --add-label "..."`
- Close: `gh issue close <number> --repo assle/traffic-light --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Skill conventions

When a skill says “publish to the issue tracker,” create a GitHub issue.

When a skill says “fetch the relevant ticket,” read the corresponding GitHub issue and its comments.

For wayfinding work, use one issue labelled `wayfinder:map` as the map and link child issues through GitHub sub-issues. Use native issue dependencies for blocking relationships where available.
