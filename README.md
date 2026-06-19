# Greg Overton's Claude Code plugins

A personal [Claude Code](https://code.claude.com) plugin marketplace. Add it
once, then install any plugin below — updates are picked up with
`/plugin marketplace update`.

## Add the marketplace

```
/plugin marketplace add The-Greg-O/claude-plugins
```

## Plugins

| Plugin | What it does |
|---|---|
| [`recursive-improvement-loop`](plugins/recursive-improvement-loop) | Run long-running agentic optimization loops — fresh-context iterations against a trusted measurement harness (hard gates, champion ratchet, statistical plateau stop), a living lab notebook, live dashboard, and full audit trail. Domain-agnostic; bring one `evaluate.py`. |

Install one with:

```
/plugin install <name>@the-greg-o
```

## License

[MIT](LICENSE)
