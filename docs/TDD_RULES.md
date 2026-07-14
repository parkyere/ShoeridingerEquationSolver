# TDD working agreement

This project is built **test-first, without exception** for domain code.

## The loop

1. **RED** — write a failing test that specifies one new behavior. Run it; watch
   it fail for the right reason.
2. **GREEN** — write the *minimum* production code to make it pass. Nothing more.
3. **REFACTOR** — clean up production and/or tests with the suite green.

No production code is ever written before a failing test demands it.

## Commit discipline (hard rule)

**A commit touches TEST files or PRODUCTION files — never both.**

- A new behavior arrives as **two commits**: first the red test, then the
  production change that turns it green.
- Refactors are their own commits: production refactors (suite stays green) are
  production commits; test refactors are test commits.

Path classification (also enforced by the hook):

| Class | Paths |
|---|---|
| TEST | `tests/**` |
| PRODUCTION | `core/include/**`, `core/src/**`, `app/**` |
| NEUTRAL | everything else (build, docs, config, tooling) |

Neutral files may ride along with either a test or a production commit.

### Commit message convention

```
test:     a new (red) test, or test refactor          -> TEST commit
feat:     production that makes a red test green       -> PRODUCTION commit
refactor: behavior-preserving production cleanup       -> PRODUCTION commit
chore: / build: / docs:   neutral infra                -> NEUTRAL commit
```

Example sequence for one behavior:

```
test:  FFT of a real constant signal is a single DC spike   (RED, tests/)
feat:  implement radix-2 forward FFT                        (GREEN, core/)
refactor: extract bit-reversal permutation helper           (core/)
```

## The guard hook

`tools/git-hooks/pre-commit` rejects any commit that stages both TEST and
PRODUCTION paths. Enable it once:

```sh
git config core.hooksPath tools/git-hooks
```

Genuine infrastructure/bootstrap commits (which legitimately span the tree once)
may bypass with `git commit --no-verify`. The initial scaffolding commit is the
canonical example.

## Humble Object exception

The SDL3 shell in `app/` (window, input, UI, the fullscreen-triangle
presentation, and the demo scheduling in `main.cpp`) is **not** unit-tested — that is the deliberate
Humble Object boundary. It must stay logic-free; any logic worth testing is
pushed down into `core/` and driven test-first there. The shell is verified by
running it — and by the headless `--selftest-*` and `--dump-frame*` arcs.

The GPU side of `app/` is **not** exempt: every `ses_vk` kernel and engine
path is verified against the unit-tested CPU double core by
`sesolver_vkcheck` (a framework-free oracle binary), which runs inside ctest. A GPU
kernel lands together with its vkcheck comparison — the GPU analogue of
red/green.
