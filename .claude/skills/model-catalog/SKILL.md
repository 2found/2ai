---
name: model-catalog
description: Check and update 2ai's model tables (context windows, reasoning-effort levels, long-context price steps) against oh-my-pi's model catalog. Use when a vendor ships or retires models, when a run is summarized too late or too early, when a provider refuses an effort level, or as a periodic check (every few weeks).
---

# Model catalog

2ai's tables are the source of truth for what a provider's own model list does
not say. A live value always wins: `ListModels` keeps a window, effort list or
price step the vendor reported and fills only the gaps from the tables.

| Table | File | Read by |
|---|---|---|
| `contextWindows`, `vendorContextWindows` | `core/ai/window.go` | `ContextWindowFor`: where a run is summarized |
| `reasoningEfforts` | `core/ai/model_efforts.go` | `ReasoningEffortsFor`, `ClampReasoningEffort`: the levels a host offers and sends |
| `LongContextThresholdFor` | `core/ai/model_efforts.go` | the input size past which a model bills more; a host summarizes before it |

Hosts read them through those functions and through `Model.ContextWindow`,
`Model.ReasoningEfforts` and `Model.LongContextThreshold` from `ListModels`.
2agent clamps each rung's effort with them, offers only served levels in its
run picker and AI settings, and summarizes before the price step.

## 1. Compare

```bash
go run ./test/modelcatalog          # fetches packages/catalog/src/models.json from can1357/oh-my-pi
go run ./test/modelcatalog -catalog path/to/models.json -dump decoded.json
```

The command changes nothing. It prints five sections, most urgent first.

## 2. Act on each section

1. **Window larger than the vendor serves.** Always fix. A run is summarized
   too late and the provider rejects it; no retry recovers. Lower the main
   entry when vendors differ and 2ai has no vendor of that name. Otherwise add
   an entry to `vendorContextWindows` for the 2ai vendor that serves less (a
   Claude Code login, Antigravity, Codex, an OpenCode gateway). Mind prefixes:
   a vendor entry for `glm-5` also matches `glm-5.2`.
2. **Unknown to 2ai.** Add the family when an agent would use it as a chat
   model through a vendor 2ai serves. Skip realtime, live-audio, tab
   completion and deep-research models. A window of 1M or more matters only
   to a host that summarizes past 300k, so it can wait.
3. **Effort levels differ from the maker's.** Follow the maker: Anthropic for
   Claude, OpenAI for GPT, Google, xAI, DeepSeek, Moonshot for Kimi, Z.ai for
   GLM, MiniMax, Alibaba for Qwen. An extra level a router or another plan
   lists is that service's mapping, not the model's (Alibaba's plan lists
   GLM without xhigh; Z.ai serves it). A model that does not reason, or
   reasons with no level to choose, gets an empty list (`effortsNone`).
4. **Long-context price step differs.** Update `LongContextThresholdFor`.
5. **Window under 90% of the catalog's.** Informational. It is right when a
   vendor table or a smaller serving vendor explains it; otherwise raise it.

Rules for every edit:

- Match by family prefix (`gpt-5.1-codex` covers `-max` and `-mini`); add a
  member only where it differs. Dated and routed ids (`openai/…`,
  `us.anthropic.…`) resolve on their own.
- Where windows differ, the smaller is the safe side: too small costs an early
  summary, too large a failed run.
- Effort lists are weakest first and use only `ReasoningEffortLevels`, never
  `none`; `TestReasoningEffortsAreLevelsInOrder` enforces it.
- Never invent an id. The tables annotate ids a vendor lists.

## 3. Verify and record

```bash
gofmt -l core test
go test ./core/ai
go run ./test/modelcatalog   # section 1 must print 0
```

Add a test case for each changed family to `core/ai/window_test.go` or
`core/ai/model_efforts_test.go`. Update the catalog date in the comments of
`window.go` and `model_efforts.go`. Add an `## Unreleased` entry to
`CHANGELOG.md` that names the families and what a run gains (for example: "a
run on GPT-5.1 Codex is summarized before 272k instead of failing past it").

Publishing a release needs an explicit request (AGENTS.md). After one, a host
picks the tables up by bumping its 2ai pin; 2agent needs no change of its own.
