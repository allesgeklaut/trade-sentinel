# Type-D: the material-news agent — design note v2 (2026-11-01)

Status: WIRED 2026-11-01 (owner approval). Shadow-first: D accumulates as
its own track beside A/B/C; fill columns NULL; the predeclared promotion
rule unchanged. First-night acceptance checklist: tools/list re-pin (done
2026-11-01, recorded above); observe from the first wired night's rows
whether the Mistral route honors temperature 0.6; watch for hang-guard
anomalies (firing is an anomaly to investigate, never normal operation).
SUPERSEDES
`docs/sell-news-exit-design-2026-10-31.md` (kept verbatim for provenance).
Owner-driven (2026-10-31/11-01): "a real agent with native function calling
… I want the model to decide what to fetch and so on — relying on the LLM
for it is the point of it," then, after the first v2 draft reintroduced
harness bounds: "Don't rely on anything already established for this
agent who should do sell decisions. Only rely on research data, if there
is any, and your knowledge on how to design it properly. Also I never
said to token limit this agent." The agent is therefore designed from the
task, on the researched pattern for agent loops (next section) — not from
the A/B/C consult scaffolding. The discipline set — shadow-first, the
frozen prompt, the declared vocabulary, the predeclared promotion rule —
survives; behavioral harness mechanisms do not: NO call caps, NO token
limits, NO search floors, NO query templates, NO wall-clock budgets.
Nothing is wired until the owner approves this note.

## The researched pattern the design follows

Two empirical anchors (checked 2026-11-01, not recalled):

1. **Termination is typed, not counted.** The industry agent loop (the
   OpenAI Agents SDK's runner, LangGraph's graph loop) runs until the
   MODEL emits a final answer; the framework's turn limit is a RUNAWAY
   safety ceiling, not the exit — the LangGraph maintainers' own framing:
   "the loop is RUNNING by default. the agent exits by emitting a finish
   action. the recursion limit becomes a safety ceiling for runaway
   loops, not the primary exit mechanism." OpenAI's default maxTurns=10 is
   documented as a safety limit (nullable), and LangGraph's documented
   remedy for legitimately long runs is to RAISE the limit. This design
   follows that: the agent's exit is its final verdict; the ceiling is a
   generous hang guard whose firing is an ANOMALY to investigate
   (recorded as a fail-safe row), never a budget.
2. **Sampling follows the model family's own guidance.** DeepSeek's
   official R1-series usage recommendations: "Set the temperature within
   the range of 0.5-0.7 (0.6 is recommended) to prevent endless
   repetitions or incoherent outputs" — their own web/app runs 0.6. The
   wire sends temperature 0.6 and every D record declares it. Honest
   caveat: DeepSeek's hosted reasoner API historically IGNORED temperature
   (a compatibility no-op); whether Mistral's route honors it is
   unverified — the records declare what the wire sends, and the wiring
   acceptance note observes the route's behavior. The R1-era "avoid system
   prompts" quirk is deliberately NOT applied (distill-era guidance; the
   Mistral route is OpenAI-standard); the family's prompting guidance IS
   applied — describe the task and the output format in detail, no
   few-shot examples.

## What changed from v1, and why the change costs nothing measurable

The v1 note's own thesis: "an LLM consult is exactly the right instrument
for the judgment 'does this headline invalidate the stated thesis?' — it
is the one step of the sell program that cannot be a rule." v1 then kept
the FETCH as a rule — a control reflex, not a requirement of the problem.
For a fat-tail early-catch problem (the research verdict: 7 events in 11
years, mean +0.80pp/event, median −0.66pp — the large wins are the early
catches), the fetch strategy is itself judgment: what to search, how to
reformulate a query, whether a headline deserves reading the article, when
to stop and decide.

The honest cost accounting, recorded up front: **no measurement is
lost.** News is un-backtestable — no point-in-time reconstruction of a
news wire exists — so the offline reference bar that A/B/C were
pre-measured against never existed for ANY D shape; the only evidence
D can produce is the forward shadow, and the agent feeds that shadow
identically. Provenance moves from determinism to RECORDING (the ledger
section below).

## The event type (unchanged from v1)

**D — the material-news scan.** Scope v1: the MONTHLY book only (the
thesis-discipline names, the same book the C step scans); the sim book's
D is out of scope until the promotion review. Cadence: nightly,
immediately after the type-C step, every trading night.

**Scope extension (2026-11-01, below): the scan now covers the UNION of
both paper books — monthly + daily-core — ONE consult per NAME per night
(a name held in both books is consulted once, never twice).**

## Scope extension 2026-11-01 — the union of both paper books, one consult per name

The monthly book and the daily-core book hold overlapping names (both are
fed from the same qv-mom universe), so scanning the monthly book alone
left daily-core-only names unconsulted. From 2026-11-01 the D scope is
the UNION of `MonthlyPosition` and `DailyCorePosition` (identical shape:
ticker, shares, avg_cost, opened_at):

- `_book_contexts` returns ONE context per NAME over the union (ordered
  by ticker). The context carries the attribution: `books` (sorted, only
  the books that hold the name — e.g. `["daily_core", "monthly"]`) and
  `positions` (per-book `{shares_held, entry, pnl_pct, months_held}`,
  the same computations as the flat fields, per position). The FLAT
  legacy fields stay byte-compatible and are computed from the PRIMARY
  position: monthly if held there, else daily_core.
- WHY one consult per name: the judgment is NAME-level ("does this
  name's stated thesis still hold?") and the thesis condition is
  book-agnostic — two books holding the same name do not double the
  question. The nightly dedupe stays exactly `D:{ticker}:{as_of}`
  (name-level), which IS the not-twice guarantee: a name held in both
  books yields one context, hence one agent run, hence one ledger row.
- The frozen prompt `sellshadow/1-d` is UNCHANGED (no `-d/2`): its
  "one holding of a paper-trading monthly book" phrasing is a documented
  approximation — for daily-core-only names the context block (not the
  prompt) carries the truth via `books`/`positions`; the position facts
  and thesis condition the model judges against are identical in shape.
- The ledger `engine` stays `"monthly"`: the scan is DRIVEN by the
  monthly cycle (`run_monthly_cycle` calls `run_news_scan`), and the
  context's `books` carries the attribution. Known coupling, recorded:
  D lives inside `run_monthly_cycle`, so the whole D step is gated by
  `sim_monthly_enabled` (its own knob remains `SEARXNG_MCP_URL` via the
  knob-off convention inside the scan). Daily-core-only names are
  therefore scanned only on nights the monthly cycle runs.
- Cost delta: nightly spend rises from the monthly book's names to the
  UNION's — the books overlap heavily, so the delta is the daily-core-
  only names (typically a handful), not a doubling.
- Manual consults (`run_manual_consult`, `POST /api/news/consult`) and
  the UI's new background full scan (`POST /api/news/scan`,
  `GET /api/news/scan/status` — fresh `:m` semantics, never deduped)
  consult any union name, including daily-core-only tickers.

## Division of labor — the harness decides who and when; the MODEL decides everything else

The scan TRIGGER stays deterministic: every name in the UNION of both
paper books (monthly + daily-core, one consult per name), every trading
night — the model never decides *whether* a name gets
looked at (a lazy agent that silently skips names is unauditable).
Everything downstream of the trigger is the model's: what to search, how
to reformulate, what to read, how deep, and WHEN IT HAS ENOUGH —
termination is the model's final answer, not a harness counter. The
harness executes tool calls, records the trace, and converts outages
into fail-safe rows; it does not shape the research: no floors, no query
templates, no recivity windows, no budgets.

The context block survives from v1: the position facts (ticker, entry
date/price, current close, PnL, months held, the stated thesis condition)
— the thesis the news is judged against. The recency window is the
model's judgment from those facts (the search tool exposes time_range;
the model picks).

## The tools — native function calling, one path, no budgets on the wire

The four tools of the shared searxng MCP (pinned live 2026-11-01 from
inside the trade-sentinel container; server ihor-sokoliuk/mcp-searxng
2.9.0; stateless — a fresh child per request, ~1.2–3s per call):

- `searxng_web_search` (query, pageno, pages 1–5, time_range
  day/week/month/year, language, safesearch, min_score, num_results
  1–20, categories — the news category is supported —, engines,
  response_format text|json, result_detail compact|full)
- `searxng_search_suggestions`
- `searxng_instance_info`
- `web_url_read` (url, startChar, maxLength, section, paragraphRange,
  readHeadings)

…are exposed to the model as NATIVE FUNCTION CALLS for the duration of
the run. The app executes the model's tool calls against the MCP (the
shared stack on the docker network, `http://searxng-mcp:8808/mcp`; the
deploy work v1 waited on is committed and live). No direct-SearXNG
client is wired for D (the owner's call, unchanged). The MCP transport
sits behind ONE internal module (the app's thin MCP client).

The wire carries NO token limits: no `max_tokens` is ever sent; no call
cap shapes the loop. The payload is exactly: model, messages, tools,
temperature, reasoning_effort.

## The prompt — `sellshadow/1-d`: frozen instruction, free input

Frozen at wiring, versioned beside A/B/C in `app/sellreview.py`; no
post-hoc edits, any change is `-d/2` with a declared reason (the
frozen-prompt discipline, unchanged). What is frozen is the INSTRUCTION;
the INPUT and the research plan are the model's. Contents (the family's
own prompting guidance: describe the task and the output format in
detail, no few-shot):

- Role and task: decide from recent news whether ONE holding's stated
  thesis still holds; the position facts + thesis condition are the
  anchor.
- The materiality instruction (carried from v1): thesis-invalidating
  news only — fraud, guidance cuts, structural breaks, delisting or
  bankruptcy risk, major litigation. Price moves are NOT D events (a
  falling stock is type A's domain, momentum decay is B's); the
  no-momentum-proxy rule keeps D from becoming a second stop-loss.
- The decision vocabulary: `EXIT | HOLD`, no params (C's contract). An
  EXIT must CITE the specific headline(s)/source(s) it acted on, its
  article read and verified — the verdict is auditable against the
  trace. Malformed output is the parser's fail-safe (mechanically
  HOLD), not a prompt threat.
- NOTHING ELSE. No process choreography, no pacing or run-length
  language, no recency windows, no tool hints (2026-11-01, owner's
  direction after the live smokes: "less is more — a clear task, no
  descriptions on how long the task should run"; the process-guidance
  bullet of the first draft was deleted — the max-effort model plans,
  refines, verifies and terminates on its own, proven live in the
  round-3 smoke: first completing verdict, EXIT with three verified
  citations, 29 turns, no coaching).

## The agent loop — model-owned termination; a runaway ceiling, not a budget

The standard agent loop, run to the model's final answer:

1. chat(messages, tools) →
2. tool_calls: execute them (concurrently — the tools are read-only and
   the MCP server is stateless), append the results, go to 1;
3. final text: the verdict — parse, record, done.

- A FAILED call returns its error text to the model as the call's result
  (standard loop semantics: the model sees tool errors and adapts). The
  outage fail-safe keys on the MIDDLEMAN erroring: when every call of the
  run failed and the MCP layer was among the failures, the run is the
  fail-safe row ("mcp unavailable") even if the model verdicted — the
  consult never obtained information. A call that failed through the
  model's OWN unparsable arguments is not an outage: the model saw the
  error and owns its verdict, the same acceptance as a zero-search one.
- **NO call cap.** A safety ceiling (default 50 turns) exists ONLY so a
  wedged run cannot stall the nightly batch; hitting it is recorded as a
  fail-safe row flagged as an anomaly to investigate, never as normal
  operation (the researched pattern: the ceiling is a runaway guard, the
  exit is typed).
- **NO max_tokens, NO wall-clock budget.** The per-call read bound is
  the app transport's own hygiene (the global read timeout).
- **Temperature 0.6** (the family's documented recommendation, chosen to
  prevent the greedy-repetition pathology), **reasoning_effort = the
  operator's knob** (max in this deployment), both recorded in every row.
- Fail-safes are ERROR PATHS only: the MCP outage / transport failure →
  the standard fail-safe row, retried the next night (a middleman outage
  never touches the book); a malformed/unparsable final answer → the
  fail-safe row. Nothing else intervenes.
- v1's "an empty-news D consult is meaningless" is superseded by the
  shape itself: the LOOK is the consult, and "searched, found nothing"
  is a legitimate HOLD with a trace, not a skip.

## The ledger — provenance by recording, not by determinism

Every D row records: event_type "D", engine "monthly", engine_branch =
the counterfactual "hold to next month-end rebuild" (unchanged), the
position facts block, the FULL PER-TURN TRACE (each query issued, a
digest of each result, each URL read), the verdict with its citations,
the model/backend, the temperature and reasoning effort sent, the prompt
version, and news_source = "searxng-mcp". Dedupe: one consult per name
per night — event_id = `D:{ticker}:{session}`.

## Shadow-first — unchanged from v1, and why it makes the rewrite safe

News is un-backtestable: no point-in-time reconstruction of a news wire
exists, so no offline reference bar can ever be measured. Therefore D
starts as a SHADOW consult with NO execution authority — the fill columns
stay NULL forever in the shadow phase; no sell is ever placed by a D
verdict.

The promotion rule (predeclared now, not at promotion time, unchanged): D
earns execution authority only after ≥ 6 months of completed windows AND
a measured positive edge vs the month-end wait on the same
completed-window math as C's verdict (the D-EXIT counterfactual is the
wait itself). Promotion is a versioned, declared change with its own audit
entry — never a silent flag flip. A D record that never promotes is not
a failure: the shadow IS the measurement.

## The boundaries of this rethink

Scoped to D. A/B/C keep their own regime (frozen constants, offline
bars, live execution); revisit them only on D-shadow evidence, as a
versioned, declared change.

## Cost reality (an observation, not a budget)

Typical investigative runs on a single name are expected in the 5–20
tool-call range; at ~1.2–3s per MCP call plus reasoning turns, a nightly
pass over the union of both books (2026-11-01 scope) is minutes to tens
of minutes of wall-clock.
There is no designed budget: the ceiling is the runaway guard, and if
nightly cost ever needs governing, that is an owner decision made from
the observed traces — not a mechanism inside the loop.

## Open items at wiring

1. The MCP client choice: a lean hand-rolled JSON-RPC over httpx
   (stateless server — no session management; httpx already a dep) vs the
   official `mcp` SDK's streamable-http client. Pinned in the wiring
   slice.
2. The backend tools smoke test (verified live 2026-11-01: the Mistral
   route answers the OpenAI tools surface with finish_reason tool_calls;
   re-pin in the acceptance note).
3. The runaway-ceiling default (50 turns) — an engineering guard, not a
   behavioral knob; its level is declared here.
4. Step-0 re-pin: `tools/list` (names/params) in the acceptance note
   (pinned live 2026-11-01, recorded above).
5. Whether the Mistral route honors temperature (observation for the
   acceptance note; the records declare what is sent either way).
6. The sim-book (type-A daily sim) D scan (revisit only at the promotion
   review; 2026-11-01 the scope was extended to the union of the MONTHLY
   and DAILY-CORE books only — see the scope-extension section).

## Sequencing

Wired only after the owner approves THIS note (the transport-fix slice is
committed, so v1's other gate is met). The live ledger starts clean under
the frozen A/B/C regime, and D accumulates as its own track beside it.
