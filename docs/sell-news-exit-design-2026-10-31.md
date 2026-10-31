# Type-D: the material-news sell consult — design note (2026-10-31)

Status: DESIGN, pre-wiring. Owner-requested (2026-10-31, the quota-day
session): "should the llm also consider current news? we do have
searxng and also a search mcp in openwebui mcpo container" — then the
correction: "not necessary via mcpo, we have to think about how to
make it accessible, currently it is only wired in the webui
container" — then the DECISION: "the cleanest will be to make the
searxng mcp reachable on the docker network and then both mcpo and
this app can reach it." The source is therefore the searxng MCP on
the shared docker network (section below). This note predeclares the
design; nothing is wired until the transport-fix slice commits and
the owner approves this note.

## Motivation

The type-C thesis-break exit waits on FILINGS — a quarterly cadence,
so the predeclared condition can only ever fire months after the world
learned the news (the research verdict: 7 events in 11 years, mean
+0.80pp/event, median −0.66pp — a fat-tail selection problem where the
large wins are the early catches). The wire moves first: a guidance
cut, a fraud accusation, a structural break is NEWS days or weeks
before it is a filing. An LLM consult is exactly the right instrument
for the judgment "does this headline invalidate the stated thesis?" —
it is the one step of the sell program that cannot be a rule.

## The event type

**D — the material-news scan.** Scope v1: the MONTHLY book only (the
thesis-discipline names, the same book the C step scans); the sim
book's D is out of scope until the promotion review. Cadence: nightly,
immediately after the type-C step, every trading night.

## The news source — the searxng MCP on the shared docker network

**DECIDED (the owner): the searxng MCP gets its own reachability on
the docker network, and both mcpo (the webui) and this app consume
it** — one shared news entry point, chosen for the MCP's better search
results (the owner's experience in the webui). The app therefore
speaks MCP for the D fetch; the honest costs are recorded here and
accepted: an MCP client surface in the app (the official `mcp` SDK's
streamable-http client, or a minimal hand-rolled JSON-RPC — the
wiring slice pins the choice) plus one protocol hop more than direct
SearXNG. The fail-safe discipline is unchanged: the fetch failing →
skip + log, NO consult, retried the next night — a middleman outage
never touches the book.

- **The deploy work (the owner's):** expose the mcp-searxng server's
  HTTP transport on the docker network both containers join (mcpo
  already speaks to it; the app's container joins the same network
  and gets a config knob with the MCP's base URL — default empty =
  the D step disabled, the same convention as every integration).
- **One path only — the MCP; no direct-SearXNG client is wired for D**
  (the owner's call: "no direct searxng"). The fetch still sits behind
  ONE internal interface (`fetch_ticker_news(ticker, window)`) so the
  D step never touches the MCP client directly, and every D ledger row
  records its `news_source` for provenance.
- **Discovery at wiring (step 0):** once reachable, enumerate the
  MCP's tools (the initialize handshake + `tools/list`), pin the
  search tool's exact name/params (query, categories, time range,
  result count) in the slice's acceptance note.

## The prompt — `sellshadow/1-d`

Frozen at wiring, versioned beside A/B/C in `sellreview.py`; no
post-hoc edits, any change is `-d/2` with a declared reason (the
frozen-prompt discipline). Context: the position facts (ticker, entry
date/price, current close, PnL, months held, the stated thesis
condition) plus a BOUNDED headline block (top N per name, each with
source and date, fetched for a declared recent window). The
materiality instruction is predeclared: **thesis-invalidating news
only** — fraud, guidance cuts, structural breaks. Price moves are NOT
D events (a falling stock is type A's domain, momentum decay is
B's) — the no-momentum-proxy rule keeps D from becoming a second
stop-loss. Decision vocabulary: `EXIT | HOLD`, no params (C's
contract).

## Shadow-first — and why it is the only honest option

News is **un-backtestable**: no point-in-time reconstruction of a
news wire exists, so no offline reference bar can ever be measured
(the A/B/C bars were pre-measured offline; D's cannot be, declared up
front). Therefore D starts as a SHADOW consult: every D row is
recorded in the ledger (event_type "D", engine "monthly",
engine_branch = the counterfactual "hold to next month-end rebuild")
with **NO execution authority — the fill columns stay NULL forever
in the shadow phase**; no sell is ever placed by a D verdict.

**The promotion rule (predeclared now, not at promotion time):** D
earns execution authority only after **≥ 6 months of completed
windows** AND a measured positive edge vs the month-end wait on the
same completed-window math as C's verdict (the D-EXIT counterfactual
is the wait itself). Promotion is a versioned, declared change with
its own audit entry — never a silent flag flip. A D record that never
promotes is not a failure: the shadow IS the measurement.

## Fail-safes, dedupe, and the quota bound

- Knob off → the step is skipped (every layer knob's convention).
- The news FETCH failing → skip + log, **no consult** (the news IS the
  event; an empty-news D consult is meaningless), retried the next
  night. No ledger row for a skipped name.
- The LLM transport failing → the standard fail-safe row
  (fail_safe=True, the error recorded) — inert in shadow anyway.
- Dedupe: `event_id = D:{ticker}:{session}` — one consult per name per
  night. The **headline-changed guard**: a name whose fetched headline
  set is byte-identical to the previous night's is SKIPPED (the story
  has not developed) with the skip recorded in the step summary. This
  is both the dedupe against a slow-moving story and the quota bound:
  consults happen only on changed news sets, typically a few per
  night across a ~10-name book, not a fixed 10 consults/night.
- The scorecard grows a D section on the same completed-window rule;
  HOLD rows are correctness records, not edge.

## Open items at wiring

1. The MCP's HTTP transport + endpoint on the shared docker network
   (the owner's deploy work) and, once reachable, the step-0
   discovery: the initialize handshake, `tools/list`, the search
   tool's exact name/params — pinned in the wiring slice's acceptance
   note, together with the client choice (the official `mcp` SDK's
   streamable-http client vs a minimal hand-rolled JSON-RPC).
2. The headline window (proposed: the last 5 trading days) and N
   (proposed: 8) — declared in the prompt version, not tuned later.
3. The sim-book D scan (revisit only at the promotion review).

## Sequencing

Wired only AFTER the transport-fix slice commits — the live ledger
starts clean under the frozen A/B/C regime, and D accumulates as its
own track beside it.
