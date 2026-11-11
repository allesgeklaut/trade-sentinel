"""Responsive sell history keeps details readable without changing scoring."""
import json
import subprocess
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "static" / "index.html"


def _render(decision):
    source = INDEX.read_text()
    start = source.index("        function renderSellScore(data) {")
    end = source.index("        async function loadSellScore()", start)
    script = r"""
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const nodes = Object.fromEntries(['sellscore-asof', 'sellscore-cards', 'sellscore-bars',
    'sellscore-history'].map(id => [id, {}]));
const document = {getElementById: id => nodes[id]};
const escapeHtml = value => String(value).replace(/[&<>"']/g, c =>
    ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const fmtPct = value => String(value ?? '—');
const fmtPp = fmtPct;
eval(input.source);
renderSellScore({decisions: [input.decision]});
console.log(JSON.stringify(nodes['sellscore-history'].innerHTML));
"""
    result = subprocess.run(
        ["node", "-e", script], input=json.dumps({"source": source[start:end],
                                                 "decision": decision}),
        capture_output=True, text=True, check=True, timeout=15)
    return json.loads(result.stdout)


def test_error_is_collapsible_escaped_and_cells_have_mobile_labels():
    html = _render({"event_type": "D", "ticker": "SYNTHETIC", "decision": "HOLD",
                    "status": "pending", "fail_safe": True,
                    "fail_reason": '<script>alert("synthetic")</script>' + "x" * 200,
                    "execution": {"status": "not_applicable", "books": {}}})
    assert '<details><summary>YES — error details</summary>' in html
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert html.count('data-label=') == 11
    assert 'data-label="execution" class="event-detail"' in html
    assert 'data-label="score status">pending' in html
    assert 'no exit requested' in html


def test_per_book_fills_remain_separate_from_pending_score():
    html = _render({"event_type": "D", "ticker": "SYNTHETIC", "decision": "EXIT",
                    "status": "pending", "execution": {"status": "filled", "books": {
                        book: {"status": "filled", "session_date": "2026-02-03",
                               "price": 42, "shares": shares}
                        for book, shares in (("monthly", 2), ("daily_core", 3))}}})
    assert 'monthly: filled 2026-02-03 @ 42 USD (2 shares)' in html
    assert 'daily-core: filled 2026-02-03 @ 42 USD (3 shares)' in html
    assert 'data-label="delta">pending' in html
    assert 'data-label="score status">pending' in html
