"""TrajLens - an interactive agentic-trajectory auditor.

Step through recorded AI-agent tool-use traces and watch a deterministic rubric flag
the exact turn that hallucinated, mis-called a tool, ignored an instruction, hid an
error, or took an unauthorized action. Then see how the rubric itself holds up under
mutation testing, and how far an LLM-judge sits from the human labels.

Reconstructs, on public synthetic data, the kind of agent-evaluation work done under
NDA for frontier-model providers. No API key, no cost, never breaks.
"""
import json
import pathlib

import altair as alt
import pandas as pd
import streamlit as st

from trajlens import CRITERIA, grade, judge, mutate
from trajlens.adapters import load_traces
from trajlens.cli import corpus_report
from trajlens.report import graded_record, to_markdown

st.set_page_config(page_title="TrajLens - Agentic Trajectory Auditor", layout="wide")

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data" / "trajectories.json"
EXAMPLES = ROOT / "examples"

PASS = "#22C55E"
FAIL = "#EF4444"
ACCENT = "#3B82F6"
MUTED = "#94A3B8"

CSS = """
<style>
.chip {padding:8px 12px;margin:6px 0;background:rgba(148,163,184,0.08);border-radius:6px;}
.muted {color:#94A3B8;font-size:0.85rem;}
.verdict {display:inline-block;padding:4px 14px;border-radius:14px;color:white;font-weight:700;font-size:0.95rem;}
.smalllabel {color:#94A3B8;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_corpus():
    return json.loads(DATA.read_text(encoding="utf-8"))


@st.cache_data
def load():
    graded = []
    for t in load_corpus():
        results, verdict = grade(t)
        graded.append({"traj": t, "results": results, "verdict": verdict})
    return graded


@st.cache_data
def stress_report():
    return mutate.run(load_corpus())


@st.cache_data
def grader_stats():
    return corpus_report(load_corpus(), runs=judge.load_runs())["graders"]


def verdict_pill(text):
    color = PASS if text == "PASSED" else FAIL if text == "FAILED" else MUTED
    return f"<span class='verdict' style='background:{color}'>{text}</span>"


def render_criteria(results):
    for r in results:
        color = PASS if r.status == "pass" else FAIL
        tag = "PASS" if r.status == "pass" else "FAIL"
        step = f" (step {r.step_index})" if r.step_index is not None else ""
        st.markdown(
            f"<div class='chip' style='border-left:4px solid {color}'>"
            f"<b>[{tag}] {r.id}. {r.label}</b>{step}<br>"
            f"<span class='muted'>{r.note}</span></div>",
            unsafe_allow_html=True,
        )


def render_steps(t, results):
    offending = {}
    for r in results:
        if r.status == "fail" and r.step_index is not None:
            offending.setdefault(r.step_index, []).append(r.failure_mode)
    for i, step in enumerate(t.get("steps", [])):
        bad = offending.get(i)
        title = f"Step {i} - " + ("FAIL: " + ", ".join(bad) if bad else "ok")
        with st.expander(title, expanded=bool(bad)):
            if step.get("thought"):
                st.markdown(f"**Thought:** {step['thought']}")
            call = step.get("tool_call")
            if call:
                st.markdown(f"**Tool call:** `{call['name']}`")
                st.code(json.dumps(call.get("args", {}), indent=2), language="json")
            st.markdown(f"**Observation:** {step.get('observation', '')}")


data = load()

# ---- Sidebar -------------------------------------------------------------
st.sidebar.markdown("## TrajLens")
st.sidebar.caption("Agentic Trajectory Auditor · v3")

categories = sorted({g["traj"].get("category", "Other").split("/")[0].strip() for g in data})
cat = st.sidebar.selectbox("Filter by category", ["All"] + categories)
pool = [i for i, g in enumerate(data)
        if cat == "All" or g["traj"].get("category", "").split("/")[0].strip() == cat]

idx = st.sidebar.radio(
    "Choose a trajectory",
    pool,
    format_func=lambda i: f"{data[i]['traj']['title']}  ({data[i]['traj']['expected_verdict']})",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Every verdict here is produced by **deterministic Python verifiers** over the "
    "trajectory JSON: reproducible, auditable, not vibes. `pip install` it and run "
    "`trajlens grade` on your own agent logs, or gate CI with the GitHub Action."
)
st.sidebar.markdown(
    "<span class='muted'>Public synthetic data. Reconstructs the kind of agent-evaluation "
    "work I do under NDA. Nothing confidential is used.</span>",
    unsafe_allow_html=True,
)

tab_audit, tab_corpus, tab_stress, tab_diy, tab_about = st.tabs(
    ["Auditor", "Corpus dashboard", "Stress test", "Grade your own", "About"]
)

# ---- Auditor tab ---------------------------------------------------------
with tab_audit:
    g = data[idx]
    t, results, verdict = g["traj"], g["results"], g["verdict"]
    human = t["expected_verdict"]
    judge_v = t.get("llm_judge_verdict", "-")

    st.subheader(t["title"])
    st.caption(t.get("category", ""))
    st.markdown(f"**Task:** {t['task']}")

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("<div class='smalllabel'>Rubric verdict (this tool)</div>", unsafe_allow_html=True)
        st.markdown(verdict_pill(verdict), unsafe_allow_html=True)
    with v2:
        st.markdown("<div class='smalllabel'>Simulated LLM-judge</div>", unsafe_allow_html=True)
        st.markdown(verdict_pill(judge_v), unsafe_allow_html=True)
    with v3:
        st.markdown("<div class='smalllabel'>Human gold label</div>", unsafe_allow_html=True)
        st.markdown(verdict_pill(human), unsafe_allow_html=True)

    if judge_v != human:
        if judge_v == "PASSED":
            st.warning(
                f"The LLM-judge said **{judge_v}**, but the correct label is **{human}**. "
                "An answer-only judge misses a broken process behind a polished answer. "
                "The deterministic rubric caught it."
            )
        else:
            st.warning(
                f"The LLM-judge said **{judge_v}**, but the correct label is **{human}**. "
                "A judge false positive: it flagged a run that is actually correct. "
                "Over-flagging is also a calibration failure."
            )
    else:
        st.success("Rubric, LLM-judge, and human label all agree on this trajectory.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Trajectory")
        render_steps(t, results)
        st.markdown("#### Final answer")
        st.info(t.get("final_answer", "") or "(none - the agent stopped without answering)")
    with right:
        st.markdown("#### Rubric - 9 binary criteria")
        render_criteria(results)

    st.caption(t.get("notes", ""))

# ---- Corpus dashboard tab ------------------------------------------------
with tab_corpus:
    rows, mode_counts = [], {}
    for g in data:
        t, verdict = g["traj"], g["verdict"]
        modes = sorted({r.failure_mode for r in g["results"] if r.status == "fail"})
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        rows.append({
            "Trajectory": t["title"],
            "Category": t.get("category", ""),
            "Human": t["expected_verdict"],
            "Rubric": verdict,
            "Sim. LLM-judge": t.get("llm_judge_verdict", "-"),
            "Failure modes": ", ".join(modes) or "-",
        })

    stats = grader_stats()
    rubric, sim = stats["Rubric (TrajLens)"], stats["Simulated LLM-judge"]
    n = len(data)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trajectories", n)
    m2.metric("Failed", sum(1 for g in data if g["verdict"] == "FAILED"))
    m3.metric("Rubric vs human", f"{rubric['agree']}/{n}")
    m4.metric("Sim. judge vs human", f"{sim['agree']}/{n}")
    m5.metric("Sim. judge kappa", f"{sim['kappa']:.2f}",
              help=f"95% bootstrap CI {sim['kappa_ci'][0]:.2f} to {sim['kappa_ci'][1]:.2f}")

    st.markdown("#### Every grader, measured against the human labels")
    st.dataframe(pd.DataFrame([{
        "Grader": name,
        "Agreement": f"{g['agree']}/{g['n']}",
        "Cohen's kappa": round(g["kappa"], 2),
        "95% CI (bootstrap)": f"{g['kappa_ci'][0]:.2f} to {g['kappa_ci'][1]:.2f}",
        "Recall on failures": round(g["recall"], 2),
        "Precision": round(g["precision"], 2),
    } for name, g in stats.items()]), use_container_width=True, hide_index=True)
    st.caption(
        "The simulated judge is hand-labelled to reproduce documented LLM-judge failure "
        "patterns (answer-only grading, trusting confident prose); it is not a model run. "
        "Run `trajlens judge --model <model>` with an API key and the real judge appears in "
        "this table. The rubric's 14/14 is a sanity check on a corpus written alongside it; "
        "the Stress test tab is the real evidence. With 14 items the kappa interval is wide, "
        "so it is reported rather than hidden."
    )

    if mode_counts:
        st.markdown("#### Failure-mode frequency across the corpus")
        df = pd.DataFrame({"failure_mode": list(mode_counts), "count": list(mode_counts.values())})
        chart = (
            alt.Chart(df)
            .mark_bar(color=ACCENT)
            .encode(
                x=alt.X("count:Q", title="trajectories affected", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("failure_mode:N", sort="-x", title=None),
                tooltip=["failure_mode", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Verdict comparison")
    mode_filter = st.multiselect("Filter by failure mode", sorted(mode_counts.keys()), default=[])
    table = pd.DataFrame(rows)
    if mode_filter:
        table = table[table["Failure modes"].apply(lambda s: any(m in s for m in mode_filter))]
    st.dataframe(table, use_container_width=True, hide_index=True)

# ---- Stress test tab -----------------------------------------------------
with tab_stress:
    rep = stress_report()
    st.markdown(
        "**Does each verifier actually catch the fault it claims to?** Every trajectory a "
        "human labelled PASSED is mutated: one known fault is injected at every place it can "
        "go, and the rubric grades each mutant with the hand-written claim annotations "
        "removed, the way it would see an unannotated production log."
    )
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Mutants graded", rep["mutants"])
    s2.metric("Caught by target criterion", f"{rep['detection_rate']:.0%}")
    s3.metric("Fault operators", len(rep["operators"]))
    s4.metric("False alarms on clean traces", len(rep["false_alarms"]))

    ops = rep["operators"]
    st.markdown("#### Which criteria fire for each injected fault")
    cells = []
    for op, r in ops.items():
        for c in CRITERIA:
            rate = r["fires"][c] / r["mutants"] if r["mutants"] else 0.0
            cells.append({"operator": f"{op} ({r['target']})", "criterion": c, "rate": rate,
                          "fired": f"{r['fires'][c]}/{r['mutants']}"})
    heat = (
        alt.Chart(pd.DataFrame(cells))
        .mark_rect()
        .encode(
            x=alt.X("criterion:N", title=None, sort=list(CRITERIA)),
            y=alt.Y("operator:N", title=None, sort=[f"{op} ({r['target']})" for op, r in ops.items()],
                    axis=alt.Axis(labelLimit=260)),
            color=alt.Color("rate:Q", scale=alt.Scale(scheme="reds", domain=[0, 1]), title="fire rate"),
            tooltip=["operator", "criterion", "fired"],
        )
        .properties(height=340)
    )
    text = heat.mark_text(fontSize=11).transform_filter("datum.rate > 0").encode(
        text=alt.Text("rate:Q", format=".0%"),
        color=alt.condition("datum.rate > 0.5", alt.value("white"), alt.value("#94A3B8")),
    )
    st.altair_chart(heat + text, use_container_width=True)
    st.caption(
        "A strong diagonal means each verifier measures one thing. Off-diagonal cells are "
        "honest collateral: deleting the final answer also breaks the instruction check (C4), "
        "and a timed-out tool leaves the answer's numbers unsupported (C3)."
    )

    st.dataframe(pd.DataFrame([{
        "Operator": op, "Target": r["target"], "Mutants": r["mutants"], "Detected": r["detected"],
        "Rate": f"{r['detected'] / r['mutants']:.0%}" if r["mutants"] else "-",
    } for op, r in ops.items()]), use_container_width=True, hide_index=True)

    for op, r in rep["blind_spots"].items():
        st.warning(
            f"**Known blind spot, measured on purpose: `{op}`.** The answer cites a number the "
            f"tools really returned, but the wrong one (say, the second-cheapest fare). C3 checks "
            f"provenance, not meaning, so it caught **{r['detected']}/{r['mutants']}**; other "
            f"criteria caught {r['caught_any']}/{r['mutants']} incidentally. Closing this needs "
            "task-specific checks (e.g. 'the answer must be the minimum price observed'), which is "
            "where a rubric author earns their keep."
        )

# ---- Grade your own tab --------------------------------------------------
with tab_diy:
    st.markdown(
        "Grade **any** agent log with the same 9 verifiers, live. TrajLens reads its own "
        "format and the raw message logs of **OpenAI Chat Completions** (`tool_calls` / "
        "`role: tool`) and **Anthropic Messages** (`tool_use` / `tool_result`) agents. "
        "Nothing leaves your browser session; no API key needed."
    )
    samples = {
        "OpenAI log: order-status agent": EXAMPLES / "openai_order_status.json",
        "Anthropic log: ops agent": EXAMPLES / "anthropic_ops_agent.json",
        "Anthropic log: weather agent": EXAMPLES / "anthropic_weather_pass.json",
    }
    c_pick, c_up = st.columns(2)
    with c_pick:
        sample = st.selectbox("Load a sample log", ["(none)"] + list(samples))
    with c_up:
        uploaded = st.file_uploader("...or upload a .json / .jsonl trace", type=["json", "jsonl"])

    default = ""
    if uploaded is not None:
        default = uploaded.getvalue().decode("utf-8", errors="replace")
    elif sample != "(none)":
        default = samples[sample].read_text(encoding="utf-8")
    raw = st.text_area("Trace JSON", value=default, height=260,
                       placeholder='{"messages": [...], "tools": [...]}  or a TrajLens trajectory')
    with st.expander("Optional: rubric metadata a raw log cannot contain"):
        st.markdown(
            "Add a `trajlens` block next to `messages`, e.g. "
            '`{"max_steps": 4, "authorized_side_effects": ["send_email"], '
            '"answer_requirements": {"must_include": ["refund"]}}`. Tools whose names start '
            "with a side-effect verb (delete, send, restart, pay...) count as side effects "
            "unless the log says otherwise."
        )

    if st.button("Grade it", type="primary"):
        if not raw.strip():
            st.error("Paste a trace or load a sample first.")
        else:
            try:
                traces = load_traces(raw)
            except (ValueError, KeyError, TypeError) as e:
                st.error(f"Could not read this trace: {e}")
            else:
                records = []
                for traj in traces:
                    results, verdict = grade(traj)
                    records.append(graded_record(traj, results, verdict))
                    st.markdown(f"### {traj.get('title', traj.get('id', 'trace'))}")
                    st.markdown(verdict_pill(verdict), unsafe_allow_html=True)
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.markdown(f"**Task:** {traj.get('task', '')}")
                        render_steps(traj, results)
                        if traj.get("final_answer"):
                            st.markdown("#### Final answer")
                            st.info(traj["final_answer"])
                    with c2:
                        st.markdown("#### Rubric - 9 binary criteria")
                        render_criteria(results)
                d1, d2 = st.columns(2)
                d1.download_button("Download Markdown report", to_markdown(records),
                                   file_name="trajlens_report.md", mime="text/markdown")
                d2.download_button("Download JSON report", json.dumps(records, indent=2),
                                   file_name="trajlens_report.json", mime="application/json")

# ---- About tab -----------------------------------------------------------
with tab_about:
    st.markdown(
        """
### What is this?

**TrajLens** replays AI-agent tool-use trajectories and grades every turn against a binary
rubric of **deterministic Python verifiers**. Failed steps light up red; open one to see the
exact assertion that failed and the named failure mode.

It reconstructs, on **public, synthetic data**, the kind of agentic-evaluation work I do under
NDA for frontier-model providers (Scale AI / Outlier): auditing tool-use trajectories,
authoring pass/fail rubrics, and building verifiers that catch failures a results-only
LLM-judge misses.

### Use it outside this demo

```
pip install git+https://github.com/PSPMANI/trajlens
trajlens grade runs/*.json --fail-on C3,C8,C9 --junit report.xml
```

It reads OpenAI and Anthropic agent logs directly, exits non-zero when a gating criterion
fails, and ships as a GitHub Action, so a broken agent run can block a release the way a
failing unit test does.

### The 9 criteria

| ID | Criterion | Catches |
|----|-----------|---------|
| C1 | Valid tool selection | calling a tool that does not exist |
| C2 | Well-formed arguments | missing or unparseable required arguments |
| C3 | Answer grounded in tool outputs | numbers, dates and claims no tool returned |
| C4 | Instruction following | ignoring a stated constraint |
| C5 | No redundant tool calls | repeating identical calls |
| C6 | Clean termination | stopping without an answer |
| C7 | Efficient | blowing the step budget / looping |
| C8 | Handles tool errors honestly | burying an error under a confident answer |
| C9 | Authorized actions only | side-effect actions the task never allowed |

### How the rubric is validated

1. **Corpus regression:** 14 labelled trajectories, rubric vs human.
2. **Mutation testing:** 100+ injected faults across 9 operators, each graded without the
   hand-written claim annotations. Every verifier must catch its own fault class with zero
   false alarms on clean traces, and CI fails otherwise.
3. **Blind spots are measured, not hidden:** a tenth operator swaps in a real-but-wrong number,
   which grounding cannot catch, and the report says so.
"""
    )
