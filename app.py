"""TrajLens - an interactive agentic-trajectory auditor.

Step through a recorded AI-agent tool-use trace and watch a deterministic
rubric flag the exact turn it hallucinated, mis-called a tool, or ignored the
instruction, then see where an LLM-judge would have been fooled.

Reconstructs, on public synthetic data, the kind of agent-evaluation work done
under NDA for frontier-model providers. No API key, no cost, never breaks.
"""
import json
import pathlib

import altair as alt
import pandas as pd
import streamlit as st

from verifiers import grade

st.set_page_config(page_title="TrajLens - Agentic Trajectory Auditor",
                   layout="wide")

DATA = pathlib.Path(__file__).parent / "data" / "trajectories.json"

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
def load():
    trajectories = json.loads(DATA.read_text(encoding="utf-8"))
    graded = []
    for t in trajectories:
        results, verdict = grade(t)
        graded.append({"traj": t, "results": results, "verdict": verdict})
    return graded


def verdict_pill(text):
    color = PASS if text == "PASSED" else FAIL if text == "FAILED" else MUTED
    return f"<span class='verdict' style='background:{color}'>{text}</span>"


data = load()

# ---- Sidebar -------------------------------------------------------------
st.sidebar.markdown("## TrajLens")
st.sidebar.caption("Agentic Trajectory Auditor")
idx = st.sidebar.radio(
    "Choose a trajectory",
    range(len(data)),
    format_func=lambda i: f"{data[i]['traj']['title']}  ({data[i]['traj']['expected_verdict']})",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Every verdict here is produced by **deterministic Python verifiers** over the "
    "trajectory JSON: reproducible, auditable, not vibes. Run `python grade.py` to "
    "regenerate them all."
)
st.sidebar.markdown(
    "<span class='muted'>Public synthetic data. Reconstructs the kind of agent-evaluation "
    "work I do under NDA. Nothing confidential is used.</span>",
    unsafe_allow_html=True,
)

tab_audit, tab_corpus, tab_about = st.tabs(
    ["Auditor", "Corpus dashboard", "About"]
)

# ---- Auditor tab ---------------------------------------------------------
with tab_audit:
    g = data[idx]
    t, results, verdict = g["traj"], g["results"], g["verdict"]
    human = t["expected_verdict"]
    judge = t.get("llm_judge_verdict", "-")

    st.subheader(t["title"])
    st.caption(t["category"])
    st.markdown(f"**Task:** {t['task']}")

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("<div class='smalllabel'>Rubric verdict (this tool)</div>", unsafe_allow_html=True)
        st.markdown(verdict_pill(verdict), unsafe_allow_html=True)
    with v2:
        st.markdown("<div class='smalllabel'>LLM-judge verdict</div>", unsafe_allow_html=True)
        st.markdown(verdict_pill(judge), unsafe_allow_html=True)
    with v3:
        st.markdown("<div class='smalllabel'>Human gold label</div>", unsafe_allow_html=True)
        st.markdown(verdict_pill(human), unsafe_allow_html=True)

    if judge != human:
        st.warning(
            f"The LLM-judge said **{judge}**, but the correct label is **{human}**. "
            "The automated judge was fooled here. The deterministic rubric got it right."
        )
    else:
        st.success("Rubric, LLM-judge, and human label all agree on this trajectory.")

    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Trajectory")
        offending = {}
        for r in results:
            if r.status == "fail" and r.step_index is not None:
                offending.setdefault(r.step_index, []).append(r.failure_mode)

        for i, step in enumerate(t["steps"]):
            bad = offending.get(i)
            title = f"Step {i} - " + ("FAIL: " + ", ".join(bad) if bad else "ok")
            with st.expander(title, expanded=bool(bad)):
                st.markdown(f"**Thought:** {step.get('thought', '')}")
                call = step.get("tool_call")
                if call:
                    st.markdown(f"**Tool call:** `{call['name']}`")
                    st.code(json.dumps(call.get("args", {}), indent=2), language="json")
                st.markdown(f"**Observation:** {step.get('observation', '')}")

        st.markdown("#### Final answer")
        st.info(t.get("final_answer", ""))

    with right:
        st.markdown("#### Rubric - 7 binary criteria")
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

    st.caption(t.get("notes", ""))

# ---- Corpus dashboard tab ------------------------------------------------
with tab_corpus:
    rows, mode_counts = [], {}
    rubric_agree = judge_agree = 0
    for g in data:
        t, verdict = g["traj"], g["verdict"]
        human = t["expected_verdict"]
        judge = t.get("llm_judge_verdict", "-")
        rubric_agree += int(verdict == human)
        judge_agree += int(judge == human)
        modes = sorted({r.failure_mode for r in g["results"] if r.status == "fail"})
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        rows.append({
            "Trajectory": t["title"],
            "Human": human,
            "Rubric": verdict,
            "LLM-judge": judge,
            "Failure modes": ", ".join(modes) or "-",
        })

    n = len(data)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trajectories", n)
    m2.metric("Failed", sum(1 for g in data if g["verdict"] == "FAILED"))
    m3.metric("Rubric vs human", f"{rubric_agree}/{n}")
    m4.metric("LLM-judge vs human", f"{judge_agree}/{n}")

    st.caption(
        "The deterministic rubric matches the human gold label on every trajectory; "
        "the LLM-judge is fooled on the cases where it does not cross-check tool outputs "
        "(a hallucinated price, a wasteful retry loop). Quantifying that gap is the point."
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
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Verdict comparison")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---- About tab -----------------------------------------------------------
with tab_about:
    st.markdown(
        """
### What is this?

**TrajLens** replays recorded AI-agent tool-use trajectories and grades every turn
against a binary rubric of **deterministic Python verifiers**. Failed steps light up
red; open one to see the exact assertion that failed and the named failure mode.

It reconstructs, on **public, synthetic data**, the kind of agentic-evaluation work
I do under NDA for frontier-model providers (Scale AI / Outlier): auditing tool-use
trajectories, authoring pass/fail rubrics, and building verifiers that catch failures
a results-only LLM-judge misses.

### Why deterministic verifiers?

Because every grade is **reproducible and auditable**: the same trajectory always gets
the same verdict, and you can read the exact line of code behind each one. Run
`python grade.py` to regenerate the whole thing.

### The 7 criteria

| ID | Criterion | Catches |
|----|-----------|---------|
| C1 | Valid tool selection | calling a tool that does not exist |
| C2 | Well-formed arguments | missing required arguments |
| C3 | Answer grounded in tool outputs | hallucinated facts / numbers |
| C4 | Instruction following | ignoring a stated constraint |
| C5 | No redundant tool calls | repeating identical calls |
| C6 | Clean termination | stopping without an answer |
| C7 | Efficient | blowing the step budget / looping |

### Rubric vs LLM-judge

Each trajectory also carries a precomputed **LLM-judge** verdict. On the flight-booking
and SQL cases the judge passes a trajectory that is actually broken: it does not
cross-check tool outputs. The rubric catches both. The corpus dashboard quantifies that
agreement gap (a Cohen's-kappa-style calibration signal).

### Extend it

Add a trajectory to `data/trajectories.json`, run `python grade.py`, and it appears here
automatically. The verifier library in `verifiers.py` is pure functions, easy to add C8+.
"""
    )
