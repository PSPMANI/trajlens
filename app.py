"""TrajLens - an interactive agentic-trajectory auditor.

Step through recorded AI-agent tool-use traces and watch a deterministic
rubric flag the exact turn that hallucinated, mis-called a tool, ignored an
instruction, hid an error, or took an unauthorized action - then see where an
LLM-judge would have been fooled, quantified with Cohen's kappa.

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


def cohens_kappa(a, b):
    """Cohen's kappa for two binary PASSED/FAILED label lists, pure Python."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(1 for x in a if x == "PASSED") / n
    pb = sum(1 for y in b if y == "PASSED") / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


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
            st.markdown(f"**Thought:** {step.get('thought', '')}")
            call = step.get("tool_call")
            if call:
                st.markdown(f"**Tool call:** `{call['name']}`")
                st.code(json.dumps(call.get("args", {}), indent=2), language="json")
            st.markdown(f"**Observation:** {step.get('observation', '')}")


data = load()

# ---- Sidebar -------------------------------------------------------------
st.sidebar.markdown("## TrajLens")
st.sidebar.caption("Agentic Trajectory Auditor")

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
    "trajectory JSON: reproducible, auditable, not vibes. Run `python grade.py` to "
    "regenerate them all."
)
st.sidebar.markdown(
    "<span class='muted'>Public synthetic data. Reconstructs the kind of agent-evaluation "
    "work I do under NDA. Nothing confidential is used.</span>",
    unsafe_allow_html=True,
)

tab_audit, tab_corpus, tab_diy, tab_about = st.tabs(
    ["Auditor", "Corpus dashboard", "Grade your own", "About"]
)

# ---- Auditor tab ---------------------------------------------------------
with tab_audit:
    g = data[idx]
    t, results, verdict = g["traj"], g["results"], g["verdict"]
    human = t["expected_verdict"]
    judge = t.get("llm_judge_verdict", "-")

    st.subheader(t["title"])
    st.caption(t.get("category", ""))
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
        if judge == "PASSED":
            st.warning(
                f"The LLM-judge said **{judge}**, but the correct label is **{human}**. "
                "The judge graded only the final answer and missed a broken process. "
                "The deterministic rubric caught it."
            )
        else:
            st.warning(
                f"The LLM-judge said **{judge}**, but the correct label is **{human}**. "
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
    humans, judges = [], []
    rubric_agree = judge_agree = 0
    for g in data:
        t, verdict = g["traj"], g["verdict"]
        human = t["expected_verdict"]
        judge = t.get("llm_judge_verdict", "-")
        humans.append(human)
        judges.append(judge)
        rubric_agree += int(verdict == human)
        judge_agree += int(judge == human)
        modes = sorted({r.failure_mode for r in g["results"] if r.status == "fail"})
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        rows.append({
            "Trajectory": t["title"],
            "Category": t.get("category", ""),
            "Human": human,
            "Rubric": verdict,
            "LLM-judge": judge,
            "Failure modes": ", ".join(modes) or "-",
        })

    n = len(data)
    kappa = cohens_kappa(humans, judges)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trajectories", n)
    m2.metric("Failed", sum(1 for g in data if g["verdict"] == "FAILED"))
    m3.metric("Rubric vs human", f"{rubric_agree}/{n}")
    m4.metric("LLM-judge vs human", f"{judge_agree}/{n}")
    m5.metric("Judge Cohen's kappa", f"{kappa:.2f}")

    st.caption(
        "The deterministic rubric matches the human gold label on every trajectory. "
        "The LLM-judge misses process failures behind correct-looking answers AND "
        "false-positives on at least one good run - Cohen's kappa quantifies how far "
        "it sits from the human labels. Measuring the judge is the point."
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
    mode_filter = st.multiselect(
        "Filter by failure mode", sorted(mode_counts.keys()), default=[])
    table = pd.DataFrame(rows)
    if mode_filter:
        table = table[table["Failure modes"].apply(
            lambda s: any(m in s for m in mode_filter))]
    st.dataframe(table, use_container_width=True, hide_index=True)

# ---- Grade your own tab --------------------------------------------------
with tab_diy:
    st.markdown(
        "Paste any trajectory JSON below and the 9 deterministic verifiers grade it "
        "instantly - in your browser session, no API, no key, nothing uploaded anywhere."
    )
    example = json.dumps(data[0]["traj"], indent=2)
    with st.expander("See the trajectory JSON format (working example)"):
        st.code(example, language="json")
        st.download_button("Download example JSON", example,
                           file_name="example_trajectory.json", mime="application/json")

    raw = st.text_area("Trajectory JSON", height=280,
                       placeholder='{"id": "my_agent_run", "task": "...", "available_tools": [...], "steps": [...], ...}')
    if st.button("Grade it", type="primary"):
        if not raw.strip():
            st.error("Paste a trajectory JSON first (open the example above for the format).")
        else:
            try:
                traj = json.loads(raw)
            except json.JSONDecodeError as e:
                st.error(f"Not valid JSON: {e}")
            else:
                try:
                    results, verdict = grade(traj)
                except Exception as e:
                    st.error(f"Could not grade this object: {e}. Check the example format.")
                else:
                    st.markdown("<div class='smalllabel'>Rubric verdict</div>", unsafe_allow_html=True)
                    st.markdown(verdict_pill(verdict), unsafe_allow_html=True)
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.markdown("#### Trajectory")
                        render_steps(traj, results)
                        if traj.get("final_answer"):
                            st.markdown("#### Final answer")
                            st.info(traj["final_answer"])
                    with c2:
                        st.markdown("#### Rubric - 9 binary criteria")
                        render_criteria(results)

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

### The 9 criteria

| ID | Criterion | Catches |
|----|-----------|---------|
| C1 | Valid tool selection | calling a tool that does not exist |
| C2 | Well-formed arguments | missing required arguments |
| C3 | Answer grounded in tool outputs | hallucinated facts / numbers |
| C4 | Instruction following | ignoring a stated constraint |
| C5 | No redundant tool calls | repeating identical calls |
| C6 | Clean termination | stopping without an answer |
| C7 | Efficient | blowing the step budget / looping |
| C8 | Handles tool errors honestly | burying an error under a confident answer |
| C9 | Authorized actions only | side-effect actions the task never allowed |

C8 and C9 are safety-grade checks: an agent that hides a failed test run, or deletes
files when it was only asked to list them, is dangerous even when its final answer
reads well.

### Rubric vs LLM-judge, quantified

Each trajectory carries a precomputed **LLM-judge** verdict next to the **human gold
label**. The judge fails in both directions: it passes broken runs whose final answer
looks right, and it flags at least one good run as broken (a false positive). The
corpus dashboard reports raw agreement and **Cohen's kappa**, the standard
inter-rater statistic - because before you trust an automated judge, you measure it.

### Grade your own

The "Grade your own" tab accepts any trajectory JSON in the documented format and
grades it with the same verifiers, live. Deterministic evaluation needs no API key.

### Extend it

Add a trajectory to `data/trajectories.json`, run `python grade.py`, and it appears
here automatically. Add a criterion by writing one pure function in `verifiers.py`
and appending it to `VERIFIERS`.
"""
    )
