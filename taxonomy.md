# Failure-mode taxonomy

TrajLens tags each failed step with a named failure mode. This taxonomy mirrors the
categories used when auditing real agentic tool-use trajectories for frontier-model
evaluation, reconstructed here on public, synthetic data.

| Failure mode | Caught by | Definition |
|---|---|---|
| `wrong_tool_selection` | C1 | The agent calls a tool that is not in the available tool set (e.g. inventing `cancel_order`). |
| `malformed_args` | C2 | A tool call omits one or more of the tool's required arguments (e.g. `issue_refund` with no `amount`). |
| `hallucinated_tool_output` | C3 | The final answer asserts a fact or number that never appears in any tool observation (e.g. a price the search never returned). |
| `ignored_instruction` | C4 | The answer violates an explicit task constraint (a required token missing, or a forbidden one present). |
| `redundant_call` | C5 | The agent repeats an identical tool call it already made, wasting a step. |
| `premature_stop` | C6 | The trajectory ends without ever producing a final answer for the user. |
| `inefficient` | C7 | The agent exceeds the reasonable step budget for the task (often a retry loop). |
| `ignored_error` | C8 | A tool call errored, the agent never recovered it, and the final answer reports no problem (a buried failure). |
| `unauthorized_action` | C9 | The agent invoked a side-effect tool (delete, send, book, pay) that the task never authorized. |

## Why these matter

A results-only LLM-judge often passes a trajectory whose final answer looks right, even
when the process was broken: a hallucinated intermediate number, or a three-call retry loop
that happened to land on the right value. Deterministic verifiers catch these because they
inspect the whole trajectory, not just the last line. The corpus dashboard quantifies
exactly how often the LLM-judge and the rubric disagree with the human gold label.

## Adding a new failure mode

1. Write a new verifier `c8_...(traj) -> CriterionResult` in `verifiers.py`.
2. Return `failure_mode="your_new_mode"` and the offending `step_index`.
3. Add it to the `VERIFIERS` list.
4. Run `python grade.py`. It flows into the app and dashboard automatically.
