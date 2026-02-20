# Virtual Accountant LLM + ML Blueprint (ERP v2)

This is the execution blueprint for a safe virtual-accountant architecture:

- deterministic accounting core (rules + ERP facts)
- probabilistic ML specialists (classification/risk/forecast/anomaly)
- LLM orchestration and explanation layer (no raw math)

## Core Rule

LLM must never invent numbers or post entries directly.

Flow:

1. user query
2. tool calls (DB + rules + ML)
3. contract validation
4. audit log append
5. LLM response from validated outputs only

## Query to Tool Map

1. `Categorize this new transaction`
- tools: `categorize_transaction`, `detect_anomalies`
- fallback: if confidence below threshold, route to review queue.

2. `Suggest product account mapping for this SKU`
- tools: product account recommender + policy/rule check.
- output must include top-k + confidence + similar-history evidence.

3. `Why is GST payable high this month?`
- tools: GST summary query, rule checks, anomaly scan.
- response must cite evidence refs (period + report id).

4. `What should I review before close?`
- tools: review-risk, reconciliation-exception, workflow routing.
- output is ranked action list (`P1/P2/P3`).

5. `Forecast next 30 days cashflow`
- tools: forecast model + deterministic sanity rules.
- include confidence and warnings for OOD / sparse history.

6. `Show suspicious transactions`
- tools: anomaly detector + dedupe checks + control-account checks.
- include reason codes and severity.

7. `Prepare monthly summary`
- tools: deterministic KPI aggregation + optional ML insights.
- KPI math stays deterministic.

## Tool Output Contract

Every tool response should include:

- `task`
- `prediction` or `results`
- `confidence` / score fields
- `reason_codes`
- `warnings`
- enough identifiers for traceability (`reference`, `period`, etc.)

For this lab, contract enforcement script:

- `scripts/37_virtual_accountant_tool_contracts.py`

Supported tool contracts:

- `categorize_transaction`
- `detect_anomalies`
- `forecast_cashflow`
- `generate_monthly_summary`

## Audit Trail Contract

Every tool call should append one JSONL record:

- timestamp (UTC)
- tool id
- actor/user id
- request SHA256
- response SHA256
- validation pass/fail + errors

Recommended path:

- `outputs/virtual_accountant_audit/tool_calls.jsonl`

## Integration Order

1. enforce tool contracts in current scripts
2. connect LLM orchestration to tool invocations only
3. block non-validated tool payloads from final chat response
4. add threshold policies per workflow family
5. add role-based action gates for posting/approval
