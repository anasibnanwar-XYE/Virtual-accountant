# Virtual Accountant Copilot Architecture (Multi-Task + Retrieval + Constraints)

Last updated: 2026-02-22

## 1. Why this redesign

Single-shot classification is not enough for accounting workflows. The copilot must model:
- ledger state,
- policy constraints,
- workflow action sequences,
- personalization boundaries,
- uncertainty and escalation.

## 2. Core architecture

## 2.1 Shared multi-task model

Train one shared backbone with multiple heads:
- transaction/workflow head,
- CoA debit head,
- CoA credit head,
- GST/tax treatment head,
- risk/review head,
- reconciliation exception head,
- next-best-action head (UI/process prediction).

Keep existing specialist heads as fallback while shared backbone matures.

## 2.2 Retrieval + rerank

For CoA and document matching:
1. retrieve candidate set from master/history/policy memory,
2. rerank candidates with context model,
3. enforce policy/constraint checks.

This handles long-tail account labels and new account additions better than flat top-1 classification.

## 2.3 Constraint-aware decision layer

Final advisory output must pass hard checks:
- debit = credit,
- period lock checks,
- GST/tax compliance checks,
- invoice/payment allocation limits,
- blocked/discouraged policy filters.

Model scores; constrained decoder selects only feasible outputs.

## 2.4 Personalization adapters

Use global + scoped adapters:
- global base model,
- company adapter,
- user/role adapter.

Guardrails:
- minimum memory evidence,
- max top-1 shift rates,
- per-family drift limits,
- auto-revert to base if violated.

## 2.5 RAG layer

RAG should support explanation and evidence, not arithmetic:
- retrieve policy notes and prior approved analogs,
- return grounded rationale and alternatives,
- cite evidence IDs/source blocks.

## 3. Data contracts

Mandatory training events:
- `state_features`,
- `candidate_set`,
- `predicted_action`,
- `approved_action`,
- `constraint_result`,
- `override_reason`.

Required identifiers:
- `decision_id`, `sample_id`, `company_code`, `user_id`, `workflow_family`, `policy_version`, `model_version`.

## 4. Serving for concurrent users

Recommended deployment pattern:
- stateless inference workers,
- adapter selection by company/user at request time,
- GPU micro-batching for throughput,
- strict tenant isolation of memory/features,
- async queue for heavy recommendation tasks.

## 5. Confidence and review routing

Every advisory output includes:
- calibrated confidence,
- abstain/escalate option,
- reason codes,
- top-k alternatives,
- constraint check summary.

Low-confidence or high-risk classes route to review queue automatically.

## 6. Rollout order

1. shared feature contract and multi-task dataset builder,
2. retrieval+rereank for CoA/matching,
3. constrained decision layer,
4. action-policy (next-click) head,
5. RAG explanation integration.

## 7. Success metrics

Model quality:
- workflow-sliced top-1/top-3,
- candidate recall@k and rerank lift,
- calibration error and abstain quality.

Accounting safety:
- policy-invalid recommendation rate,
- blocked suggestion leak rate,
- period-lock violation rate.

Operational impact:
- first-pass acceptance,
- review queue precision,
- correction loop rate,
- close-cycle productivity improvement.
