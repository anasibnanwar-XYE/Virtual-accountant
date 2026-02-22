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

Practical low-footprint baseline:
- shared trunk: `input -> 128 -> 64`
- thin heads on top of the `64` representation.

Rationale:
- avoids relearning accounting context separately in each head,
- improves cross-head consistency (`tx`, `coa`, `risk`, `recon`),
- keeps parameter footprint small enough for fast iteration.

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

Recommended rollout for personalization:
1. Phase P1 (1-5 companies): per-company logit bias vectors only.
2. Phase P2: add low-rank company adapters (`64 -> r -> 64`, small `r` like `8`).
3. Phase P3: include user/role adapters where enough memory exists.

This gives fast personalization now without locking architecture.

## 2.5 COA head evolution (classification -> embedding retrieval)

Current fixed-size debit/credit heads are acceptable for bootstrapping but should evolve to:
- context embedding `h`,
- company-scoped account embedding table `E_company[account_id]`,
- score via dot product (retrieve/rank), then rerank top-N with richer account features.

Benefits:
- handles company-specific COA growth without hardcoded class dimensions,
- naturally supports long-tail accounts,
- cleaner online addition of new accounts.

## 2.6 RAG layer

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

1. shared feature contract + shared trunk with thin heads,
2. company logit-bias personalization,
3. retrieval+rereank for CoA/matching,
4. company low-rank adapters,
5. constrained decision layer,
6. action-policy (next-click) head,
7. COA embedding retrieval head migration,
8. RAG explanation integration.

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
