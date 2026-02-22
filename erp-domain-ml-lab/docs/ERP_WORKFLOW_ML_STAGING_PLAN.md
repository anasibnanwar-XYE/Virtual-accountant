# ERP Workflow ML Staging Plan (Copilot Redesign)

Last updated: 2026-02-22

This roadmap redesign moves from isolated label models to a ledger-aware accounting copilot, while preserving strict advisory safety and champion/challenger governance.

Related docs:
- `docs/VIRTUAL_ACCOUNTANT_BLUEPRINT.md`
- `docs/VIRTUAL_ACCOUNTANT_COPILOT_ARCHITECTURE.md`
- `docs/ERP_V2_FLOW_AND_CONCURRENCY_PLAN.md`
- `docs/INDIA_GST_BOOKKEEPING_ML_PLAYBOOK.md`

## 1. Goal and Non-Goal

Goal:
- advisory-first virtual accountant that predicts, ranks, and explains accounting actions with policy-safe outputs.

Non-goal:
- unrestricted autonomous posting.

Mandatory safety:
- human approval in high-risk workflows,
- policy and period-lock hard checks,
- full lineage: dataset hash, model hash, policy version, evaluation bundle.

## 2. Architecture Redesign

### 2.1 Ledger-aware decision modeling

Move from row-only inference to state-aware inference using:
- vendor/customer/account priors,
- recent posting behavior windows,
- open invoice/payment allocation context,
- workflow state and period state,
- tax/GST context and rule flags.

### 2.2 Multi-task copilot core

Target training architecture:
- shared backbone + specialized heads.

Heads:
- transaction intent/workflow,
- CoA debit,
- CoA credit,
- tax/GST treatment,
- review risk,
- reconciliation exception,
- next-best-action (click/process policy).

### 2.3 Retrieval + rerank

For long-tail accounting labels, use:
1. candidate retrieval (CoA/match candidates from historical memory and master data),
2. context rerank model,
3. policy/constraint filtering.

### 2.4 Constraint-aware finalization

ML scores candidates; constrained layer enforces accounting validity:
- debit=credit,
- period lock constraints,
- tax/GST constraints,
- allocation bounds,
- blocked/discouraged policy bands.

### 2.5 Personalization adapters

Serving model strategy:
- global base model,
- company adapter,
- user/role adapter.

Guardrails:
- min memory thresholds,
- capped top-1 shift rates,
- automatic revert-to-base on instability.

### 2.6 RAG for explanation and evidence

RAG scope:
- retrieve policy snippets, prior approved analogs, and workflow evidence,
- generate accountant-facing rationale and alternatives.

RAG is not allowed to replace accounting arithmetic or policy hard checks.

### 2.7 Concurrency model

Production-safe operating pattern:
- many-reader stateless inference,
- single-writer training/promotion,
- lock-safe feedback ingestion,
- immutable artifacts and replayable decisions.

## 3. Redesigned Roadmap

### Phase 0 (Completed): Safety baseline

Delivered:
- champion/challenger loops for tx+coa/product/risk/reconciliation,
- GST staged guardrails,
- override-memory enrichment,
- user personalization with guardrail auto-revert,
- lock-safe ingestion and single-writer loop locks.

### Phase 1 (Now): Copilot feature foundation

Deliver:
1. unified ledger-state feature contract,
2. multi-task training set builder (shared sample ID across heads),
3. backward-compatible adapters to existing per-head scripts.

Exit gates:
- no regression against current champion gates,
- stable workflow-sliced holdout metrics.

### Phase 2: Retrieval + rerank rollout

Deliver:
1. CoA candidate retriever,
2. invoice/PO/receipt matching candidate retriever,
3. rerank integration into advisory payloads.

Exit gates:
- top-k improvement on low-frequency classes,
- reduced override rates for long-tail mappings.

### Phase 3: Constraint-aware action engine

Deliver:
1. constrained decoder for proposed journal/action outputs,
2. explicit constraint violation taxonomy,
3. deterministic replay for blocked recommendations.

Exit gates:
- zero policy-invalid accepted outcomes in evaluation slices.

### Phase 4: Action policy copilot

Deliver:
1. next-best-action head for click/process prediction,
2. review routing policy from calibrated uncertainty,
3. abstain/escalate outputs for low-confidence cases.

Exit gates:
- improved review precision,
- lower correction-loop rate in pilot workflows.

### Phase 5: Explainable assistant layer

Deliver:
1. RAG index for policy/history evidence,
2. grounded explanation outputs with citations,
3. conflict-aware response mode.

Exit gates:
- accountant explanation acceptance target,
- no unsupported claim in sampled audits.

## 4. Data Program Redesign

Synthetic-first remains, but contracts expand from labels to decision process traces:
- `state -> candidates -> selected_action -> result`,
- constraint failure reasons,
- override reason and corrected action,
- personalization context (`company_code`, `user_id`, `role`).

Required IDs:
- `decision_id`, `sample_id`, `workflow_family`, `policy_version`, `model_version`.

## 5. Immediate 2-Sprint Plan

Sprint A:
1. ledger-state feature extraction schema and builder,
2. candidate retrieval index generation,
3. scorecard extension for candidate recall and rerank lift.

Sprint B:
1. retrieval+rereank in tx+coa advisory path (feature-flagged),
2. constrained post-processing for proposed accounting actions,
3. next-best-action prototype head using workflow traces.

## 6. Current Operations and Commands

Full continual tick:

```bash
cd /home/realnigga/Virtual-accountant-sync/erp-domain-ml-lab
bash scripts/17_run_full_continual_learning_tick.sh -t data/training/virtual_accountant_training_synthetic_batch_20260220.csv -u accountant.a
```

GPU run:

```bash
cd /home/realnigga/Virtual-accountant-sync/erp-domain-ml-lab
V2_DEVICE=cuda COA_DEVICE=cuda PRODUCT_DEVICE=cuda RISK_DEVICE=cuda RECON_DEVICE=cuda \
  bash scripts/17_run_full_continual_learning_tick.sh -t data/training/virtual_accountant_training_synthetic_batch_20260220.csv -u accountant.a
```

Learning monitor:

```bash
cd /home/realnigga/Virtual-accountant-sync/erp-domain-ml-lab
bash scripts/35_monitor_virtual_accountant_learning.sh
```

## 7. Governance Rules (Fixed)

- advisory-first default,
- gated promotion only,
- hard policy constraints before recommendation output,
- auditable replay for every decision path,
- automatic fallback to previous champion on regressions.
