# ERP Workflow ML Staging Plan (Synthetic-First)

Last updated: 2026-02-20

This plan aligns ML training with `ERP_STAGING_MASTER_PLAN` stability goals.

Full-scope target blueprint:
- `docs/VIRTUAL_ACCOUNTANT_BLUEPRINT.md`
- `docs/INDIA_GST_BOOKKEEPING_ML_PLAYBOOK.md`

## 1. Goal and Guardrails

- Goal: advisory-only ERP intelligence for transaction classification, CoA linkage, and product-account suggestions.
- Non-goal: autonomous posting.
- Hard guardrails:
  - keep human approval before posting,
  - keep confidence threshold + review queue,
  - keep model/artifact lineage per run (dataset + model hashes).

## 2. M18 Roadmap Mapping

- M18-S3 workflow closure (O2C/P2P): workflow-aware features + policy reranking.
- M18-S5 split settlement safety: synthetic split settlement patterns + split-entry features.
- M18-S6 GST/non-GST hardening: GST/tax settlement synthetic patterns + GST account features.
- M18-S4 override hardening: policy banding (`preferred`/`discouraged`/`blocked`) in CoA rerank.

## 3. Synthetic Data Program

Use deterministic synthetic profiles:

- `balanced`: broad coverage baseline.
- `staging_m18`: heavier payment/settlement/GST focus.
- `accounting_hardening`: strongest stress on edge postings and reversals.

Synthetic scenarios now included in training corpus:

- cash vs receivable sales settlement,
- GRN/GRNI-style purchase postings,
- split settlements (bank+cash+discount lines),
- tax settlement postings,
- payroll journal/payment clearing patterns,
- standard COGS/write-off/inventory count/period lock flows.

ERP-v2 flow rebalance (new):
- workflow custom training now rebalances synthetic rows to ERP-v2 family targets by default:
  - `scripts/29_rebalance_training_by_erp_flow.py`
  - `configs/erp_v2_flow_targets.json`
- this keeps the same CSV schema but shifts family distribution toward roadmap-critical flows.

## 4. Training Strategy

1. Train transaction classifier v2 with workflow profile `staging_m18`.
2. Train CoA recommender v1 on same corpus (rows with journal lines).
3. Train product-account recommender on synthetic + transaction-derived product aggregates.
4. Run holdout synthetic snapshot eval.
5. Gate on metrics:
   - tx accuracy and auto-accept safety at 0.90/0.95 thresholds,
   - CoA top-1/top-3 for debit+credit,
   - product-account top-1/top-3 across revenue/cogs/inventory/tax/discount,
   - workflow policy output presence and consistency.

## 5. Runtime Policy Layer

CoA reranker policy now reasons over workflow families:

- sale / purchase / payment,
- settlement_split,
- tax_settlement,
- payroll,
- returns / cogs / write-off / inventory_count / period_lock.

Each recommendation includes:

- `raw_score`,
- `policy_delta`,
- `policy_band`,
- `policy_reason`,
- `workflow_context`, `policy_summary`.

## 6. Operational Runbook

Primary one-command run:

```bash
cd /home/realnigga/erp-domain-ml-lab
WF_PROFILE=staging_m18 WF_V2_EPOCHS=6 WF_COA_EPOCHS=6 \
  bash scripts/09_train_workflow_custom_synthetic.sh 30000 17 77 4000
```

Quick baseline generation:

```bash
cd /home/realnigga/erp-domain-ml-lab
SYNTHETIC_WORKFLOW_PROFILE=balanced bash scripts/02_generate_training_data.sh 50000
# optional:
# ERP_V2_FLOW_REBALANCE=true bash scripts/02_generate_training_data.sh 50000
```

Product-account training:

```bash
cd /home/realnigga/erp-domain-ml-lab
bash scripts/10_train_product_account_recommender.sh
```

## 7. Exit Criteria for “Virtual Accountant Readiness Prep”

Before any broader autonomy experiments:

- stable accuracy across multiple seeds,
- stable CoA top-1/top-3 across multiple seeds,
- no policy violations for blocked account classes in target workflows,
- strict human-approval workflow remains enforced in ERP.

## 8. Continual Learning Loop (Synthetic-First -> Real Feedback)

Product-account continual loop is now formalized with champion/challenger gating:

1. ingest accountant corrections into canonical memory:
   - `bash scripts/12_ingest_product_feedback.sh /path/to/feedback.csv`
   - memory file: `data/labels/product_account_feedback_memory.jsonl`
2. train challenger with synthetic + transaction aggregates + memory:
   - `bash scripts/13_train_product_account_continual.sh`
3. gate for safe promotion:
   - no material drop on core targets (`revenue/cogs/inventory/tax`)
   - bounded drop allowance on `discount`
   - bounded overall mean holdout drop
4. auto-promote only when gate passes; otherwise revert to previous champion pointers.
5. efficiency hardening for unattended runs:
   - product continual loop can skip retraining when no data/feedback/parameter change is detected (`PRODUCT_CONTINUAL_SKIP_IF_NO_CHANGE=true`)
   - each cycle still writes auditable summary artifacts and updates fingerprint state

Operational cadence:
- daily or weekly `scripts/14_run_continual_learning_tick.sh`
- include `-r reviewed_transaction_queue.csv` when transaction review labels are available
- include `-f product_feedback.csv` for new product-account corrections
- controlled status snapshot across tx+coa/product/risk/reconciliation/personalization:
  - `bash scripts/35_monitor_virtual_accountant_learning.sh`

TX+CoA continual loop (implemented):
- evaluate any tx+coa pair with workflow slices:
  - `bash scripts/15_eval_tx_coa_scorecard.sh <snapshot> <tx_model_dir> <coa_bundle_dir>`
- generate workflow-specific threshold/routing policy + routed actions:
  - `bash scripts/28_generate_workflow_routing.sh <run_dir> <workflow_scorecard.json>`
- champion/challenger tx+coa cycle:
  - `bash scripts/16_train_tx_coa_continual.sh`
- full one-command continual tick (tx+coa + product + review-risk):
  - `bash scripts/17_run_full_continual_learning_tick.sh`
- accountant-facing explanation pack for review UI templates:
  - `bash scripts/18_generate_explanation_pack.sh`
- GST/tax advisory audit pass on tx+coa outputs:
  - `bash scripts/19_run_gst_tax_audit.sh`

Hard safety checks now included in tx+coa promotion gate:
- blocked top-1 policy-band rate limit
- discouraged top-1 policy-band rate limit
- drift limits for blocked/discouraged top-1 policy rates
- threshold-level acceptance-quality drift limits (auto-accept rate, auto-accept accuracy, review-rate)
- strict `PERIOD_LOCK` auto-accept protection (disabled only via explicit override env)
- GST/tax audit regression checks (major fail-rate, critical issues, and sale/purchase tax-missing deltas)
- staged GST rollout supported by `TX_COA_GST_GUARDRAIL_PROFILE` (`stage1` -> `stage2` -> `stage3`)
- stage progression can be automated in continual loop with persistent state (`scripts/21_update_gst_guardrail_stage.py`)
- workflow-family threshold policy artifacts generated each tx+coa eval:
  - `workflow_routing_policy.json`
  - `workflow_routing_decisions.jsonl`
  - `workflow_routing_summary.json`

Override memory learning (implemented):
- ingest accountant override reasons into canonical memory:
  - `bash scripts/27_ingest_override_reason_feedback.sh /path/to/override_feedback.csv`
  - memory file: `data/labels/override_reason_feedback_memory.jsonl`
- tx+coa continual cycle enriches training CSV using override memory before retraining:
  - `scripts/override_reason_enrich_training_csv.py`
- review-risk and reconciliation loops include override memory in train fingerprints and optional model training input.

Review-risk continual loop (implemented):
- baseline train:
  - `bash scripts/23_train_review_risk_model.sh`
- champion/challenger continual cycle:
  - `bash scripts/24_train_review_risk_continual.sh`
- predicted-positive-rate drift guardrails added in champion/challenger gate.
- included in full one-command continual tick:
  - `bash scripts/17_run_full_continual_learning_tick.sh`

Reconciliation continual loop (implemented):
- baseline train:
  - `bash scripts/25_train_reconciliation_exception_model.sh`
- champion/challenger continual cycle:
  - `bash scripts/26_train_reconciliation_continual.sh`
- predicted-positive-rate drift guardrails added in champion/challenger gate.
- included in full one-command continual tick:
  - `bash scripts/17_run_full_continual_learning_tick.sh`

User personalization learning (implemented):
- ingest user-level correction memory:
  - `bash scripts/32_ingest_user_personalization_feedback.sh /path/to/user_personalization_feedback.csv`
  - memory file: `data/labels/user_personalization_feedback_memory.jsonl`
- apply per-user rerank to tx+coa advisory outputs:
  - `bash scripts/33_apply_user_personalization.sh <user_id> [tx_coa_eval_run_dir] [company_code]`
- optional in full one-command continual tick:
  - `bash scripts/17_run_full_continual_learning_tick.sh -u <user_id>`
- personalization is advisory-only and does not bypass policy/routing guardrails.
- quality-first personalization guardrails:
  - minimum user-memory rows required before applying rerank,
  - minimum workflow-family memory rows before family priors can be used,
  - global-only conservative rerank for low-evidence workflow families,
  - max allowed top-1 shift rates for tx/debit/credit rerank outputs,
  - max allowed per-family top-1 shift rates on sufficiently represented families,
  - auto-revert to base outputs when guardrails are violated.

Concurrency/persistence hardening (implemented):
- feedback ingestors now use lock-safe append + dedupe and fsync durability:
  - `scripts/product_feedback_ingest.py`
  - `scripts/override_reason_feedback_ingest.py`
- continual loops now enforce single-writer locking:
  - `scripts/13_train_product_account_continual.sh`
  - `scripts/16_train_tx_coa_continual.sh`
  - `scripts/17_run_full_continual_learning_tick.sh`
  - `scripts/24_train_review_risk_continual.sh`
  - `scripts/26_train_reconciliation_continual.sh`
- lock utility:
  - `scripts/_common.sh` (`acquire_lab_lock`)
