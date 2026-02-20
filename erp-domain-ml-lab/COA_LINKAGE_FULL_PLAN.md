# ERP CoA Linkage Program (Roadmap-Aligned)

Last updated: 2026-02-19

This is the full plan for training ERP-domain ML suggestions so the system can propose:
- likely debit account(s)
- likely credit account(s)
- confidence + review flag
- accountant override capture for continuous learning

Companion full-scope virtual accountant blueprint:
- `docs/VIRTUAL_ACCOUNTANT_BLUEPRINT.md`

Scope is advisory-first and accounting-safe:
- no autonomous ledger posting
- human approval remains mandatory
- all suggestions must respect ERP accounting and workflow invariants

## 1) Alignment With ERP Staging Master Plan

Source roadmap:
- `Desktop/orchestrator_erp/docs/system-map/Goal/ERP_STAGING_MASTER_PLAN.md`
- `Desktop/orchestrator_erp/tickets/ERP_STAGING_MASTER_PLAN_EXECUTION_QUEUE.md`

Roadmap constraints that govern ML:
- stabilization first, no speculative feature expansion
- canonical write path only
- deterministic idempotency and replay safety
- hard accounting invariants (balanced journals, close boundaries, explicit reversals)
- virtual accountant remains deferred until base stabilization

M18 crosswalk for ML work:
- `M18-S3` workflow census and duplicate-path closure: use canonical event path as the only training source.
- `M18-S4` approval and override hardening: capture ML overrides with reason codes for re-training.
- `M18-S5` split-payment idempotency closure: include split-settlement cases and replay-safe labels.
- `M18-S6` GST/non-GST hardening: train and evaluate separately by tax mode and enforce tax-safe account constraints.
- `M18-S9` API contract maturity: freeze advisory JSON contracts and block drift.
- `M18-S10` staging rehearsal evidence: include model reproducibility and policy-violation metrics in evidence pack.

## 2) Target Product Behavior

For each transaction/journal candidate:
- top-k debit recommendations
- top-k credit recommendations
- confidence scores (calibrated where available)
- review routing signal (`auto_accept` only for high-confidence safe bands)
- immutable model/snapshot identity hashes

Acceptance behavior:
- high confidence: pre-fill suggestion, accountant confirms
- medium confidence: pre-fill with warning
- low confidence: force review queue

## 3) Training Objectives

Primary supervised objectives:
1. transaction intent label prediction (existing `transaction_classifier.v2`)
2. debit account ranking (existing `coa_recommender.v1` debit head)
3. credit account ranking (existing `coa_recommender.v1` credit head)
4. product-account mapping suggestions for master setup (`product_account_recommender.v1`)

Secondary objectives:
1. review-likelihood prediction from historical overrides
2. policy-violation propensity estimate for proactive routing

Long-term objective (post-stabilization):
1. intent-to-simulation assistant that proposes posting outcomes before any approval

## 4) Dataset Program (D0 -> D4)

- `D0`: synthetic baseline (pipeline and determinism smoke tests)
- `D1`: synthetic + weak labels + review loop labels
- `D2`: real snapshot-derived journals (posted-only, balanced-only)
- `D3`: accountant-reviewed correction set from review queue feedback
- `D4`: workflow-complete staged dataset (O2C, P2P, production, payroll, reversals, settlement edges)

Quality gates per dataset build:
- enforce required CSV fields expected by v2/CoA trainers
- include deterministic `journal_lines` serialization
- drop entries with missing postings
- drop entries failing debit/credit balance tolerance
- default to posted-only rows

## 5) Feature Plan (Schema-Aware)

Base features already supported:
- text: reference, memo, journal text
- numeric: amount, line totals, tax flags, account-type totals
- context: party frequency and party/global account priors

ERP schema enrichments to add next:
1. workflow state features:
   - source module and stage (order, GRN, invoice, settlement, reversal)
2. period-control features:
   - open/closed period flag, posting-at-boundary markers
3. tax mode features:
   - GST vs non-GST tenant mode, tax treatment lineage
4. settlement features:
   - split payment count, allocation fraction, idempotency conflict marker
5. approval features:
   - override domain, reason code family, maker-checker context

## 6) Safety and Governance Layer

Hard gates after model inference:
1. suggested account code must exist and be active in tenant CoA
2. type-safe mapping checks per transaction class (example: avoid AP as sale-receipt debit)
3. blocked/closed period transactions must never auto-accept
4. always-review label list for high-risk classes (`PERIOD_LOCK`, manual adjustments, reversal chains)

Operational safeguards:
1. advisory-only outputs in this phase
2. deterministic reruns: same snapshot + same model => same output
3. immutable artifact linking:
   - snapshot manifest hash
   - model hash
   - training manifest hash
   - config hash

## 7) Metrics and Release Gates

Core quality metrics:
- debit top-1/top-3 accuracy
- credit top-1/top-3 accuracy
- confidence calibration error (ECE)
- coverage at threshold bands
- false auto-accept rate
- policy-invalid suggestion rate

Suggested go-live thresholds:
- debit top-1 >= 82%
- credit top-1 >= 82%
- debit/credit top-3 >= 95%
- policy-invalid on accepted suggestions = 0%
- stable rerun drift = 0% for fixed inputs

Business metrics:
- manual account selection time reduction
- suggestion acceptance rate
- override rate by workflow and account family

## 8) Implementation Plan (Phased)

### Phase A: Stabilized Baseline (Now)
1. keep synthetic and current advisory loop operational
2. run snapshot-derived dataset generation for real schema grounding
3. train v2 + CoA from snapshot-derived CSV
4. validate with accountant sample review

Deliverable: stable advisory pipeline grounded in actual ERP snapshot schema.

### Phase B: Staging-Ready Training Discipline
1. split evaluation by GST/non-GST and workflow family
2. freeze advisory contracts for frontend integration
3. add policy-gate regression checks into training release checklist
4. maintain champion/challenger model comparison per training cycle

Deliverable: release-gated model updates aligned to `M18-S6/S9/S10`.

### Phase C: Controlled Production Rollout
1. rollout by tenant cohort and workflow scope
2. enable override reason capture everywhere suggestions appear
3. schedule weekly retraining and drift monitoring

Deliverable: measurable productivity gains without accounting safety regressions.

### Phase D: Virtual Accountant Readiness (Deferred)
1. define intent schema and simulation-only output contract
2. route all AI outcomes through approval and canonical APIs

Deliverable: simulation-grade assistant, still non-autonomous for posting.

## 9) Commands (Current Lab)

```bash
cd /home/realnigga/erp-domain-ml-lab
bash scripts/00_init.sh

# Option A: synthetic training baseline
bash scripts/02_generate_training_data.sh 50000
bash scripts/03_train_models.sh

# Option B: schema-grounded training from snapshot
bash scripts/01_capture_snapshot.sh live
bash scripts/07_generate_training_data_from_snapshot.sh
bash scripts/03_train_models.sh
# or single command:
# bash scripts/08_train_models_from_snapshot.sh

# Advisory inference
bash scripts/04_run_advisory.sh

# Product account model (new product entry assistance)
bash scripts/10_train_product_account_recommender.sh
# bash scripts/11_run_product_account_suggestions.sh /path/to/product_candidates.csv

# Human feedback loop
bash scripts/05_export_review_csv.sh
# fill chosen_label in CSV
bash scripts/06_import_labels_and_train_v1.sh /path/to/review_filled.csv
```

## 10) Immediate Next Sprint Focus

1. run weekly snapshot-derived training cycles (not only synthetic)
2. evaluate confusion hot spots for `SALE`, `PURCHASE`, `PAYMENT`, `SALE_RETURN`, `COGS`
3. enforce policy-invalid suggestion checks before promoting challenger model
4. feed override reasons into label loop for better long-tail account mapping

## 11) Continual Learning Implementation (Now in Lab)

For product master account mapping suggestions:

1. capture accountant corrections in CSV/JSONL and ingest:
   - `bash scripts/12_ingest_product_feedback.sh /path/to/product_feedback.csv`
2. run continual training + promotion gate:
   - `bash scripts/13_train_product_account_continual.sh`
3. optional recurring tick:
   - `bash scripts/14_run_continual_learning_tick.sh`

Promotion uses champion/challenger guardrails on holdout top-1:
- core targets (`revenue`, `cogs`, `inventory`, `tax`) must not degrade beyond small tolerance
- `discount` has separate tolerance band
- overall mean holdout top-1 must remain stable within tolerance

Artifacts and audit trail:
- champion pointers:
  - `outputs/current_product_account_champion_bundle_dir.txt`
  - `outputs/current_product_account_champion_metrics.json`
- per-cycle decision/report:
  - `outputs/continual_learning/product_account/cycle_<timestamp>/`

TX+CoA champion/challenger now implemented in lab:
- evaluation scorecard:
  - `scripts/15_eval_tx_coa_scorecard.sh`
  - `scripts/workflow_scorecard.py`
- gate comparator:
  - `scripts/tx_coa_champion_gate.py`
  - includes policy-invalid regression checks (blocked/discouraged top-1 bands, period-lock safety)
- continual cycle:
  - `scripts/16_train_tx_coa_continual.sh`
- full tick including tx+coa and product:
  - `scripts/17_run_full_continual_learning_tick.sh`
