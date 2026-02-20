# ERP v2 Flow + Concurrency Plan (ML Lab)

## 1) Source of Truth Used
- `Desktop/orchestrator_erp_stabilize/docs/system-map/Goal/ERP_STAGING_MASTER_PLAN.md`
- `Desktop/orchestrator_erp_stabilize/SCOPE.md`
- `Desktop/orchestrator_erp_stabilize/portal-permissions-matrix.md`

These define the target workflow reality for ML behavior:
- O2C, P2P, Production, Payroll, Returns/Reversals, Period Close, Tax (GST/non-GST), split settlements, and approval/override controls.

## 2) Human-Accountant Workflow Families for Training
ML training is now aligned to these workflow families:
- `o2c`: dealer order/dispatch/invoice/AR/receipt
- `p2p`: PO/GRN/vendor bill/AP/payment
- `inventory_production`: material consumption, production, stock moves, COGS/write-off
- `banking_settlement`: cash/bank receipts, split settlements, allocation behavior
- `tax_settlement`: GST output/input liability and tax payment contexts
- `payroll`: payroll journal, salary payable, payout clearing
- `returns_reversal`: credit/debit notes and reversal-linked postings
- `period_close`: close lock/reopen-sensitive period controls
- `approvals_override`: manual override and maker-checker contexts

Target family weights are in:
- `configs/erp_v2_flow_targets.json`

## 3) New Training Rebalance Step
Added script:
- `scripts/29_rebalance_training_by_erp_flow.py`

What it does:
- classifies each synthetic training row into an ERP workflow family
- rebalances training rows to target ERP-v2 family mix
- keeps the same CSV schema (safe for existing training CLIs)
- writes a coverage report with before/after counts and missing families

Wired into:
- `scripts/09_train_workflow_custom_synthetic.sh` (default enabled)
- `scripts/02_generate_training_data.sh` (optional via `ERP_V2_FLOW_REBALANCE=true`)

## 4) Concurrent User Safety (Persistence + Learning)
### 4.1 Feedback ingestion safety
Updated ingestors:
- `scripts/product_feedback_ingest.py`
- `scripts/override_reason_feedback_ingest.py`

Now they:
- acquire an OS file lock on memory JSONL (`*.lock`)
- dedupe safely inside lock (prevents race duplicates)
- append with fsync (durable write)

### 4.2 Continual training safety
Added lab lock utility in:
- `scripts/_common.sh` (`acquire_lab_lock`)

Applied to:
- `scripts/13_train_product_account_continual.sh`
- `scripts/16_train_tx_coa_continual.sh`
- `scripts/17_run_full_continual_learning_tick.sh`
- `scripts/24_train_review_risk_continual.sh`
- `scripts/26_train_reconciliation_continual.sh`

This enforces single-writer semantics for model promotion and pointer updates.

### 4.3 Concurrency smoke test
Added:
- `scripts/30_feedback_ingest_concurrency_smoke.sh`

It runs parallel feedback ingests to one memory file and verifies uniqueness is preserved.

## 5) Practical Operating Pattern for Multi-User ERP
- Inference path: many concurrent readers (stateless model usage).
- Feedback path: many concurrent writes (now lock-safe append/dedupe).
- Training path: one active trainer per loop/model lock domain.
- Promotion path: champion pointers update only from locked trainer flow.

This is the recommended “many readers + one controlled learner” pattern for ERP accounting safety.
