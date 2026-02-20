# Virtual Accountant Blueprint (Full ERP Scope)

Last updated: 2026-02-20

This blueprint defines how to move from advisory ML to a full virtual accountant assistant across ERP workflows, while keeping accounting control and audit safety.

## 1. Product Vision

The virtual accountant should:
1. understand business context from ERP transactions, masters, workflows, and history
2. suggest correct accounting outcomes with evidence
3. prepare drafts for accountants to approve
4. learn continuously from approvals, overrides, and corrections
5. stay advisory-first until strict safety gates are consistently passed

## 2. Scope (Full ERP, Not Single Model)

Required workflow coverage:
1. O2C: quotation, SO, dispatch, invoice, receipt, return, credit note
2. P2P: PR, PO, GRN, vendor bill, payment, debit note, return
3. Inventory: transfers, production issues/receipts, adjustments, write-offs
4. Banking: receipts, payments, reconciliations, settlement splits
5. Tax: GST input/output, tax settlement, TDS/TCS style flows
6. Payroll: payroll journals, liabilities, payout and settlement
7. Fixed assets: capitalization, depreciation, disposal, impairment
8. Period close: accruals, reversals, close checks, lock controls
9. Reconciliation: GL vs subledger checks, unmatched and anomaly routing

## 3. Capability Map

Core capability groups:
1. transaction intent and workflow state classification
2. debit/credit account recommendation and policy reranking
3. product/customer/vendor master-account suggestion
4. anomaly and risk scoring
5. reconciliation matching and exception prioritization
6. natural language accounting assistant (explainable, evidence-backed)
7. workflow next-best-action suggestions (review queue routing)

## 4. Target ML/AI Stack

Use a multi-model architecture:
1. discriminative models:
   - transaction classifier
   - CoA ranker (debit/credit)
   - product-account mapper
   - risk and review-likelihood model
2. retrieval and memory:
   - similar historical postings
   - tenant policy/rule memory
   - chart-of-accounts and taxonomy retrieval
3. LLM reasoning layer:
   - explain suggestions in accountant language
   - summarize differences between predicted and approved postings
   - generate draft narratives and working notes
4. policy engine:
   - hard constraints and blocked mappings
   - period/tax/workflow safety checks
   - maker-checker gating

## 5. Data Program (Synthetic -> Real -> Continual)

Dataset maturity:
1. D0: deterministic synthetic data (pipeline + schema coverage)
2. D1: synthetic with hard edge cases (reversals, split settlement, tax edges)
3. D2: snapshot-derived posted journals from ERP
4. D3: accountant-reviewed corrections and override reasons
5. D4: longitudinal tenant-specific behavior dataset

Mandatory labels and signals:
1. transaction type and workflow stage
2. debit/credit account ground truth
3. policy decision and override reason
4. confidence band and acceptance outcome
5. reconciliation result and closure outcome

## 6. Training and Evaluation System

Every model update must follow:
1. train challenger with fixed manifest
2. evaluate on stable holdouts by workflow/tax mode
3. compare to champion with guardrails
4. promote only if safe and better
5. publish immutable artifacts and run evidence

Minimum evaluation slices:
1. GST vs non-GST
2. each workflow family (O2C/P2P/Inventory/Payroll/Banking/Tax/Close)
3. high-risk labels (period lock, reversal chains, manual adjustments)
4. new-master cold-start vs known-master warm-start

## 7. Safety, Control, and Governance

Non-negotiable controls:
1. advisory-only until autopilot criteria are passed
2. no posting without explicit approval in high-risk flows
3. zero tolerance for policy-invalid accepted suggestions
4. full lineage for each decision:
   - model version
   - training manifest hash
   - snapshot/data hash
   - policy version
5. deterministic replay for audit and incident review

## 8. Rollout Stages

### Stage A: Full Advisory Coverage

Goal:
1. assistant suggests and explains across all modules
2. accountants approve or edit

Exit criteria:
1. strong top-k quality in all key workflows
2. review queue quality improves (less noise, better prioritization)

### Stage B: Auto-Draft Mode

Goal:
1. assistant prepares full draft postings and notes
2. accountant approves in one click for low-risk cases

Exit criteria:
1. high acceptance in low-risk classes
2. no policy-invalid approved suggestions

### Stage C: Guarded Autopilot (Low Risk Only)

Goal:
1. auto-apply only for explicitly whitelisted scenarios
2. continuous post-facto audit and rollback hooks

Exit criteria:
1. stable precision and near-zero incident rate
2. strong drift detection and automatic fallback to advisory

### Stage D: Virtual Accountant Assistant

Goal:
1. assistant handles end-to-end accounting operations as drafts plus selective autopilot
2. humans focus on exceptions, controls, and business judgment

## 9. Operating Model

Team lanes:
1. accounting policy and controls
2. data generation and labeling
3. model training and evaluation
4. MLOps and deployment safety
5. product/workflow integration

Cadence:
1. daily inference monitoring
2. weekly training and challenger evaluation
3. bi-weekly accounting review of top override patterns
4. monthly governance and threshold recalibration

## 10. KPI Framework

Quality KPIs:
1. debit/credit top-1 and top-3 per workflow
2. policy-invalid suggestion rate
3. false auto-accept rate
4. calibration and coverage at thresholds

Business KPIs:
1. posting preparation time reduction
2. first-pass acceptance rate
3. close-cycle duration reduction
4. reconciliation backlog reduction

Trust KPIs:
1. explainability acceptance by accountants
2. override reason distribution trending down in known patterns
3. drift detection lead time

## 11. 90-Day Execution Blueprint

### Days 1-30

1. freeze advisory contracts for all current modules
2. formalize feedback schema across transaction and product suggestions
3. build baseline scorecards by workflow/tax mode
4. enable scheduled continual-learning ticks in lab

### Days 31-60

1. add reconciliation and risk models to the same champion/challenger pipeline
2. integrate override reasons into feature set
3. deploy workflow-specific thresholding and routing
4. run accountant acceptance pilots on 2-3 workflows

### Days 61-90

1. launch auto-draft mode for low-risk cohorts
2. run governance audit pack (lineage, replay, safety)
3. publish go/no-go for guarded autopilot in narrow scope
4. finalize Stage B handoff checklist

## 12. Immediate Next Builds in This Lab

1. expand continual-learning loop from product-account to:
   - CoA recommender
   - transaction classifier
   Status: completed with `scripts/16_train_tx_coa_continual.sh`
2. create workflow-sliced scorecard generator
   Status: completed with `scripts/workflow_scorecard.py` + `scripts/15_eval_tx_coa_scorecard.sh`
3. create policy-invalid regression tests as release gate
   Status: completed via policy-band and period-lock guardrails in `scripts/tx_coa_champion_gate.py`
4. add explanation templates for accountant-facing review UI
   Status: completed with `scripts/accountant_explanation_pack.py` + `scripts/18_generate_explanation_pack.sh`
5. add GST/tax audit guardrails for India-focused accounting behavior
   Status: completed with `scripts/gst_tax_audit.py` + `scripts/19_run_gst_tax_audit.sh`
6. automate GST guardrail stage progression across continual cycles
   Status: completed with `scripts/21_update_gst_guardrail_stage.py`
7. add review-risk model to champion/challenger continual pipeline
   Status: completed with `scripts/review_risk_model.py` + `scripts/23_train_review_risk_model.sh` + `scripts/24_train_review_risk_continual.sh`
8. add reconciliation-exception model to champion/challenger continual pipeline
   Status: completed with `scripts/reconciliation_exception_model.py` + `scripts/25_train_reconciliation_exception_model.sh` + `scripts/26_train_reconciliation_continual.sh`
9. integrate override-memory learning and acceptance-quality drift guardrails
   Status: completed with `scripts/27_ingest_override_reason_feedback.sh` + `scripts/override_reason_enrich_training_csv.py` + threshold drift checks in `scripts/tx_coa_champion_gate.py` + predicted-positive-rate drift checks in `scripts/review_risk_champion_gate.py` and `scripts/reconciliation_champion_gate.py`
10. deploy workflow-specific thresholding and routing artifacts
   Status: completed with `scripts/workflow_routing_policy.py` + `scripts/workflow_routing_apply.py` + `scripts/28_generate_workflow_routing.sh` and automatic generation in `scripts/15_eval_tx_coa_scorecard.sh`
