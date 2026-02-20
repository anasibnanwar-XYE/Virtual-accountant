# ERP Domain ML Lab

This folder is a separate workspace to train and fine-tune ERP advisory models using your `AUDITOR` analytics stack.

Roadmap-aligned execution plan:
- `docs/ERP_WORKFLOW_ML_STAGING_PLAN.md`
- `docs/VIRTUAL_ACCOUNTANT_BLUEPRINT.md`
- `docs/INDIA_GST_BOOKKEEPING_ML_PLAYBOOK.md`
- `docs/ERP_V2_FLOW_AND_CONCURRENCY_PLAN.md`
- `docs/ERP_V2_SCHEMA_PERSONALIZATION_NOTES.md`
- `docs/VIRTUAL_ACCOUNTANT_DATA_SCHEMA.md`

Goal:
- Build accountant-assistive suggestions for ERP (not auto-posting).
- Keep accounting-safe behavior: model outputs are advisory, human-reviewed.

## What You Get

- Snapshot capture from ORCHESTRATOR ERP (`live` or `synthetic`).
- Training pipeline for:
  - `transaction classifier v2`
  - `CoA recommender v1`
  - `product-account recommender v1`
  - `review-risk model v1`
  - `reconciliation-exception model v1`
- Advisory inference pipeline:
  - transaction suggestions
  - account mapping suggestions
  - journal risk scoring
- Human feedback loop:
  - export review queue to CSV
  - import corrected labels
  - retrain label-driven model (v1 quick loop)

## Folder Layout

```text
erp-domain-ml-lab/
  .env.example
  scripts/
    _common.sh
    00_init.sh
    01_capture_snapshot.sh
    02_generate_training_data.sh
    03_train_models.sh
    04_run_advisory.sh
    05_export_review_csv.sh
    06_import_labels_and_train_v1.sh
    07_generate_training_data_from_snapshot.sh
    08_train_models_from_snapshot.sh
    09_train_workflow_custom_synthetic.sh
    10_train_product_account_recommender.sh
    11_run_product_account_suggestions.sh
    12_ingest_product_feedback.sh
    13_train_product_account_continual.sh
    14_run_continual_learning_tick.sh
    15_eval_tx_coa_scorecard.sh
    16_train_tx_coa_continual.sh
    17_run_full_continual_learning_tick.sh
    18_generate_explanation_pack.sh
    19_run_gst_tax_audit.sh
    20_run_tax_overlay_audit.sh
    21_update_gst_guardrail_stage.py
    22_run_long_horizon_autopilot.sh
    23_train_review_risk_model.sh
    24_train_review_risk_continual.sh
    25_train_reconciliation_exception_model.sh
    26_train_reconciliation_continual.sh
    27_ingest_override_reason_feedback.sh
    28_generate_workflow_routing.sh
    29_rebalance_training_by_erp_flow.py
    30_feedback_ingest_concurrency_smoke.sh
    31_run_compute_training_sweep.sh
    32_ingest_user_personalization_feedback.sh
    33_apply_user_personalization.sh
    34_personalization_guardrail_smoke.sh
    35_monitor_virtual_accountant_learning.sh
    accountant_explanation_pack.py
    gst_tax_audit.py
    tax_aware_coa_overlay.py
    product_feedback_ingest.py
    user_personalization_feedback_ingest.py
    user_personalization_rerank.py
    export_priority_review_csv.py
    override_reason_feedback_ingest.py
    override_reason_enrich_training_csv.py
    workflow_routing_policy.py
    workflow_routing_apply.py
    product_champion_gate.py
    review_risk_champion_gate.py
    review_risk_model.py
    reconciliation_champion_gate.py
    reconciliation_exception_model.py
    workflow_scorecard.py
    tx_coa_champion_gate.py
    product_account_recommender.py
    build_training_csv_from_snapshot.py
    virtual_accountant_learning_monitor.py
  data/
    templates/
      virtual_accountant_training_csv_template.csv
      virtual_accountant_product_feedback_template.csv
      virtual_accountant_override_feedback_template.csv
      virtual_accountant_user_personalization_feedback_template.csv
      virtual_accountant_review_labels_template.csv
      virtual_accountant_product_candidates_template.csv
  snapshots/
  models/
  outputs/
  cache/
  configs/
    erp_v2_flow_targets.json
    erp_v2_flow_targets_stage_a.json
```

## Quick Start (Synthetic)

From `/home/realnigga/erp-domain-ml-lab`:

```bash
bash scripts/00_init.sh
cp .env.example .env
# edit .env (company/email/base-url/password env var)

# 1) Make training CSV with journal lines (needed for CoA model)
bash scripts/02_generate_training_data.sh 3000

# 2) Train v2 classifier + CoA recommender
bash scripts/03_train_models.sh

# 3) Build snapshot (synthetic for quick test; use live for real ERP)
bash scripts/01_capture_snapshot.sh synthetic
# bash scripts/01_capture_snapshot.sh live

# 4) Run advisory outputs
bash scripts/04_run_advisory.sh
```

`scripts/03_train_models.sh` also trains the product-account recommender by default.

## Quick Start (Schema-Grounded, Real Snapshot)

This is the recommended path for ERP-domain adaptation:

```bash
bash scripts/00_init.sh
bash scripts/01_capture_snapshot.sh live

# Build training CSV from normalized ERP snapshot journals
bash scripts/07_generate_training_data_from_snapshot.sh

# Train v2 + CoA on that snapshot-derived dataset
bash scripts/03_train_models.sh
# or one-command flow:
# bash scripts/08_train_models_from_snapshot.sh

# Run advisory outputs
bash scripts/04_run_advisory.sh
```

Optional env knobs for snapshot-derived dataset generation:
- `SNAPSHOT_LABEL_FIELD` (default `syntheticLabel`)
- `BALANCE_TOLERANCE` (default `0.01`)
- `CURRENCY_CODE` (default `INR`)
- `INCLUDE_NON_POSTED=true` (default false)
- `ALLOW_UNKNOWN_LABELS=true` (default false)

## Workflow-Custom Synthetic Program

This workflow trains and evaluates a workflow-aware ERP model pack in one command:

```bash
bash scripts/09_train_workflow_custom_synthetic.sh 30000 17 77 4000
```

Arguments:
- `rows` (default `30000`)
- `train_seed` (default `17`)
- `holdout_seed` (default `77`)
- `holdout_journal_entries` (default `4000`)

Useful env overrides:
- `WF_PROFILE` (default `staging_m18`) values: `balanced`, `staging_m18`, `accounting_hardening`
- `WF_V2_EPOCHS` (default `6`)
- `WF_COA_EPOCHS` (default `6`)
- `WF_TX_THRESHOLD` (default `0.90`)
- `WF_TOPK` (default `3`)
- `ERP_V2_FLOW_REBALANCE` (default `true` in this script)
- `ERP_V2_FLOW_TARGETS_JSON` (default `configs/erp_v2_flow_targets.json`)

You can run standalone rebalance on an existing training CSV:

```bash
python3 scripts/29_rebalance_training_by_erp_flow.py \
  --input-csv /path/to/training.csv \
  --output-csv /path/to/training_rebalanced.csv \
  --profile-json configs/erp_v2_flow_targets.json \
  --report-out /path/to/rebalance_report.json
```

## Product Account Suggestions

Train a dedicated model for mapping product accounts (revenue/cogs/inventory/tax/discount):

```bash
bash scripts/10_train_product_account_recommender.sh
```

Run suggestions for new product entries:

```bash
bash scripts/11_run_product_account_suggestions.sh /path/to/product_candidates.csv
```

Accepted candidate input fields:
- `sku`
- `product_name`
- `category`
- `product_kind` (`FINISHED_GOOD` or `RAW_MATERIAL`)
- `uom`
- `gst_rate`
- `base_price`
- `avg_cost`

Output includes:
- top-k ML suggestions per account target
- combined best suggestion (ML + similar-product history)
- similar-product evidence for accountant review

## Review-Risk Model

Train review-risk model (advisory routing, not posting):

```bash
bash scripts/23_train_review_risk_model.sh
```

Run continual champion/challenger cycle for review-risk:

```bash
bash scripts/24_train_review_risk_continual.sh
# optional:
# bash scripts/24_train_review_risk_continual.sh -t /path/to/training.csv
```

## Reconciliation-Exception Model

Train reconciliation-exception model (advisory routing for unmatched/exception candidates):

```bash
bash scripts/25_train_reconciliation_exception_model.sh
```

Run continual champion/challenger cycle for reconciliation:

```bash
bash scripts/26_train_reconciliation_continual.sh
# optional:
# bash scripts/26_train_reconciliation_continual.sh -t /path/to/training.csv
```

## Fine-Tuning Loop (Human in the loop)

After running advisory:

```bash
# Export review queue
bash scripts/05_export_review_csv.sh
# if explanation/routing outputs exist, this auto-exports priority-ranked review queue.
# you can also pass run/explanations path:
# bash scripts/05_export_review_csv.sh /path/to/tx_coa_eval_run_or_explanations_dir

# Fill `chosen_label` in CSV manually, then:
bash scripts/06_import_labels_and_train_v1.sh /path/to/review_filled.csv
```

This quick loop retrains the label-driven classifier (v1) from accountant-reviewed examples.

For product-account model retraining with user corrections:

```bash
PRODUCT_FEEDBACK_JSONL=/path/to/product_feedback.jsonl \
  bash scripts/10_train_product_account_recommender.sh
```

For transaction/CoA override-memory ingestion:

```bash
bash scripts/27_ingest_override_reason_feedback.sh /path/to/override_feedback.csv
```

Canonical override memory file:
- `data/labels/override_reason_feedback_memory.jsonl`

For user-personalization memory ingestion:

```bash
# sample format: data/training/user_personalization_feedback_sample.csv
bash scripts/32_ingest_user_personalization_feedback.sh /path/to/user_personalization_feedback.csv
```

Canonical user personalization memory file:
- `data/labels/user_personalization_feedback_memory.jsonl`

Concurrency smoke test for feedback ingestion:

```bash
bash scripts/30_feedback_ingest_concurrency_smoke.sh 8
```

## Continual Learning (Always Improving)

Transaction + CoA champion/challenger loop:

```bash
bash scripts/16_train_tx_coa_continual.sh
# optional:
# bash scripts/16_train_tx_coa_continual.sh -t /path/to/training.csv -r /path/to/review_queue_filled.csv
```

Workflow scorecard generation for any model/snapshot pair:

```bash
bash scripts/15_eval_tx_coa_scorecard.sh /path/to/snapshot /path/to/tx_model_dir /path/to/coa_bundle_dir
```

Generate workflow-specific threshold + routing decisions from tx+coa outputs:

```bash
bash scripts/28_generate_workflow_routing.sh
# optional:
# bash scripts/28_generate_workflow_routing.sh /path/to/eval_or_advisory_run_dir /path/to/workflow_scorecard.json
```

Routing outputs:
- `workflow_routing/workflow_routing_policy.json`
- `workflow_routing/workflow_routing_decisions.jsonl`
- `workflow_routing/workflow_routing_summary.json`

Generate accountant-facing explanation templates from tx+coa outputs (optionally product suggestions):

```bash
bash scripts/18_generate_explanation_pack.sh
# optional:
# bash scripts/18_generate_explanation_pack.sh /path/to/advisory_or_eval_run_dir /path/to/product_suggest_run_dir
```

Explanation pack outputs:
- `transaction_coa_explanations.jsonl`
- `transaction_review_priority_queue.jsonl` (ranked manual-review queue)
- `product_account_explanations.jsonl` (when product suggestion input exists)
- `explanation_summary.json`
- `review_brief.md`

If workflow routing artifacts exist in the run dir, explanation pack auto-attaches:
- routed action/priority/reason
- workflow-family threshold used for routing

Action hints emitted per suggestion:
- `quick_confirm`
- `manual_review_required`
- `reject_or_override_required`

Run GST/tax accounting audit checks on tx+coa outputs:

```bash
bash scripts/19_run_gst_tax_audit.sh
# optional:
# bash scripts/19_run_gst_tax_audit.sh /path/to/advisory_or_eval_run_dir
```

GST audit knobs:
- `GST_AUDIT_MAX_MAJOR_FAIL_RATE` (default `0.02`)
- `GST_AUDIT_MAX_CRITICAL_ISSUES` (default `0`)

Run tax-aware overlay (fast GST hardening) and re-audit:

```bash
bash scripts/20_run_tax_overlay_audit.sh
# optional:
# bash scripts/20_run_tax_overlay_audit.sh /path/to/advisory_or_eval_run_dir
```

Overlay knob:
- `TAX_OVERLAY_TOPK_LIMIT` (default `5`)

Canonical feedback memory file:
- `data/labels/product_account_feedback_memory.jsonl`

Ingest new accountant feedback from CSV/JSONL:

```bash
# sample format: data/training/product_feedback_sample.csv
bash scripts/12_ingest_product_feedback.sh /path/to/product_feedback.csv
```

Run continual learning cycle (ingest optional + retrain + champion/challenger gate + promote/reject):

```bash
bash scripts/13_train_product_account_continual.sh
# with explicit inputs:
# bash scripts/13_train_product_account_continual.sh -t /path/to/training.csv -f /path/to/new_feedback.csv
```

Product continual loop efficiency control:
- `PRODUCT_CONTINUAL_SKIP_IF_NO_CHANGE=true` (default) skips retraining when training CSV, feedback memory, and product-model hyperparameters are unchanged.

One-command daily/weekly tick (transaction review loop optional):

```bash
bash scripts/14_run_continual_learning_tick.sh
# optionally include reviewed transaction CSV and new product feedback:
# bash scripts/14_run_continual_learning_tick.sh -r /path/to/review_queue_filled.csv -f /path/to/product_feedback.csv
```

Full one-command tick (tx+coa + product + review-risk + reconciliation loops):

```bash
bash scripts/17_run_full_continual_learning_tick.sh
# optional:
# bash scripts/17_run_full_continual_learning_tick.sh -t /path/to/training.csv -r /path/to/review_queue_filled.csv -f /path/to/product_feedback.csv -o /path/to/override_feedback.csv
# bash scripts/17_run_full_continual_learning_tick.sh -p /path/to/user_personalization_feedback.csv -u accountant.a
```

`scripts/17_run_full_continual_learning_tick.sh` also runs GST/tax audit on tx+coa outputs and can include:
- review-risk continual loop (`RUN_REVIEW_RISK_LOOP=true` by default)
- reconciliation continual loop (`RUN_RECONCILIATION_LOOP=true` by default)
- per-user personalization rerank artifacts (when `-u <user_id>` is provided)

Controlled virtual-accountant learning monitor:

```bash
bash scripts/35_monitor_virtual_accountant_learning.sh
```

This writes one normalized report across tx+coa/product/review-risk/reconciliation/personalization statuses + guardrail alerts.

Apply user personalization to latest tx+coa run (without retraining):

```bash
bash scripts/33_apply_user_personalization.sh accountant.a
# optional:
# bash scripts/33_apply_user_personalization.sh accountant.a /path/to/tx_coa_eval_run BBP
```

Run personalization guardrail regression smoke:

```bash
bash scripts/34_personalization_guardrail_smoke.sh
# optional sample override:
# bash scripts/34_personalization_guardrail_smoke.sh /path/to/user_personalization_feedback_sample.csv
```

Personalization quality guardrails (defaults, configurable via env):
- `USER_PERSONALIZATION_TX_ALPHA=0.25`
- `USER_PERSONALIZATION_COA_ALPHA=0.20`
- `USER_PERSONALIZATION_MIN_MEMORY_ROWS=5`
- `USER_PERSONALIZATION_MIN_FAMILY_MEMORY_ROWS=3`
- `USER_PERSONALIZATION_GLOBAL_ONLY_ALPHA_SCALE=0.40`
- `USER_PERSONALIZATION_MAX_TX_TOP1_CHANGE_RATE=0.30`
- `USER_PERSONALIZATION_MAX_COA_DEBIT_TOP1_CHANGE_RATE=0.35`
- `USER_PERSONALIZATION_MAX_COA_CREDIT_TOP1_CHANGE_RATE=0.35`
- `USER_PERSONALIZATION_MIN_FAMILY_EVAL_ROWS=25`
- `USER_PERSONALIZATION_MAX_FAMILY_TOP1_CHANGE_RATE=0.50`

Guardrail behavior:
- if user memory is below threshold, personalization is skipped with base outputs preserved
- if workflow-family memory is below threshold, rerank uses conservative global-only priors with reduced alpha scale
- if rerank top-1 shift rates exceed threshold, personalization is reverted to base outputs
- if any workflow family exceeds family-level top-1 shift threshold, personalization is reverted to base outputs
- full continual tick status prints as `User personalization: ran:<status>`

Synthetic data contract + templates:
- contract doc: `docs/VIRTUAL_ACCOUNTANT_DATA_SCHEMA.md`
- templates:
  - `data/templates/virtual_accountant_training_csv_template.csv`
  - `data/templates/virtual_accountant_product_feedback_template.csv`
  - `data/templates/virtual_accountant_override_feedback_template.csv`
  - `data/templates/virtual_accountant_user_personalization_feedback_template.csv`
  - `data/templates/virtual_accountant_review_labels_template.csv`
  - `data/templates/virtual_accountant_product_candidates_template.csv`

Example cron (daily at 01:30):

```bash
30 1 * * * cd /home/realnigga/erp-domain-ml-lab && PRODUCT_EPOCHS=4 PRODUCT_SYNTH_ROWS=3000 bash scripts/14_run_continual_learning_tick.sh >> /home/realnigga/erp-domain-ml-lab/outputs/continual_learning_cron.log 2>&1
```

Example full-loop cron (daily at 02:00):

```bash
0 2 * * * cd /home/realnigga/erp-domain-ml-lab && V2_EPOCHS=4 COA_EPOCHS=4 PRODUCT_EPOCHS=4 PRODUCT_SYNTH_ROWS=3000 bash scripts/17_run_full_continual_learning_tick.sh >> /home/realnigga/erp-domain-ml-lab/outputs/continual_learning_full_cron.log 2>&1
```

Promotion guardrails (defaults, configurable via env):
- `CL_MAX_DEGRADE_CORE=0.005` for `revenue/cogs/inventory/tax` holdout top-1
- `CL_MAX_DEGRADE_DISCOUNT=0.02` for `discount` holdout top-1
- `CL_MIN_OVERALL_DELTA=-0.001` for mean holdout top-1 across all targets
- `PRODUCT_CONTINUAL_SKIP_IF_NO_CHANGE=true` to avoid unnecessary product retrains when no new signal is present
- runtime env overrides passed on command invocation now take precedence over `.env` for key training knobs (epochs/features/loop toggles/personalization guardrails), enabling compute-efficient ad-hoc runs.

Review-risk guardrails (defaults, configurable via env):
- `RUN_REVIEW_RISK_LOOP=true`
- `RISK_MAX_FNR_INCREASE=0.02`
- `RISK_MAX_RECALL_DROP=0.02`
- `RISK_MIN_F1_DELTA=-0.01`
- `RISK_MIN_PRECISION=0.55`
- `RISK_MIN_ACCURACY=0.65`
- `RISK_MAX_PRED_POS_RATE_INCREASE=0.08`
- `RISK_MAX_PRED_POS_RATE_DROP=0.08`
- `RISK_OVERRIDE_MEMORY_JSONL=/path/to/override_reason_feedback_memory.jsonl`
- `RISK_CONTINUAL_SKIP_IF_NO_CHANGE=true`

Reconciliation guardrails (defaults, configurable via env):
- `RUN_RECONCILIATION_LOOP=true`
- `RECON_MAX_FNR_INCREASE=0.02`
- `RECON_MAX_RECALL_DROP=0.02`
- `RECON_MIN_F1_DELTA=-0.01`
- `RECON_MIN_PRECISION=0.55`
- `RECON_MIN_ACCURACY=0.65`
- `RECON_MAX_PRED_POS_RATE_INCREASE=0.08`
- `RECON_MAX_PRED_POS_RATE_DROP=0.08`
- `RECON_OVERRIDE_MEMORY_JSONL=/path/to/override_reason_feedback_memory.jsonl`
- `RECON_CONTINUAL_SKIP_IF_NO_CHANGE=true`

TX+CoA guardrails (defaults, configurable via env):
- `TX_COA_MAX_TX_DROP_PPM=2000`
- `TX_COA_MAX_COA_TOP1_DROP=0.004`
- `TX_COA_MAX_COA_TOP3_DROP=0.002`
- `TX_COA_MIN_OVERALL_DELTA=-0.001`
- `TX_COA_MIN_FAMILY_EXAMPLES=80`
- `TX_COA_MAX_FAMILY_TX_DROP=0.02`
- `TX_COA_MAX_POLICY_BLOCKED_TOP1_RATE=0.0`
- `TX_COA_MAX_POLICY_DISCOURAGED_TOP1_RATE=0.25`
- `TX_COA_MAX_POLICY_BLOCKED_TOP1_RATE_DELTA=0.0`
- `TX_COA_MAX_POLICY_DISCOURAGED_TOP1_RATE_DELTA=0.05`
- `TX_COA_THRESHOLD_KEY=0.900000`
- `TX_COA_MIN_AUTO_ACCEPT_RATE_PPM=0`
- `TX_COA_MIN_AUTO_ACCEPT_ACCURACY_PPM=900000`
- `TX_COA_MAX_AUTO_ACCEPT_RATE_DROP_PPM=80000`
- `TX_COA_MAX_AUTO_ACCEPT_ACCURACY_DROP_PPM=30000`
- `TX_COA_MAX_REVIEW_RATE_INCREASE_PPM=100000`
- `TX_COA_ENABLE_OVERRIDE_ENRICHMENT=true`
- `ROUTING_MIN_FAMILY_EXAMPLES=80`
- `ROUTING_TARGET_AUTO_ACCEPT_RATE_PPM=850000`
- `ROUTING_MIN_AUTO_ACCEPT_ACCURACY_PPM=980000`
- `ROUTING_PREFERRED_THRESHOLD=0.90`
- `ROUTING_CONSERVATIVE_THRESHOLD=0.95`
- `ROUTING_DEFAULT_THRESHOLD=0.90`
- `TX_COA_ALLOW_PERIOD_LOCK_AUTOACCEPT=false` (default strict)
- `TX_COA_GST_GUARDRAIL_PROFILE=stage1` (profiles: `permissive`, `stage1`, `stage2`, `stage3`)
- `TX_COA_GST_STAGE_AUTO=true` (auto-advance/demote profile based on cycle outcomes)
- `TX_COA_GST_STAGE_INITIAL=stage1`
- `TX_COA_GST_STAGE_PROMOTE_STAGE1_AFTER=2`
- `TX_COA_GST_STAGE_PROMOTE_STAGE2_AFTER=3`
- `TX_COA_GST_STAGE_PROMOTE_STAGE3_AFTER=5`
- `TX_COA_GST_STAGE_DEMOTE_AFTER_FAILS=2`
- `TX_COA_GST_STAGE_REQUIRE_PROMOTED_STATUS=false`
- `TX_COA_GST_STAGE_ALLOW_DEMOTE_TO_PERMISSIVE=false`

GST profile defaults:
- `permissive`: emergency/noisy bring-up (minimal blocking)
- `stage1`: soft launch, allows moderate GST failure while preventing sharp regressions
- `stage2`: tighter rollout, no GST regression allowed
- `stage3`: strict rollout, GST improvement expected each cycle

Default thresholds by profile:
- `permissive`: major_fail<=`1.0`, issue_rate<=`1.0`, major_fail_delta<=`0.05`, issues_delta<=`1000`, sale_delta<=`500`, purchase_delta<=`500`
- `stage1`: major_fail<=`0.75`, issue_rate<=`0.85`, major_fail_delta<=`0.02`, issues_delta<=`50`, sale_delta<=`25`, purchase_delta<=`25`
- `stage2`: major_fail<=`0.55`, issue_rate<=`0.70`, major_fail_delta<=`0.0`, issues_delta<=`0`, sale_delta<=`0`, purchase_delta<=`0`
- `stage3`: major_fail<=`0.35`, issue_rate<=`0.50`, major_fail_delta<=`-0.01`, issues_delta<=`-20`, sale_delta<=`-10`, purchase_delta<=`-10`

GST threshold env overrides (optional; override profile defaults):
- `TX_COA_MAX_GST_MAJOR_FAIL_RATE`
- `TX_COA_MAX_GST_ISSUE_RATE`
- `TX_COA_MAX_GST_CRITICAL_ISSUES`
- `TX_COA_MAX_GST_MAJOR_FAIL_RATE_DELTA`
- `TX_COA_MAX_GST_ISSUES_TOTAL_DELTA`
- `TX_COA_MAX_GST_SALE_MISSING_DELTA`
- `TX_COA_MAX_GST_PURCHASE_MISSING_DELTA`

Persistent stage state files:
- `outputs/current_tx_coa_gst_guardrail_profile.txt`
- `outputs/gst_guardrail_stage_state.json`

Run long-horizon unattended autopilot loops:

```bash
# 6 cycles, 10-minute sleep between cycles
bash scripts/22_run_long_horizon_autopilot.sh 6 600

# with fixed training CSV + benchmark snapshot
# bash scripts/22_run_long_horizon_autopilot.sh 12 900 /path/to/training.csv "" /path/to/benchmark_snapshot
# with override feedback input
# bash scripts/22_run_long_horizon_autopilot.sh 12 900 /path/to/training.csv "" /path/to/benchmark_snapshot "" /path/to/override_feedback.csv
```

Autopilot outputs:
- `outputs/latest_autopilot_run_dir.txt`
- `outputs/autopilot_run_<timestamp>/cycles_manifest.jsonl`
  - includes `tx_coa_cycle_summary`, `tx_coa_routing_summary`, `product_cycle_summary`, `review_risk_cycle_summary`, and `reconciliation_cycle_summary`

Champion pointers:
- `outputs/current_product_account_champion_bundle_dir.txt`
- `outputs/current_product_account_champion_metrics.json`
- `outputs/current_review_risk_champion_bundle_dir.txt`
- `outputs/current_review_risk_champion_metrics.json`
- `outputs/current_reconciliation_champion_bundle_dir.txt`
- `outputs/current_reconciliation_champion_metrics.json`
- `outputs/latest_override_reason_memory_jsonl.txt`
- `outputs/latest_override_reason_ingest_report.txt`
- `outputs/latest_user_personalization_memory_jsonl.txt`
- `outputs/latest_user_personalization_ingest_report.txt`
- `outputs/latest_user_personalization_dir.txt`
- `outputs/latest_user_personalization_report_json.txt`
- `outputs/latest_user_personalized_tx_jsonl.txt`
- `outputs/latest_user_personalized_coa_jsonl.txt`
- `outputs/latest_personalization_guardrail_smoke_dir.txt`
- `outputs/latest_personalization_guardrail_smoke_summary_json.txt`
- `outputs/latest_virtual_accountant_monitor_dir.txt`
- `outputs/latest_virtual_accountant_monitor_report_json.txt`
- `outputs/latest_workflow_routing_policy_json.txt`
- `outputs/latest_workflow_routing_decisions_jsonl.txt`
- `outputs/latest_workflow_routing_summary_json.txt`

Continual fingerprint state:
- `outputs/current_product_account_training_fingerprint.json`
- `outputs/current_review_risk_training_fingerprint.json`
- `outputs/current_reconciliation_training_fingerprint.json`

Each cycle writes auditable artifacts under:
- `outputs/continual_learning/product_account/cycle_<timestamp>/`
- `outputs/continual_learning/review_risk/cycle_<timestamp>/`
- `outputs/continual_learning/reconciliation/cycle_<timestamp>/`

Note:
- `scripts/00_init.sh` also creates `.venv/` and installs `AUDITOR` analytics dependencies there.

## Important Accounting Guardrails

- Advisory only: never auto-post journals from ML output.
- Keep approval in accounting workflow.
- Log model version + input snapshot hash for auditability.
- Enforce confidence thresholds; low-confidence predictions must go to review queue.

## “Can it work like an accountant?”

Near term: it can become a strong assistant (classification, account suggestions, risk flags, prioritization).
Not near term: autonomous chartered-accountant replacement. Use it as decision support under accountant control.
