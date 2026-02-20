# Virtual Accountant Data Schema (Synthetic Contract)

Last updated: 2026-02-20

Use this as the exact contract for synthetic data generation for current training pipelines in `erp-domain-ml-lab`.

## 1) Core Training CSV (tx + coa + risk + reconciliation + product aggregation)

Template:
- `data/templates/virtual_accountant_training_csv_template.csv`

Accepted by:
- transaction classifier v2 loader: `ledgerstudio_analytics/ml/training_data_v2.py`
- CoA trainer loader: `ledgerstudio_analytics/trainers/coa_recommender_v1.py`
- risk/reconciliation trainers also consume this shape via lab scripts

### Required columns (hard-required)
- `type` (string, label)
- `reference` (string, unique-like document reference)
- `date` (YYYY-MM-DD)
- `sku` (string)
- `party` (string)
- `notes` (string)
- `qty` (number; can be 0)
- `price` (number; can be 0)
- `cost` (number; can be 0)
- `tax_rate` (number; use decimal fraction like `0.18`)

### Optional but strongly recommended columns
- `doc_type` (string)
- `doc_status` (string; common: `POSTED`, `LOCKED`, `DRAFT`)
- `memo` (string)
- `payment_method` (string; common: `NEFT`, `RTGS`, `UPI`, `CASH`, `CHEQUE`, `BANK_TRANSFER`, `MULTI_MODE`)
- `gst_treatment` (string; common: `TAXABLE`, `EXEMPT`, `NON_GST`)
- `gst_inclusive` (boolean-like string: `true`/`false`)
- `currency` (string; e.g. `INR`)
- `journal_lines` (string; required for CoA signal)

### Recommended label set for `type`
- `SALE`
- `PURCHASE`
- `PAYMENT`
- `SETTLEMENT_SPLIT`
- `TAX_SETTLEMENT`
- `PAYROLL`
- `SALE_RETURN`
- `COGS`
- `WRITE_OFF`
- `INVENTORY_COUNT`
- `PERIOD_LOCK`
- `OPENING_BALANCE`
- `JOURNAL_ADJUSTMENT`

### `journal_lines` grammar (exact parser-compatible format)
- entry separator: `||`
- token separator: `|`
- each line needs: `<ACCOUNT_CODE> | <D/C> | <AMOUNT> | <description optional>`
- example:
  - `AR | D | 11800 | dealer receivable || SALES | C | 10000 | revenue || GST_OUTPUT | C | 1800 | output gst`

Notes:
- CoA trainer requires valid non-empty `journal_lines` rows.
- `PERIOD_LOCK` rows can have empty `journal_lines`.

## 2) Review Labels CSV (manual correction loop)

Template:
- `data/templates/virtual_accountant_review_labels_template.csv`

Produced by:
- `scripts/05_export_review_csv.sh` / `scripts/export_priority_review_csv.py`

Imported by:
- `scripts/06_import_labels_and_train_v1.sh`

### Strict minimum required columns for import
- `example_id`
- `chosen_label`
- `text`

All other columns are advisory metadata for traceability and downstream memory ingestion.

## 3) Product Account Feedback CSV (continual product mapping)

Template:
- `data/templates/virtual_accountant_product_feedback_template.csv`

Canonical memory ingestor:
- `scripts/product_feedback_ingest.py`

### Required minimum
- `sku`

### Strongly recommended full fields
- `sku`, `product_name`, `product_kind`, `category`, `uom`
- `gst_rate`, `base_price`, `avg_cost`
- `revenue_account_code`, `cogs_account_code`, `inventory_account_code`, `tax_account_code`, `discount_account_code`

## 4) Override Reason Feedback CSV (why suggestions were overridden)

Template:
- `data/templates/virtual_accountant_override_feedback_template.csv`

Canonical memory ingestor:
- `scripts/override_reason_feedback_ingest.py`

### Practical minimum for useful learning
- one of `reference`/`reason_code`/`reason_text` must be present
- include suggested vs approved label/account codes whenever possible:
  - `suggested_label`, `approved_label`
  - `suggested_debit_account_code`, `approved_debit_account_code`
  - `suggested_credit_account_code`, `approved_credit_account_code`

Recommended columns:
- `reference`, `type`, `doc_type`, `doc_status`, `party`, `payment_method`, `gst_treatment`, `currency`
- fields above + `reason_code`, `reason_text`, `action`, `source`

## 5) User Personalization Feedback CSV (per-user adaptation)

Template:
- `data/templates/virtual_accountant_user_personalization_feedback_template.csv`

Canonical memory ingestor:
- `scripts/user_personalization_feedback_ingest.py`

### Required minimum
- user identifier: `actor_user_id` (or alias `user_id`)
- at least one approved correction:
  - `chosen_label` or
  - `approved_debit_account_code` or
  - `approved_credit_account_code`

### Recommended full fields
- `actor_user_id`, `company_code`, `workflow_family`, `record_referenceNumber`
- suggested/approved label + debit + credit codes
- `reason_code`, `reason_text`, `action`

## 6) Product Candidates CSV (for suggestion-time inference)

Template:
- `data/templates/virtual_accountant_product_candidates_template.csv`

Used by:
- `scripts/product_account_recommender.py suggest`

Recommended fields:
- `sku`, `product_name`, `category`, `product_kind`, `uom`, `gst_rate`, `base_price`, `avg_cost`

## 7) Placement and run commands

Suggested drop folder for your synthetic batches:
- `data/training/`

Common commands:
- full continual tick:
  - `bash scripts/17_run_full_continual_learning_tick.sh -t /path/to/training.csv -u accountant.a`
- ingest product feedback:
  - `bash scripts/12_ingest_product_feedback.sh /path/to/product_feedback.csv`
- ingest override reasons:
  - `bash scripts/27_ingest_override_reason_feedback.sh /path/to/override_feedback.csv`
- ingest personalization feedback:
  - `bash scripts/32_ingest_user_personalization_feedback.sh /path/to/user_personalization_feedback.csv`
