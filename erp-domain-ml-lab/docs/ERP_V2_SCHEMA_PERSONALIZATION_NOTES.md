# ERP v2 Schema Notes for Personalization

Goal: keep quality-first global models while adding safe per-user adaptation for advisory suggestions.

## 1) v2 Schema Anchors (Read-Only References)
- Multi-company partitioning is first-class (`company_id` on operational tables).
- Journal identity and accounting references are stable per company:
  - `journal_entries(company_id, reference_number)`
  - `journal_reference_mappings`
- User/company relation exists in auth scope:
  - `app_users`, `user_companies`

References reviewed:
- `Desktop/orchestrator_erp_stabilize/docs/db/SCHEMA_EXPECTATIONS.md`
- `Desktop/orchestrator_erp_stabilize/docs/db/ER_INTENT.md`
- `Desktop/orchestrator_erp_stabilize/docs/system-map/Goal/ERP_STAGING_MASTER_PLAN.md`

## 2) Personalization Design in ML Lab
- Global model quality remains champion-gated.
- User personalization is a post-model rerank layer (advisory only).
- Required personalization context:
  - `user_id`
  - `company_code`
  - optional `workflow_family`
  - correction targets (`approved_label`, debit/credit account corrections)

## 3) Implemented Pipeline
- Ingest user personalization memory:
  - `scripts/32_ingest_user_personalization_feedback.sh`
  - `scripts/user_personalization_feedback_ingest.py`
- Apply per-user reranking:
  - `scripts/33_apply_user_personalization.sh`
  - `scripts/user_personalization_rerank.py`
- Optional auto-apply in full continual tick:
  - `scripts/17_run_full_continual_learning_tick.sh -u <user_id>`
- Guardrail regression smoke:
  - `scripts/34_personalization_guardrail_smoke.sh`

## 4) Safety
- Personalization does not post entries.
- Existing tx+coa policy/routing/gst guardrails remain in force.
- Personalization now has its own quality guardrails:
  - minimum user-memory evidence gate,
  - workflow-family evidence gate (`min_family_memory_rows`),
  - conservative global-only fallback for low-evidence families,
  - maximum allowed top-1 rerank shift rates for tx/debit/credit,
  - maximum allowed per-family top-1 rerank shift rates,
  - automatic fallback to base (non-personalized) outputs on violation.
- No edits were made to ERP main code; all changes are in `erp-domain-ml-lab`.
