# India GST + Bookkeeping ML Playbook (Synthetic-First)

Last updated: 2026-02-19

Scope:
- strengthen ERP advisory ML for Indian GST-aware accounting behavior,
- keep human approval and auditability,
- use synthetic-first training until production feedback volume is sufficient.

## 1) Primary References Used

GST legal/procedural references:
- GST Council portal (latest decisions/notifications): https://gstcouncil.gov.in/
- GST portal tutorials + FAQs (returns, GSTR-2B, RCM flows): https://tutorial.gst.gov.in/userguide/returns/index.htm
- ICAI GST publication hub (technical compendium, updates): https://idtc.icai.org/gstpublications.php
- ICAI technical document (reverse charge): https://resource.cdn.icai.org/84775bos68359-3.pdf
- ICAI technical document (blocked credit under section 17(5)): https://resource.cdn.icai.org/84809bos68359-6.pdf
- ICAI technical document (returns + payment mechanisms): https://resource.cdn.icai.org/84818bos68359-12.pdf

Bookkeeping/accounting references:
- ICAI Self-Paced Online Module (recording, ledger, trial balance, final accounts): https://boslive.icai.org/self-paced-online-modules/
- Open University accounting concepts chapter (objectives + qualitative characteristics): https://www.open.edu/openlearn/mod/oucontent/view.php?id=31733

Note:
- This playbook is implementation guidance for ERP advisory ML, not legal/tax advice.
- GST law changes frequently through notifications/circulars; keep periodic review with a CA/tax advisor.

## 2) Accounting Ground Truth We Want the Model to Respect

Core bookkeeping invariants:
1. Every suggested posting must be double-entry balanced.
2. Suggested account class must match economic event.
3. Trial-balance integrity and period controls must stay intact.
4. Low confidence or policy-sensitive cases must route to review.

GST-specific intent:
1. Taxable sales should surface output-tax liability signals (`GST_OUTPUT` / `TAX_PAYABLE`).
2. Taxable purchases should surface input-tax credit signals (`GST_INPUT`) where eligible.
3. Non-GST/exempt flows should not over-suggest tax accounts.
4. Tax settlement/payment workflows should surface liability + bank/cash sides.
5. Reverse-charge-like patterns should be explicitly review-routed unless high-confidence and policy-safe.

## 3) Synthetic Data Requirements (Next Iteration)

Add dedicated GST cohorts in synthetic generation:
1. intra-state B2B sale (CGST+SGST style tax split represented via configured tax account labels),
2. inter-state sale (IGST style),
3. taxable purchase with eligible ITC,
4. blocked-credit purchase patterns (section 17(5)-type contexts),
5. reverse charge purchase/service patterns,
6. tax settlement and payment cycles,
7. credit/debit note adjustments with tax impact.

For each cohort generate:
- correct posting patterns,
- near-miss alternatives (wrong tax side, missing tax line),
- narrative variants (memo/reference wording),
- confidence stress cases for review-queue calibration.

## 4) Feature Engineering Upgrades for GST/Tax Reasoning

Current lab already has useful signals (`tax_rate`, `gst_treatment_*`, `has_tax_line`, workflow/doc-type features).
Add/derive:
1. `gst_direction_hint` (input-vs-output expected side),
2. `tax_account_presence_ratio` by side in candidate top-k,
3. `is_tax_settlement_context`,
4. `rcm_hint_from_text`,
5. `itc_eligibility_proxy` (synthetic tag, later feedback-derived),
6. `period_tax_cutoff_risk` near return/settlement dates.

## 5) Evaluation + Release Gates (GST Layer)

Track these GST-specific metrics per model pair:
1. taxable sale rows where credit top-k includes output-tax account,
2. taxable purchase rows where debit top-k includes input-tax account,
3. non-GST rows with tax account at top-1 (should be near zero),
4. tax settlement rows with liability + liquid account signal coverage,
5. blocked/discouraged policy top-1 rate (already gated in tx+coa gate).

Recommended go/no-go:
- critical policy violations: zero tolerance,
- major GST fail-rate threshold: configurable and conservative.
- staged rollout implemented via `TX_COA_GST_GUARDRAIL_PROFILE`:
  - `stage1`: soft rollout guardrails,
  - `stage2`: strict no-regression guardrails,
  - `stage3`: improvement-required guardrails.
- profile defaults currently used in continual gate:
  - `stage1`: major_fail<=`0.75`, issue_rate<=`0.85`, issues_total_delta<=`50`,
  - `stage2`: major_fail<=`0.55`, issue_rate<=`0.70`, issues_total_delta<=`0`,
  - `stage3`: major_fail<=`0.35`, issue_rate<=`0.50`, issues_total_delta<=`-20`.

## 6) Implemented in This Lab (Current Step)

Added GST/Tax audit tooling:
- `scripts/gst_tax_audit.py`
- `scripts/19_run_gst_tax_audit.sh`
- tx+coa promotion gate now supports GST regression guardrails via:
  - `scripts/tx_coa_champion_gate.py`
  - `scripts/16_train_tx_coa_continual.sh`
- staged profile state automation:
  - `scripts/21_update_gst_guardrail_stage.py`

This audit reads tx + CoA advisory outputs and reports:
- per-check fail rates,
- issue severity (`critical`/`major`/`minor`),
- by-label issue distribution,
- pass/fail gate summary for synthetic benchmarking.

## 7) Human Feedback Loop Design (Tax-Aware)

For accountant review captures, ensure fields include:
1. accepted/rejected suggested tax accounts,
2. corrected tax-side mapping (debit vs credit),
3. override reason code (`ineligible_itc`, `wrong_tax_side`, `rcm_mismatch`, `non_gst_case`, etc.),
4. confidence trust indicator (did explanation help?).

Use this feedback to:
- retrain ranking behavior on long-tail tax cases,
- tune thresholds per workflow family,
- improve explanation templates and review routing.

## 8) 30/60/90 GST Hardening Path

Days 1-30:
1. run `19_run_gst_tax_audit.sh` on each eval cycle,
2. baseline GST fail-rates by workflow profile,
3. add synthetic cohorts for missing cases.

Days 31-60:
1. add tax-specific features to training data pipeline,
2. tune policy reranking with accountant-reviewed corrections,
3. tighten GST gate thresholds.

Days 61-90:
1. segment thresholds by company/workflow risk,
2. enable guarded auto-draft only for low-risk GST cohorts,
3. publish tax-control evidence pack for release readiness.
