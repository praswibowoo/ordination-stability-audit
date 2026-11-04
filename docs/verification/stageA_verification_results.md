# Stage-A Verification — Results & Adjudication (D54, 2026-10-01)

## Outcome: ACCEPTED WITH CORRECTIONS

| metric | value |
|---|---|
| Papers verified | 39/39 filled (complete) |
| Claims checked | 117 |
| Claims OK | 104 (88.9%) — below 95% gate |
| Scenario-sets OK | 38/39 |
| Overall rows OK/ISSUE | 33 OK / 6 ISSUE |

## Corrections applied (13 cells — all over-marked Y, verified absent)
B034 pH · B065 NO2+Chl-a · B066 DO×2 · B069 pH+Salinity · B092 Secchi · B137 Secchi · B182 Secchi+PO4+Salinity · B166 TSS

- All 13 set to blank (NR) + correction note in cause_effect_note
- Directionality: 100% over-marking (agents marked Y where params absent) — no under-marking found
- Post-correction in-sample accuracy: 117/117 (100% by construction)

## Adjudication notes
- **Error pattern 1 — synonym/figure confusion**: Secchi (4×): papers report "brightness"/figure-only data → reviewers 判定 absent. Watch list widened for Secchi.
- **Error pattern 2 — parameter truly absent** (B065/66/69/166): local over-marking.
- **B054**: scenarios_N resolved = 2 stations intentionally excluded (non-WQ biology only) — extraction unchanged, awaiting user confirm.
- **Global suspect scan**: **246-cell watch list** (zero-synonym scan suspects — advisory, distinct from the 13 user-verified corrections; includes figure-only false positives) for Stage-B (`suspect_ycells_watchlist_stageB.csv`); includes scan artifacts (figure-only reports legitimately absent from text) — advisory, not errors.

## Interpretation & acceptance rationale
Fails were **localized, unidirectional (over-marking), and corrected**; the systematic error modes are documented; no chunk-level re-extraction required. Stage-A is accepted for Phase 3 with the watch list carried forward.
Gate math: 88.9% original → targeted correction path (protocol D49 provision) → acceptance documented.

## User acceptance (session-2, 2026-10-01)
D54 adjudication **ACCEPTED** by user (recorded as D57 in kickoff_notes.md): the 13 verified corrections + post-correction gate stand; Stage-A confirmed accepted for Phase 3 launch. B054 scenario-count note remains open pending user confirm.
