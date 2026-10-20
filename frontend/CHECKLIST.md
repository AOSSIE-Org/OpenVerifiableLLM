# Contributor brief checklist

| Requirement | Delivered evidence / remaining input |
| --- | --- |
| Overview, explorer/detail, releases, verification and inference view | All six implemented in src/pages.tsx; all adapt to mobile. |
| Explain project and inspect source-to-release process | Overview includes purpose, seven declared stages, source references and separate workflow disclosure. |
| No fake released model | Two unavailable role cards; downloads disabled; production inference always unavailable. |
| Explicit fixtures and no success fallback | Persistent fixture disclosure; schema/mode validation; public mode rejects synthetic evidence even with approval fields; loader fails closed. |
| Named, scoped results | Check profile/scope/result/explanation/operator/attestation and evidence IDs; publisher/project/consumer/independent attribution separated. |
| Empty checks never overall success | Tested empty snapshot; no overall verification verdict exists. |
| Pilot versus production | Scope filters and record scope; synthetic pilot pass retains production NOT_RUN. |
| Missing/invalid/unsupported evidence | Invalid/schema/HTTP states; missing detail routes and supporting reference warnings; UNAVAILABLE/UNSUPPORTED check results. |
| Digest copy and accessible feedback | Full digest; Clipboard API result checked; failure fallback to manual copying. |
| Parent/child/superseded report navigation | Stable hash routes; missing references visible; original failed reports preserved. |
| Original evidence references | HTTPS-only links and raw supplied revisions/paths; null values stay unknown. Default synthetic records do not invent original artifact URLs. |
| Approved real metadata traceability | Schema and configuration ready. **Blocked on approved pinned snapshot, current goal documents and adapter mapping review.** |
| Five verification profiles | Artifact/identity, data reconstruction, sampled replay, full replay and inference reproduction remain distinct. |
| Tested commands only | No invented commands. Future commands require approval/source/validation/resource metadata; copying only in public mode. |
| Mock inference isolated and labelled | Development-only lazy import; release/settings checks; success/error/timeout/empty/cancel/retry/history/copy behavior. Production bundle scan and browser tests. |
| Future available releases | Development fixture with exact identity/inventory/runtime/parent/report rendering; immutable file-URL builder; synthetic downloads disabled. |
| Accessibility | Labels, focus, skip link, semantic tables/details, keyboard menu, announced data/copy states, mobile wrapping, reduced motion. Actual results and limits in validation report. |
| No untrusted raw HTML | React text rendering; safe protocols; escaped and bounded JSON. |
| No large automatic fetches or private data collection | Only small same-origin metadata; bounded loader; self-hosted fonts; no analytics/account/prompt persistence. |
| Reproducible install/build | Exact versions, lockfile and documented Node floor; verified scripts in validation report. |
| Static-safe base-path routes | Hash routes and relative assets; browser suite uses a non-root static prefix without SPA fallback. |
| Training and trust policy untouched | Entire deliverable confined to frontend/. No live resource or repository action. |
| Handoff | README, tests, screenshots, validation report, proposed PR text and exact remaining inputs. |

This checklist demonstrates the frontend assignment's fixture-backed scope. It does not mark the overall model project complete. Actual project status and inference remain deferred to the maintainer.
