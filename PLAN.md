# PLAN.md — Hackathon execution plan (Claude Science, July 2026)

> Companion to [`CLAUDE.md`](CLAUDE.md) (the technical charter). This file tracks
> **schedule, submission requirements, and scoring strategy** — the "ship it" layer.
> CLAUDE.md says *what* we build and *why*; PLAN.md says *by when* and *how we win*.
> **Local tracking file — not for pushing unless we decide to.**

---

## 0. The two anchors that govern everything

| Event | Time (IST / Pune) | Notes |
|---|---|---|
| **Hacking begins** | **10:00 PM IST, Tue Jul 7** | Kickoff 9:30 PM IST. Discord team formation right after. |
| **Submission deadline** | **6:30 AM IST, Tue Jul 14** | = Mon Jul 13, 9:00 PM ET. **In practice: finish & submit Monday night IST.** Do NOT plan to work at 6 AM. |

ET in July = EDT → **Pune is +9:30 ahead** of the schedule's native ET.

**Real working days = Jul 8–13 (six days), not a leisurely seven.**

---

## 1. Daily schedule (IST)

| Date | Event / Focus |
|---|---|
| **Tue Jul 7** | 9:30 PM kickoff · 10:00 PM hacking begins + Discord team formation |
| **Wed Jul 8** | 🗓️ 9:30–10:30 PM **Live Session 1** — Claude Science overview (Alexander Tarashansky, Anthropic). **Build focus: DATA.** |
| **Thu Jul 9** | Build focus: **fine-tune** (Colab A100). |
| **Fri Jul 10** | 🗓️ 9:30–10:30 PM **Live Session 2** — Gladstone talk, "genome to inference without touching a pipette," virtual PPI screening (Sukrit Silas, Gladstone). **Judge-adjacent, on our exact thesis — attend.** Build focus: **working end-to-end slice.** |
| **Sat Jul 11** | All-day hacking. Focus: **trust layer.** |
| **Sun Jul 12** | All-day hacking. Focus: **calibration proof.** **FEATURE FREEZE tonight.** Draft the 100–200 word summary. |
| **Mon Jul 13** | **SHIP DAY.** Polish + record 3-min demo video + write summary + submit. **No pipeline code at 3 AM.** |
| Wed Jul 16 | 9:30 PM top-6 + final judging · 11:00 PM closing ceremony (top 3). |

**Office hours** are 2:30 AM IST (5–6 PM ET, Tue–Fri) — impractical live. Post async in Discord **#office-hours**. Don't lose sleep.

Both live sessions land at 9:30 PM IST — **attend both.**

---

## 2. Judging criteria → what we do about it

| Criterion | Weight | Our move |
|---|---:|---|
| **Demo** | **30%** | Stage 1 is **async off a 3-min video** — judges score the *video*, not the live tool. Monday is entirely polish + record. Feature-freeze Sun so Mon is calm. |
| **Claude Use** | **25%** | **Put Claude *inside* the product**, not just the build. A Claude agent reads the evidence chain (model Δ + frozen Enformer + eQTL + ClinVar + motif) and writes the curator-facing interpretation — narrating why confidence is high/low and honestly flagging conflicts. **Stretch:** expose the interpreter as an **MCP server** callable from any Claude client (leverages MCP cert). |
| **Depth & Execution** | **20%** | "Did the team push past their first idea?" → our story: moved off the generic "chat with your data" clone to the **trust-calibration** angle. Tell that story explicitly in the video + summary. |
| Remaining | 25% | Novelty / impact / usefulness — covered by the named-user framing (deployable interpreter that outlives the week). |

---

## 3. Build calendar (feature map)

```
Wed Jul 8   DATA          real Deng ingest → train/val + held-out calibration set
Thu Jul 9   FINE-TUNE     HyenaDNA on A100 (primary cortex; organoid 2nd model)
Fri Jul 10  E2E SLICE     interpret_variant() runs end to end on a real variant
Sat Jul 11  TRUST LAYER   grounding + agreement/conflict aggregation
Sun Jul 12  CALIBRATION   isotonic calibration proof + Claude-in-product agent  ── FREEZE
Mon Jul 13  SHIP          polish · 3-min video · 100–200w summary · submit
```

---

## 4. Submission checklist (required — eligibility gating)

- [ ] **LICENSE file** — MIT or Apache 2.0 (approved open-source license). **Non-negotiable.** *(TODO: add now.)*
- [ ] **3-min demo video** — YouTube or Loom. Record Monday.
- [ ] **Open-source repo** — public.
- [ ] **Written summary** — 100–200 words. **Draft Sunday, not Monday.**
- [ ] **Submit via the CV / judging platform** — not just a GitHub push.

### Rules compliance
- [ ] **New Work Only** — all code written Jul 7–13. ✅ Charter + plan are design docs (allowed). Medchat model is portfolio reference only, **not** submission material. **Do not import pre-existing private code.**
- [x] **Deng et al. licensing** — CC-BY-NC → fine for non-commercial hackathon. **Do NOT commit raw data** (`.gitignore` already excludes `data/raw/` — good; prep script downloads it). ✅ Deng + HyenaDNA + Enformer attributed in README (License + Acknowledgements sections).
- [x] **Checkpoint licenses** — ✅ verified 2026-07-07 via HF: HyenaDNA (`LongSafari/hyenadna-tiny-1k-seqlen` + `-hf`) = **BSD-3-Clause**; Enformer (`EleutherAI/enformer-official-rough`) = **CC-BY-4.0**. Both attribution-only, no NC/copyleft → clean under Apache 2.0. **Attribute both in README.** ⚠️ If swapping to AlphaGenome, its Google license is NC/API-restricted — re-check before use.
- [ ] **No assets without rights** — general hygiene sweep before submit.

---

## 5. Open TODOs (surfaced, not yet done)

1. Add `LICENSE` (MIT or Apache 2.0).
2. ~~Add attributions to README~~ ✅ done — Deng (CC-BY-NC) + HyenaDNA (BSD-3) + Enformer (CC-BY-4.0) in README License + Acknowledgements.
3. ~~Confirm HyenaDNA + Enformer checkpoint licenses.~~ ✅ done — BSD-3 / CC-BY-4.0, both clean.
4. Build the **Claude-in-product** interpretation agent (evidence chain → curator narrative).
5. Stretch: **MCP server** wrapper around `interpret_variant()`.
6. ~~Locate exact Deng et al. accession + format~~ ✅ done — PMID 38781390 / `adh0559` / PMC12085231; Synapse `syn51090452`. Provenance §4–§8a updated.
7. ~~Prior-art / novelty scan~~ ✅ done — [`docs/04_prior_art.md`](docs/04_prior_art.md). Whitespace confirmed; "why not AlphaGenome" answer pre-loaded for judges.

---

## 6. Guardrails (do-not-violate)

- **Feature-freeze Sunday night.** Monday is ship-only.
- **Video is 30% and judged async.** A great video off a modest tool beats a great tool with no video.
- **Protect Monday.** No new pipeline code Monday.
- **Attend both live sessions** (Wed + Fri, 9:30 PM IST). Fri is judge-adjacent.
