# Claude Science toolchain — every connector, skill & agent wired into this repo

This is the map of **how Claude Science is used** in the Regulatory Variant Interpreter —
the "show us how Claude Science got you there" record for the Research track. Each tool is
tied to a concrete repo task, not listed for its own sake. Irrelevant connectors
(Kite/finance, Gmail, Calendar, Canva, Drive) are deliberately **not** wired in.

Read-only science connectors are pre-approved in [`.claude/settings.local.json`](../.claude/settings.local.json);
the three project agents live in [`.claude/agents/`](../.claude/agents/).

---

## 1. Project agents (`.claude/agents/`)

Repeatable, scoped sub-agents. Each is read-only and returns a compact table so the main
thread spends few tokens. Invoke via the Agent tool (`subagent_type: <name>`).

| Agent | Backing connector | Repo task |
|---|---|---|
| **`literature-grounder`** | PubMed | Verify claims in `docs/`, pull correct citations for `src/evidence.py`, prior-art scan for `docs/04_prior_art.md`. Never invents a citation. |
| **`checkpoint-scout`** | Hugging Face Hub | Verify a backbone's config (`d_model`, `rcps`, `bidirectional`, license) before wiring into `src/predictor.py` / `train/finetune_hyenadna.py`. Flags the RC-equivariant Caduceus `-ps` (wrong for orientation-specific MPRA, per `03_results.md §6`). |
| **`synapse-data-scout`** | Synapse.org | Inspect the raw Deng deposit `syn51090452` metadata + DUA/license terms; keep the processed-supplement-is-source-of-truth invariant. Never downloads DUA-restricted data. |

---

## 2. MCP connectors → pipeline stage

| Connector | Used for | Maps to |
|---|---|---|
| **PubMed** | Literature grounding, citation verification, prior art | `docs/04_prior_art.md`, `src/evidence.py` (ClinVar/GTEx/MPRA source refs), all doc claims |
| **Synapse.org** | Raw-data provenance + licensing (`syn51090452`, PsychENCODE, DUA) | `docs/01_data_provenance.md`, `data/load_deng.py` |
| **Hugging Face Hub** | Backbone + dataset discovery & config/license verification | `src/predictor.py`, `train/finetune_hyenadna.py`, `docs/03_results.md §2b/§8` (backbone A/B, Caduceus) |
| **Mermaid Chart** | Render the pipeline diagram (validate before embedding) | `CLAUDE.md §4` pipeline, README |

---

## 3. Skills

| Skill | Repo task |
|---|---|
| **`deep-research`** | Multi-source, fact-checked prior-art / method scans feeding `docs/04_prior_art.md` and the decision log. |
| **`dataviz`** | Calibration reliability diagram (`03_results.md §4`), variant-effect scatter, backbone A/B bars — any results figure. |
| **`scientific-problem-selection`** | Framing / de-risking the trust-first thesis and the "why not just AlphaGenome" positioning. |
| **`code-review` / `verify`** | Guard the leakage-safe split invariant and the `interpret_variant` contract before commits. |

---

## 4. What is intentionally NOT wired in

Per the Research track (not the Product track), these are out of scope and excluded to keep
the repo focused: in-product Claude agent, custom MCP **server**, BioRender figure polish,
Slides/Canva decks, and all finance/mail/calendar connectors. Listed here so the exclusion
is a **decision**, not an omission.

---

## 5. Reproduce the tool usage

Every agent is read-only and deterministic in intent. Example calls:

```
# verify a citation before it ships in a doc
Agent(subagent_type="literature-grounder",
      prompt="Verify: Deng et al. 2024 Science adh0559 reports 164 emVars at 10% FDR. PMID + quote.")

# confirm a Caduceus checkpoint config before a GPU run
Agent(subagent_type="checkpoint-scout",
      prompt="Confirm kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16: rcps, bidirectional, params, license.")

# inspect raw Deng deposit licensing
Agent(subagent_type="synapse-data-scout",
      prompt="syn51090452: license / DUA terms, entity type, file inventory. Does it gate redistribution?")
```
