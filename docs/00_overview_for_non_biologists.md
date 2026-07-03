# The whole subject, from first principles (for non-biologists)

This document explains **what this project is and why it matters** without assuming any
biology. If you can read a recipe and understand a volume knob, you can follow all of
it. Read top to bottom once; after that the code and the other docs will make sense.

> **The 30-second version.** Your DNA is a 3-billion-letter instruction book. A small
> part spells out *recipes* for proteins; a much larger part is *control notes* that
> decide **how loudly** each recipe is used, in which cell, at which time. A one-letter
> typo in a recipe is well understood by existing tools. A one-letter typo in the
> *control notes* — a "regulatory variant" — is usually a shrug: labs can see the typo
> but can't say whether it matters. This project reads those control notes with an AI
> model, predicts what the typo does to the "volume," explains the mechanism, and — the
> important part — **cross-checks the prediction against independent evidence and reports
> an honest, calibrated confidence** so a clinical curator can actually act on it.

---

## Part 1 — The biology, built up from zero

### 1.1 DNA is text with a 4-letter alphabet
Every cell carries the same long string of letters drawn from `{A, C, G, T}` (the four
"bases", or "nucleotides"). The human string is about **3.2 billion letters** long. Think
of it as one enormous book — the **genome** — copied into essentially every cell of your
body.

### 1.2 Genes are recipes for proteins
Scattered through the book are **genes**: passages that spell out how to build a
**protein**. Proteins are the molecular machines and building blocks that do most of the
work in a cell (enzymes, structural parts, signaling molecules). Genes are only about
**1–2%** of the book.

### 1.3 The surprise: most of the book is *control notes*, not recipes
For decades the other ~98% was dismissed as "junk." It isn't. Much of it is
**regulatory DNA** — the stage directions and volume knobs that decide **which recipes
are used, where, when, and how strongly**. A liver cell and a neuron carry the *same
recipes*; they differ because they read a *different set of control notes*. Regulation is
what makes a cell a particular kind of cell.

Key pieces of regulatory vocabulary you'll meet in this repo:

| Term | Plain meaning |
|---|---|
| **Gene expression** | How many copies of a gene's product get made — its "volume." |
| **Promoter** | The "play button" right at the start of a gene. |
| **TSS** (transcription start site) | The exact letter where reading of the gene begins. |
| **Enhancer** | A stretch of control DNA that **turns the volume up** on a gene — often far away in the book, but folded close in 3D. |
| **Transcription factor (TF)** | A protein that acts as a *reader*: it docks onto a specific short DNA "word" and nudges the volume up or down. |
| **Motif** | The short DNA "word" a TF recognizes (e.g. a ~6–12 letter pattern). "This sequence contains a CTCF motif" = "a CTCF protein can dock here." |

> **Analogy.** The genome is a giant cookbook shared by a huge restaurant chain. Recipes
> (genes) are the same in every location. What makes the Paris branch different from the
> Tokyo branch is the *margin notes* (regulatory DNA): "double this sauce here," "skip
> the dessert station on weekdays." Transcription factors are the managers who read those
> notes; motifs are the specific phrases they look for.

### 1.4 How "volume" is physically measured: RNA
To use a gene, the cell first makes a working copy of it in a related alphabet called
**RNA** (a **transcript**). More usage → more RNA copies. So if you want to know how
loudly a gene (or a piece of regulatory DNA driving a gene) is playing, you **count RNA
copies**. This single idea — *activity ≈ how much RNA is produced* — is the backbone of
the experiment that generates our training data (Part 3).

---

## Part 2 — The clinical problem this project attacks

### 2.1 A "variant" is a typo
Your genome differs from the reference human genome, and from your neighbor's, at
millions of positions. Each difference is a **variant**. The simplest and most common is
a single-letter swap — a **SNV** (single-nucleotide variant): the reference says `A`
here, you have `G`. We write one as `(chrom, pos, ref, alt)`: *on this chromosome, at this
position, reference base → alternate base.* That 4-tuple is exactly the input to our
tool.

### 2.2 Coding vs. regulatory variants — and why one is "solved" and one isn't
- A typo **inside a recipe** (coding variant) may change an ingredient of the protein.
  We have decades of tools (SIFT, PolyPhen, AlphaMissense, …) that predict the impact
  reasonably well, because "which amino acid changed" is a well-defined question.
- A typo **inside the control notes** (regulatory variant) doesn't change any protein's
  ingredients. It changes *how loud a gene plays* — maybe only in one cell type, at one
  developmental moment. That is much harder to reason about, and mainstream tools mostly
  **don't even try**.

### 2.3 VUS: the shrug that this project is built to end
When a lab finds a variant in a patient but **cannot say whether it causes disease**,
they classify it a **VUS — Variant of Uncertain Significance**. VUS are a massive, growing
problem in genomic medicine: a real change is sitting in the report and nobody can act on
it. Regulatory variants are *disproportionately* VUS, precisely because we lack good
predictors for them.

> **This project's target user** is the person staring at that VUS: a variant-curation
> scientist or clinical geneticist who needs a *principled, defensible* call — not just a
> number, but a number they can trust and audit.

---

## Part 3 — How we make a prediction

### 3.1 MPRA: measuring regulatory "volume" for 100,000 sequences at once
We can't train a model without examples of "sequence → how loud it is." The experiment
that produces those examples at scale is a **Massively Parallel Reporter Assay (MPRA)**.
Mechanically:

1. **Synthesize** a library of ~100,000 candidate regulatory snippets (each ~200 letters
   long), each attached to a **reporter gene** and a unique **barcode**.
2. **Put them into living cells** (here: developing human brain cells).
3. **Count DNA** (how many copies of each snippet you put in) and **count RNA** (how
   loudly each snippet drove the reporter). The ratio **RNA / DNA** is that snippet's
   **activity** — its measured "volume." (In practice it's a log-ratio, averaged over
   barcodes and replicates.)

> **Analogy.** You print 100,000 different billboards, each with a hidden serial number,
> and put them all up around a city. Later you count how many people mention each serial
> number (RNA) versus how many copies of that billboard you posted (DNA). The ratio tells
> you which billboards actually grabbed attention. MPRA does this for DNA control
> sequences, all in parallel.

**Testing a variant with MPRA.** Include *both* alleles as separate snippets — one with
`ref`, one with `alt`, identical except for that single letter. If the two drive
measurably different volume, the variant is an **emVar** ("expression-modulating
variant"). Those measured ref-vs-alt differences are **experimental ground truth** for
what a single-base change does — which is exactly what our tool tries to predict. This is
why they are gold for *calibration* (Part 4).

### 3.2 Our specific dataset (Deng et al. 2024)
We use a published cortical MPRA (Deng et al., *Science* 2024) that measured activity for
**102,767** sequences in developing human brain cells and organoids, and tested
**17,069** psychiatric-disorder-associated variants, of which **164** significantly
changed activity. Details, formats, and access rules are in
[`01_data_provenance.md`](01_data_provenance.md). Two facts matter for intuition:
- The sequences are **short (~200 letters)** — trivially small for a modern DNA model.
- There are **two biological contexts** (primary cortex tissue and lab-grown "organoids").
  We exploit that: see the two-model idea in Part 4.

### 3.3 A "DNA language model" — like an LLM, but for genome text
An LLM learns the statistics of human language by reading mountains of text. A **DNA
language model** does the same for genome text: trained on lots of DNA, it learns the
"grammar" of regulatory sequence — which letter patterns tend to co-occur, where motifs
sit, what looks like a promoter. We don't train one from scratch (that takes months); we
take a pre-trained one and adapt it.

**Why HyenaDNA specifically?** Our whole method hinges on changing **one letter** and
seeing what happens. Some DNA models chop sequence into chunks of several letters
("tokens") for speed; a one-letter change can get blurred inside a chunk. **HyenaDNA
reads one letter at a time (single-nucleotide resolution)**, so a single-base swap is
crisp and visible. It's also fast and small enough to adapt on a single rented GPU in a
week. (Full reasoning: [`02_decision_log.md`](02_decision_log.md), decision D2.)

### 3.4 Fine-tuning: teaching the general reader our specific job
**Fine-tuning** = taking the pre-trained, general-purpose DNA reader and training it a bit
more on *our* labeled examples (MPRA sequence → measured activity) so it specializes in
*predicting regulatory volume*. Output of this step: a model that, given any ~200-letter
sequence, predicts its activity.

### 3.5 In-silico saturation mutagenesis: the core trick
Now the payoff. "**In-silico**" = "in the computer." "**Saturation mutagenesis**" = "try
*every* mutation." Given a sequence, we ask the fine-tuned model:

> *If I change position 1 to each of the other three letters, how does predicted volume
> change? Position 2? … all the way to position 200?*

That yields a full **sensitivity map**: which letters matter, and in which direction. For
a **real patient variant**, we don't need the whole map — we just read off the model's
predicted change at that one position for that specific `ref → alt` swap. That predicted
change (Δ activity) is our first-pass answer: *does this typo turn the volume up, down, or
not at all, and by how much?*

### 3.6 Motif annotation: explaining *why*
A number ("volume drops 0.8") isn't an explanation. So we scan the reference and
alternate sequences against a **library of known TF motifs (JASPAR)** and ask: does the
typo **destroy** a motif (a TF can no longer dock) or **create** one (a new TF can now
dock)? That converts the number into a mechanism a curator recognizes:
*"the alt allele disrupts a predicted CTCF binding site."* (The Deng paper itself found
specific TF families driving activity, so this mechanism layer is grounded in the same
biology.)

---

## Part 4 — The hard part: making the prediction *trustworthy*

A model that outputs a confident number is easy. A model a clinician should **act on** is
not. Models are wrong in weird ways, and "the neural net said 0.8" is not something you
put in a medical report. **The real contribution of this project is the trust layer.**

### 4.1 Ground the prediction in *independent* evidence
For each variant we gather signals that were produced **independently** of our model, and
ask whether they agree with it:

| Evidence source | What it independently tells us |
|---|---|
| **Held-out MPRA measurement** | Did the actual wet-lab experiment measure this variant's effect? (Our ground truth — see 4.3.) |
| **GTEx eQTL** | In large human populations, is this variant statistically associated with changes in a gene's expression in real tissues? (An **eQTL** = a variant linked to expression level.) |
| **ClinVar** | Has anyone clinically classified this variant (benign / pathogenic)? |
| **TSS proximity** | Is it close to a gene's start, where regulatory effects are more plausible? |
| **A frozen foundation model** (Enformer / AlphaGenome) | A completely different, much larger model's independent opinion on the variant's regulatory effect. We use it **zero-shot** (never trained by us) precisely so it's an *outside* voice. |
| **Our organoid model** | A **second** fine-tuned model trained on a *different biological context* (lab-grown organoids vs. primary tissue). If two independently trained context-models agree, that's real corroboration; if they diverge, we flag it. |

> **Why two of our own models?** We deliberately train the primary-cortex model and the
> organoid model **separately** (not as one shared network). Kept separate, their
> agreement is genuine evidence; fused into one network, it would be circular. This is a
> core design decision — see [`02_decision_log.md`](02_decision_log.md), decision D4.

### 4.2 Calibration: making "confidence" mean something
If the tool says "70% confident," that should mean it's right about 70% of the time.
Raw neural-network scores do **not** have this property out of the box. **Calibration** is
the statistical step that maps raw scores to honest probabilities, learned by comparing
predictions against outcomes we already know.

### 4.3 …and the ground truth we calibrate against — without cheating
Our calibration outcomes are the **17,069 variants the MPRA actually measured**. We tune
confidence so that, on those variants, the tool's stated confidence matches its real
accuracy.

**The cardinal rule (leakage):** we must **never train the model on a sequence we later
use to judge it.** A variant's reference sequence *is* one of the library snippets (with
one letter swapped), so this is a real trap. We prevent it by splitting the data by
**genomic locus** (region): an entire region goes *either* to training *or* to
calibration, never both — including neighboring overlapping snippets. This is enforced in
code and **asserted** at the end of data prep (`data/splits.py`,
`data/prepare_data.py`). If the check ever fails, data prep aborts. (Decision D5.)

### 4.4 Agreement vs. conflict — and *surfacing* conflict
The trust layer combines the model's prediction with the evidence above:
- **They agree** (model says "disruptive," GTEx + ClinVar + the frozen model concur) →
  **high** calibrated confidence.
- **They conflict** (model says "big effect," but population data and clinical databases
  say "benign") → the tool **reports the conflict openly** and lowers confidence, instead
  of averaging it into a single misleadingly-confident number.

Honest conflict is a *feature*. A curator learns far more from "the model and the
population data disagree, here's how" than from a smooth fake-confident score.

### 4.5 The evidence chain: an auditable receipt
Every call ships with an **evidence chain**: the ordered list of every input that fed the
decision (each model number, each database hit, each motif change) plus **provenance**
(which model checkpoint, which data versions). A curator — or an auditor, or a regulator —
can retrace exactly how the tool reached its call. Nothing is a black box.

---

## Part 5 — Putting it together: one call, end to end

`interpret_variant("chr2", 162279995, "A", "G")` conceptually does:

1. **Build** the ~200-letter reference window around the position; make the alt copy with
   the single letter changed.
2. **Predict** activity for both alleles with the fine-tuned primary-cortex model;
   Δ = alt − ref. (Also run the organoid model.)
3. **Explain**: scan both alleles for motif gain/loss → candidate mechanism.
4. **Ground**: look up GTEx eQTL, ClinVar, TSS distance, held-out MPRA (if present), and
   the frozen foundation model's score.
5. **Trust**: check agreement, compute a **calibrated** confidence, list agreements and
   **conflicts**, assemble the **evidence chain**.
6. **Return** an `Interpretation` object (defined in [`../src/schema.py`](../src/schema.py))
   carrying the call, the confidence, the mechanism, the evidence, and the provenance.

That object — not a bare number — is the product.

---

## Glossary (quick reference)

- **Base / nucleotide** — one DNA letter: A, C, G, or T.
- **Genome** — the full ~3.2-billion-letter DNA string in a cell.
- **Gene** — a passage that codes for a protein (~1–2% of the genome).
- **Protein** — the molecular machine a gene builds.
- **Regulatory DNA** — non-coding sequence that controls *how much* genes are expressed.
- **Gene expression** — how much of a gene's product is made ("volume").
- **Promoter / TSS** — the start region / exact start letter of a gene.
- **Enhancer** — regulatory DNA that boosts a gene's expression.
- **Transcription factor (TF)** — a protein that binds a specific DNA motif to tune expression.
- **Motif / PWM** — the short DNA pattern a TF recognizes; a PWM is its probabilistic spelling.
- **RNA / transcript** — the working copy of a gene; more usage → more RNA.
- **Variant / SNV** — a difference from the reference genome; SNV = single-letter swap.
- **ref / alt** — the reference base vs. the alternate (variant) base.
- **VUS** — Variant of Uncertain Significance: a variant we can't yet classify.
- **MPRA** — the parallel experiment measuring regulatory activity of many sequences.
- **Activity** — measured regulatory volume, ≈ log(RNA / DNA).
- **emVar** — a variant whose two alleles drive measurably different activity.
- **eQTL** — a variant statistically linked to a gene's expression level (e.g. from GTEx).
- **DNA language model** — a model trained on genome text (here: HyenaDNA).
- **Fine-tuning** — adapting a pre-trained model to our specific task.
- **Saturation mutagenesis (in silico)** — computationally scoring the effect of every
  possible single-base change.
- **Calibration** — making stated confidence match real-world accuracy.
- **Leakage** — accidentally training on data you later test/calibrate on (forbidden here).
- **Evidence chain / provenance** — the auditable record behind each call.
