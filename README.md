# mpnn_detect

Screening for de novo and inverse-folding designed proteins (ProteinMPNN,
LigandMPNN, ESM-IF and relatives).

Replaces a two-HMM classifier that could not work. No system dependencies,
offline-capable, Colab-ready.

**Measured against real ProteinMPNN output**, not simulated, at the intended
configuration: 3,544-sequence PF00069 family, full DCA at Meff/M = 3.14,
α = 0.01, no fallbacks. Two numbers, because they answer different questions:

| | designs | detected | false positives |
|---|---|---|---|
| **In-family backbones only** (cdk2, cdk8, p38) | 540 | **97.0%** | 0.0% of 44 |
| All six backbones, 8 repeated splits | 1,080 | 93.3% [88.7–98.0] | 0.71% |

The first removes the confound and has no interval; the second has an interval
and includes three tyrosine-kinase backbones at 34–45% identity to this
Ser/Thr-dominated family, whose detection is partly distance rather than
design. They bracket each other, which is about as much as three proteins
support. `backbone_membership()` catches that confound before you measure.

At high sampling temperature (T ≥ 0.5) detection falls to 60–70%. Three
proteins is the honest sample size. See
[Real ProteinMPNN output](#real-proteinmpnn-output) and
[the control section](#the-control-earned-its-place-half-the-backbones-were-out-of-family);
everything above those is still an upper bound from simulated designs.

### Fixed in 5.3.3

The 5.3.2 checks all behaved on the next run — membership flagged the three
tyrosine kinases, the native control reported 1 of 6 rather than 2 of 12, and
the restricted rate came out at 97.0%. That run also produced the project's
second simulated-vs-real comparison, which contradicted the first on the
covariation layer and so retired a claim this file had been making. Details
under [what simulation actually flattered](#what-simulation-actually-flattered--measured-twice-because-once-was-wrong).

### Fixed in 5.3.2

A reported run at the full intended configuration — 3,544 sequences, DCA at
Meff/M = 3.14, α = 0.01, no fallbacks — came back at 98.5% detection with the
native-backbone control flagging 2 of 12. Three of the six design backbones
turned out to be tyrosine kinases at 34–39% identity to a family of small
Ser/Thr kinases. `md.backbone_membership()` now checks this before measuring,
against the family's own identity distribution, and the notebook reports the
in-family rate as the result. Also: the native control was counting each native
once per output folder ("2 of 12" meant 1 of 6), and section 14 inherited an α
its own null could not place.

### Fixed in 5.3.1

A reported Colab run came back with a 433-sequence family where 3,544 were
available, which forced mutual information instead of DCA and a 96-sequence
null that could not place α = 0.01. Cause: the guard added in 5.3.0 for the
new local-FASTA option tested `SOURCE`, which the UniProt path also sets, so
the reviewed → unreviewed top-up broke on its second pass. That is the third
bug this acquisition logic has shipped from a notebook cell no test could
reach, so it is now `md.assemble_family()` with a pluggable fetcher and ten
tests. Also in this release: section 14 adopts the α its own (smaller,
domain-trimmed) null can place instead of inheriting one it cannot; and
dereplication is 6× faster after the 5.3.0 length-bias fix removed its early
exit — 115 s → 19 s on 11,993 kinases, provably the same output.

```
pip install pyhmmer numpy
```

**Colab:** open `designed_protein_detector_colab.ipynb` → Runtime → Run all.
Nothing to upload; the module is embedded and written to disk by the first cell.
`THOROUGH = True` (default) takes 20–45 min on a free CPU runtime; `False` gives
a ~5 min pass. Results, models and figures are written to
`MyDrive/mpnn_detect/runs/<RUN_NAME>/`, and re-running the same `RUN_NAME`
reloads the saved profile instead of rebuilding it.

---

## Fixed in this version: the retrieval failure

A reported run showed `retrieved 14, 14 unusable` for a Pfam query that should
return thousands, then silently fell back to a simulated family.

**Cause: the response body was gzipped twice.** UniProt compressed the payload
and the transport layer compressed it again. One `gzip.decompress` left a body
that was *still* gzip, and those compressed bytes were decoded as text. The
mojibake parsed into a handful of junk records, all of which failed cleaning.
The reported first bytes — `\x8b\x08\x00\x00\x00\x00\x00\x00` — are a gzip
header whose leading `\x1f` had been stripped by `str.lstrip()` (`'\x1f'` is
whitespace to Python), which is exactly the signature of this double encoding.

Three fixes, because three separate layers let it through:

| Layer | Was | Now |
|---|---|---|
| Decoding | one `gzip.decompress` | `_decompress_fully` loops (bounded) over gzip and zlib |
| Validation | network bodies only | `_implausible_records` screens length, alphabet and count, on fetch **and** on cache read |
| Caching | key `sha256(query)`; garbage persisted on Drive forever | key versioned to `v2`, poisoned entries rejected, renamed `.rejected`, re-fetched |

The cache fix matters most: the bad response had been written to Drive, so
without versioning the decode fix would never have fired for anyone who had
already cached one.

---

## Measured on real data

The first run against real UniProt sequences (433 dereplicated PF00069 kinases,
8 independent splits) changed the picture substantially. The simulated numbers
further down are upper bounds, and these are what the method actually does.

| | simulated family | real PF00069 kinases |
|---|---|---|
| Overall AUROC | 1.0000 [1.000, 1.000] | 0.9946 [0.984, 1.000] |
| Designs detected | 100% | **97.9%** [95.0, 100] |
| Signals carrying evidence | 17 of 21 | **14 of 21** |
| Couplings | DCA (Meff/M = 3.6) | **MI** — DCA declined, Meff/M = 0.56 |

Three findings came out of it, each with a fix.

**1. The model was twice as long as the domain.** Full-length proteins aligned
into a 1,301-column alignment, and sparsely occupied flanking columns became
match states: a **459-position model for a ~260-position domain**. Only 272
columns were occupied by half the members. Occupancy trimming (`symfrac=0.6`,
HMMER's `--symfrac`) is now the default, and the structural and phylogenetic
signals roughly recover:

| signal | untrimmed | trimmed |
|---|---|---|
| `exposed_llr` | 0.662 | **0.922** |
| `ancestral_implausibility` | 0.900 | **0.965** |
| `tree_violation` | 0.654 | **0.813** |
| `variable_llr` | 0.666 | **0.807** |
| `pendant_excess` | 0.586 | **0.750** |
| `indel_deficit` | 0.459 | 0.680 |
| `rate_incongruence` | 0.773 | 0.436 |

The last row is the honest one: `rate_incongruence` *lost* power under
trimming. The padded flanking columns had extreme apparent site rates and the
simulated designs differed there systematically, so its untrimmed 0.773 was an
artefact of the padding.

**2. The operating point was unreachable.** With 433 sequences the null held 47,
so the smallest p-value the empirical null can produce is 1/48 = 0.021. Running
at α = 0.01 forced every call through the tail fit:

| α | null n | empirical floor | designs detected |
|---|---|---|---|
| 0.01 | 47 | 0.021 | 73.5% [53.5, 93.8] |
| 0.05 | 47 | 0.021 | **97.9%** [95.0, 100] |
| 0.05 | 86 | 0.012 | 98.3% [97.1, 99.6] |

Same model, same signals — 24 points of detection lost to an α the data could
not place. `fit_null` now reports `recommended_alpha` and says so, the split
allocation sizes the null from α instead of a fixed fraction, and the notebook
adopts the reachable α rather than running quietly at an impossible one.

**3. Seven signals carry nothing on this family, and one is backwards.**
`stratum_contrast` scored above 0.95 on the simulated family and **0.374** on
real kinases — reliably worse than chance, pulling against the rest.
`coupling_violation` sat at 0.518 because DCA never engaged.

`Detector.apply_validation(report)` now feeds the validation back into the
weights, zeroing signals whose interval spans chance:

| weights | held-out AUROC | detected | FPR |
|---|---|---|---|
| all 21 signals | 0.9956 | 97.9% [94.5, 100] | 4.38% |
| 14 informative only | **0.9990** | **99.8%** [98.9, 100] | 3.96% |

This prunes for *that* family, not in general — a signal dead on kinases may be
the one that fires elsewhere.

**What would make DCA engage on real kinases.** Meff runs about 0.7× the
dereplicated count, and the build split takes a little over half, so a
270-position model needs roughly 1,200+ sequences before splitting. SwissProt
holds only ~700 reviewed kinases in this length window, so the notebook now
tops up with unreviewed entries instead of stopping at the reviewed set.

---

## Second real run: 1,513 kinases

A larger pull (reviewed + unreviewed PF00069, 1,513 after dereplication,
symfrac trimming giving M = 301) produced three more findings.

### A saved detector was being paired with a rebuilt profile

The run reloaded a detector whose null held **93** sequences against a freshly
rebuilt profile that had **333** available. Scores are not comparable across
profiles — a different model length or member set moves the whole
distribution — so every p-value in the single-query, batch, window and library
cells was computed against a null belonging to a different model. The
validation numbers were unaffected, because `cross_validate` builds its own
profiles and detectors.

Caught by arithmetic: the saved `null_info` reported `resolution = 1/93`
exactly, while the current code computes `1/(n+1)`, and the
`recommended_alpha` field was missing entirely — so that file had been written
by an older build and loaded from Drive, not produced by the run.

`Detector` now stores a fingerprint of the profile its null was fitted
against, and `load()` refuses a mismatch outright rather than warning. The
notebook only reuses a saved detector when the profile was also reused.

### DCA was still out of reach, for a reason I had got wrong

My earlier guidance assumed effective sequence counts run about 0.7× the
dereplicated count. On real kinases it is **0.47** — 1,513 sequences gave
Meff = 381 against M = 301, so Meff/M = 1.27 where the full 20-residue Potts
model needs 2.0.

Rather than fall all the way back to mutual information, there is now a middle
tier: **reduced-alphabet DCA** over seven chemical classes. That needs about
eight times fewer parameters, so it fits at Meff/M ≈ 0.7, and it still
conditions away the transitive correlations MI counts — it just cannot tell
which residue within a class. On this family:

| coupling model | pairs | `coupling_violation` AUROC |
|---|---|---|
| mutual information | 389 | 0.555 — chance |
| reduced-alphabet DCA | 365 | **0.796** |

Over 8 splits the signal went from 0.535 (uninformative) to **0.849**, and the
count of signals carrying evidence rose from 14 to 16. The build cell now
reports which tier was used and, when the full model is out of reach, computes
the sequence count *this* family would need from its own redundancy.

### Three signals are reliably inverted on real kinases

| signal | simulated | real kinases |
|---|---|---|
| `rate_incongruence` | 0.84 | **0.292** [0.256, 0.317] |
| `stratum_contrast` | 0.95 | **0.355** [0.277, 0.427] |
| `kmer_repeat` | 0.48 | 0.461 [0.431, 0.499] |

These are not noise — the intervals are tight and the permutation p-values are
1.0. With prior weights they pull against the rest, which is why the aggregate
detection rate sat below what the individual signals suggest. Pruning them
recovers it:

| weights | AUROC | detected | FPR |
|---|---|---|---|
| all 21 prior weights | 0.9649 | 79.7% [60.8, 97.9] | 6.11% |
| prune uninformative | 0.9679 | **86.4%** [72.9, 96.7] | 6.67% |
| prune + flip inverted | 0.9705 | 84.4% [69.6, 97.1] | **4.44%** |

Flipping an inverted signal buys a little AUROC and a lower false-positive rate
but no extra detection, so `flip_inverted` stays off by default: a signal
backwards on one family is more likely mis-specified than sign-flipped, and
flipping bakes that family's quirk into the model.

**Current state on real PF00069** (1,513 sequences, reduced-alphabet DCA,
α = 0.05, validation-pruned weights): AUROC 0.968, detection 86.4%
[72.9, 96.7], false positives 6.67%. Still measured against simulated designs,
so still an upper bound.

---

## Third real run: 3,544 kinases, full DCA

Raising the target to 6,000 pulled 11,993 PF00069 entries (reviewed plus
unreviewed), 3,544 after dereplication, giving Meff = 976 against M = 311 —
**Meff/M = 3.14, enough for the full 20-residue Potts model**. Both fixes from
the previous round behaved as intended: the profile rebuilt, and the saved
detector was refused rather than silently reused.

| | value |
|---|---|
| Overall AUROC | 0.9972 [0.9953, 0.9994] |
| Designs detected | **92.7%** [81.2, 98.3] at α = 0.01 |
| False positives | 1.17% [0.12, 2.10] against a nominal 1% |
| Signals carrying evidence | 15 of 21 |

α = 0.01 is now reachable: the null holds 780 sequences, so the empirical floor
is 1/781 and no tail extrapolation is needed to place the threshold.

The coupling signal tracked the model it was given, across three runs:

| family | coupling model | `coupling_violation` AUROC |
|---|---|---|
| 433 sequences | mutual information | 0.493 — chance |
| 1,513 sequences | mutual information | 0.535 — chance |
| 1,513 sequences | reduced-alphabet DCA | 0.849 |
| 3,544 sequences | **full 20-residue DCA** | **0.888** |

### Adding signals can cost detection

The stress test turned up something that matters more than any single AUROC.
On composition-matched designs the **full detector detected fewer designs than
the evolutionary signals alone**, at a higher AUROC:

| composition matched | evolutionary only | all signals, prior weights |
|---|---|---|
| none | 12% missed | 4% missed |
| 0.4 | 34% missed | 36% missed |
| 0.7 | 34% missed | 54% missed |
| full | **48% missed** | **94% missed** (AUROC 0.972) |

(from the user's own run; the same pattern appears on every family tested)

The ranking was fine; the threshold was not. Every signal contributes variance
to the aggregate score, so one that does not discriminate still widens the
natural null and lifts the α-quantile — while a composition-matched design
gains nothing from it. AUROC does not see this, because AUROC is
threshold-free.

That makes hand-set weights the weak link, and the validation report already
holds the information needed to fix them. `apply_validation(reweight=True)`
scales each weight by its measured separation, `2·(AUROC − ½)` floored at zero:

Measured on the 3,544-kinase family against designs whose composition was 70%
matched, over 8 independent design samples on one fixed profile, null and
evaluation set:

| weights | detected | 90% interval | FPR |
|---|---|---|---|
| hand-set priors | 38.9% | [32.5, 48.2] | 1.00% |
| evolutionary signals only | 60.6% | [57.1, 66.6] | 0.33% |
| **pruned + reweighted** | **60.0%** | [53.0, 67.4] | 0.38% |

Reweighting recovers essentially the whole gap — 21 points of detection — and
on paired samples it **ties** evolutionary-only rather than beating it
(−0.6%, winning 3 of 8). So on composition-matched designs the composition
signals add nothing even when rescaled; what rescaling fixes is the damage they
were doing.

The user's own run shows the same thing across the blend range:

| composition matched | prior weights | reweighted |
|---|---|---|
| none | 8% missed | **2%** |
| 0.5 | 46% missed | **24%** |
| full | 82% missed | **70%** |

**A correction.** An earlier version of this file reported reweighting as
*beating* evolutionary-only (61.3% vs 55.0%). That came from a single split;
over 8 design samples the two are indistinguishable. Single-split differences
of a few percent on 80 designs are a handful of sequences, and this one did not
survive repetition.

**And a refuted hypothesis.** It is tempting to validate against the hardest
designs you expect, so the weights are tuned for that threat. Measured, that is
slightly worse: reweighting from raw generator output detected 60.0% of
composition-matched designs against 56.7% from a report computed on matched
designs, winning 7 of 8 paired samples. The report's job is to *rank* the
signals, and harder designs separate less, so they estimate that ranking less
precisely. Use the cleanest designs you have.

The caveat on all of it: those weights are estimated from the same simulated
designs the report scored, so they are a better use of a measurement already
made rather than independent evidence.

### Two smaller corrections

`construct_motifs` was being reported as "inverted — points the wrong way" at
AUROC 0.498. No kinase in the set carries a purification tag, so the signal is
constant and its AUROC is 0.5 by construction; a hair below half is not an
inversion. Testing the interval edge alone cannot separate that from a real
inversion, because a constant signal has a 0.002-wide interval that "excludes
chance" while a genuinely backwards one (`stratum_contrast` at 0.394) has a
wide one. The verdict now requires both a meaningful effect (mean below 0.48)
and an interval that excludes chance, which labels all seven of that run's
non-informative signals correctly.

And the earlier claim that phylogenetic placement beats nearest-neighbour
identity does not hold on this family: 0.889 against 0.896, a tie. It held on
the simulated family (0.995 vs 0.970) and on the 433-sequence pull. With 3,544
real sequences the nearest neighbour is genuinely close enough to be
informative on its own, which is the honest result.

---

## What it measures

Four layers, each harder to defeat than the last.

| Layer | Signals | Asks |
|---|---|---|
| **Composition** | `comp_llr_family`, `charge_excess`, `cys_depletion`, `rare_depletion`, `low_complexity`, `kmer_repeat`, `hydrophobic_moment`, `construct_motifs` | Does residue usage match the generator rather than this family? |
| **Structural** | `exposed_llr`, `stratum_contrast`, `orphan_fold`, `variable_llr` | Is the buried core preserved while the surface is rewritten? |
| **Covariation** | `coupling_violation` | Does it satisfy position pairs this family *directly* couples? |
| **Evolutionary** | `subst_implausibility`, `directional_drift`, `equilibrium_drift`, `tree_violation`, `rate_incongruence`, `indel_deficit` | Is the implied history one evolution produces? |
| **Phylogenetic** | `pendant_excess`, `ancestral_implausibility` | Does any inferred ancestor explain it? |

Each signal carries its own layer on its `FeatureSpec`, so reports cannot
mislabel it.

### Mean-field DCA instead of mutual information

MI counts every correlation, including transitive ones: if *i* couples to *k*
and *k* to *j*, then *i* and *j* correlate without touching. Inverting the
covariance matrix conditions each pair on all other positions, leaving direct
couplings (Morcos et al. 2011).

Measured on a synthetic chain — A–B coupled, B–C coupled, A–C **not**:

| | A–B (direct) | B–C (direct) | A–C (transitive) | leakage A–C/A–B |
|---|---|---|---|---|
| MI | z = 42.0 | 31.4 | 30.0 | **0.71** |
| DCA | z = 55.7 | 28.6 | 26.1 | **0.47** |

DCA raises the true pair and halves the transitive one. It is not free: the
pseudocount must scale with Meff (λ = Meff is the standard mean-field choice —
with a small absolute pseudocount the covariance is near-singular, its inverse
is numerical noise, and the known coupled pairs scored *1.6 sd below
background*). It also needs Meff well above the model positions; below that it
declines and falls back to MI, saying so in the log.

### Phylogenetic placement instead of nearest neighbour

Comparing a query to its closest relative measures the distance between two
cousins. This builds a neighbour-joining tree on the query's nearest ~16
relatives, reconstructs the ancestor at its root by Fitch parsimony (ties broken
by the profile's own emissions, gaps treated as missing data), and measures the
pendant distance from that ancestor.

The null is leave-one-out: for every member, the same construction *excluding
itself*. So "this sequence attaches nowhere" has a scale behind it.

| measure | AUROC |
|---|---|
| nearest-neighbour identity | 0.970 |
| pendant length from inferred ancestor | **0.995** |

NJ is verified exact on additive data (it recovers branch lengths 0.1/0.2/0.15/0.25
from a known tree to 1e-9).

---

## Repeated-split validation

A single split gives a single number with no error bar, so there is no way to
tell a signal that works from one that got lucky.

```python
report = md.cross_validate(family, n_repeats=8, alpha=0.01, coupling_method="dca")
print(report.report())
```

This refits **everything** — profile, couplings, evolutionary model, placement
null, per-family references, score null — on independent splits, and reports
each signal's AUROC with a 90% interval plus a label-permutation p-value.

Measured: 8 splits of a 2,500-sequence **simulated** family, 354 s. (For the
real-kinase version of this table, see "Measured on real data" above — it is
the one to trust.)

```
Overall AUROC      : 1.0000  [1.0000, 1.0000]
False positives    : 0.606%  [0.000%, 2.576%]   at alpha=0.01
Designs detected   : 100.000%

signal                           AUROC        90% interval    perm p
subst_implausibility             1.000      [1.000, 1.000]    0.0005
ancestral_implausibility         0.998      [0.997, 1.000]    0.0005
exposed_llr                      0.987      [0.978, 0.995]    0.0005
pendant_excess                   0.985      [0.979, 0.992]    0.0005
coupling_violation               0.942      [0.919, 0.958]    0.0005
indel_deficit                    0.720      [0.607, 0.798]    0.0005
rate_incongruence                0.639      [0.583, 0.695]    0.0005
construct_motifs                 0.500      [0.500, 0.500]    1.0000  not distinguishable
cys_depletion                    0.489      [0.424, 0.533]    0.7771  not distinguishable
kmer_repeat                      0.480      [0.466, 0.494]    0.9155  not distinguishable
rare_depletion                   0.403      [0.326, 0.463]    1.0000  not distinguishable
```

Two things only the repeated run reveals:

1. **The false-positive rate is not the nominal one.** It averaged 0.6% at
   α = 0.01 but its interval reached **2.6%** — and the point estimate from any
   one split would have looked fine. With a few-hundred-sequence null the
   empirical floor is 1/(n+1), so α below that relies on tail extrapolation.
   More null sequences is the only real fix.
2. **Four signals carry no evidence on this family**, one of them
   (`rare_depletion`, AUROC 0.403) pointing the wrong way. Soft `tanh`
   orientation keeps their net pull near zero so they do no damage, but their
   weights are doing nothing. On a different family they may be the ones that
   fire — this is a property of the family, not a bug.

---

## Usage

```python
import mpnn_detect as md

profile = md.build_family_profile(family="PF00069", min_length=150, max_length=400)
profile.fit_feature_reference(md.extract_features, md.PROTEINMPNN_PRIOR,
                              reference_sequences=held_out_homologues)

detector = md.Detector(profile=profile, alpha=0.01, generator_agnostic=True)
detector.fit_null(other_held_out_homologues)

print(detector.predict(my_sequence).report())
print(md.screen_batch(submission, detector).report())
md.apply_fdr(results, fdr=0.05)        # q-values, for screens of any size
```

Use three **disjoint** sets: one to build the profile, one for the per-family
reference distributions, one for the null. Reusing sequences between the last
two makes the threshold run loose; the detector warns if it detects the overlap.

### With real generator output

The path that turns the numbers above from bounds into measurements:

```python
# ProteinMPNN writes <out_folder>/seqs/*.fa; point at either, both work.
gens    = md.load_generator_output(["pmpnn_out/vanilla", "pmpnn_out/soluble"])
designs = md.generator_sequences(gens)     # the native record is skipped
print(md.describe_generator_output(gens))  # recovery, temperatures, composition

# Measure the generator's composition instead of assuming it.
fp = md.fit_fingerprint("my_pipeline", [s.seq for s in designs])
md.register_fingerprint(fp)

# Compare like with like: trim the natural class to the aligned domain, because
# a design is one domain and a UniProt entry usually is not.
natural = md.extract_domains(profile, family)

report = md.cross_validate(natural, designed_sequences=designs,
                           fingerprint=fp, alpha=0.01)
print(report.report())        # says "REAL generator output", not "upper bounds"

detector = md.Detector(profile=profile, fingerprint=fp, alpha=0.01)
detector.apply_validation(report, drop_uninformative=True, reweight=True)
detector.fit_null(md.extract_domains(profile, null_set))

# Read these two before believing the pooled rate.
print(md.detection_by_group(detector, gens, by="source"))      # per backbone
natives = [g for g in md.load_generator_output("pmpnn_out/vanilla",
                                               include_native=True)
           if g.is_native]
for g in natives:                       # must NOT come out designed
    print(g.source, detector.predict(g.sequence).p_value)
```

Generating the designs in the first place, which is what the notebook's
section 14 automates:

```bash
git clone --depth 1 https://github.com/dauparas/ProteinMPNN   # weights included
python ProteinMPNN/protein_mpnn_run.py \
    --pdb_path backbones/tyk2.pdb --pdb_path_chains A \
    --out_folder pmpnn_out/vanilla --num_seq_per_target 30 \
    --sampling_temp "0.1 0.2 0.3" --seed 37 --batch_size 1
```

CLI:

```bash
python mpnn_detect.py build  --family PF00069 --out profile_kinase
python mpnn_detect.py screen queries.fasta --profile profile_kinase --csv out.csv
python mpnn_detect.py calibrate --natural nat.fasta --designed mpnn.fasta --out det.json
python mpnn_detect.py fingerprint my_designs.fasta --name my_pipeline
```

### Without knowing the generator

`generator_agnostic=True` drops every signal measured against a fingerprint,
keeping only those that ask whether the sequence could have descended from the
family. Against designs from a generator with no fingerprint it loses nothing —
and the fingerprints shipped here are literature-derived priors, not
measurements.

### Screening at scale

`ScreeningLibrary` holds one calibrated detector per family and routes each
query to whichever profile recognises it. A sequence no profile matches is
flagged `no_family_match`: **not assessed, not cleared.** The routing floor is
deliberately low (10 bits) because a redesign keeps the fold while bit score
falls with how much was rewritten — a high floor rejects the most heavily
redesigned sequences first, which is backwards.

---

---

## Real ProteinMPNN output

Every number above this point was measured against `simulate_designed`, which
samples from the profile the detector scores against and tilts the composition
with the fingerprint the detector tests for. The README has been saying for
three versions that the single highest-value thing left to do was to run it on
real generator output. This is that run.

ProteinMPNN was cloned and run locally on six real kinase catalytic-domain
backbones — `tyk2`, `cmet`, `p38`, `syk`, `cdk2`, `cdk8`, taken from the
OpenFF protein–ligand benchmark because those files have complete N/CA/C/O
backbones — with two trained model variants (`vanilla` and `soluble`) and
sampling temperatures 0.1/0.2/0.3, plus a second wave at 0.5/0.8. The natural
class is the user's own 3,544-sequence PF00069 family, reproduced exactly from
the cached UniProt download.

Measured sequence recovery was 45–48% across every backbone, which matches what
ProteinMPNN reports, so the generator was behaving normally.

### Three things had to be fixed before any number meant anything

**The native record.** ProteinMPNN writes the input sequence as the first
record of every output FASTA, with `designed_chains=` in the header and no
`T=`. It is a natural sequence. `read_proteinmpnn_fasta` skips it by default,
because scoring it as a design puts a guaranteed miss in the design set and
makes the detector look worse than it is.

**The length confound in `dereplicate`.** `max_keep` stopped a longest-first
traversal early, so capping a family returned the *longest* `max_keep`
sequences. On the 3,544 kinases, asking for 1,400 returned sequences of 361–400
aa only, while the designs are 275–349 aa. The first clean-looking run reported

```
natural kinase domains              42       2.4%  <- false positives
REAL ProteinMPNN designs           701      99.9%
the backbones own native seqs        8      50.0%  <- the tell
```

99.9% detection, and half the designs' own *native* backbone sequences also
called designed. That is not detection of design, it is detection of length.
`max_keep` now subsamples at random after dereplication; the length
distribution of a capped family then matches the uncapped one (164–400 aa,
median 348, against 151–400 and median 347).

**Domain context.** A backbone is one domain carved out of a larger protein:
the TYK2 catalytic domain is 288 residues and TYK2 is 1,187. Comparing bare
domains against whole UniProt entries measures domain boundaries, not design.
`extract_domains()` trims the natural class to the same aligned span.

The control that catches all three is the design backbones' own native
sequences, scored as a separate class. They are natural proteins presented
exactly as the designs are — same length, same domain boundaries, same missing
loops. If they get flagged, the detection rate is measuring something else, and
the notebook now says so and names which of the two causes it is, by checking
whether those natives match the profile at all.

### What the composition prior actually looks like against real output

`PROTEINMPNN_PRIOR` was set from published descriptions: strong E/K/A/L/R
enrichment, near-total cysteine avoidance. Measured against real output on
natural kinase backbones, it disagrees in **direction** on D, G, P, W and Y.
Those published descriptions come from de novo helical bundles. ProteinMPNN
redesigning a natural backbone has to satisfy that backbone's burial pattern,
and produces a visibly different distribution — P, W and Y all come out
*enriched* where the prior has them flat or depleted.

The Jensen–Shannon divergence between the prior and real output is about the
same as between the prior and natural sequences, which is the compact way of
saying the prior is not a description of this generator. `fit_fingerprint()` on
real output is strictly better, and `PROTEINMPNN_MEASURED` ships as a default
measured on these six backbones — with the caveat that composition follows the
structure given, so refit on your own targets.

### Detection, measured

Measured on 1080 real ProteinMPNN designs from 6 kinase backbones (cdk2, cdk8, cmet, p38, syk, tyk2), dca couplings, M=313, Meff/M=2.88, family 3544 PF00069 sequences, alpha=0.01, natural class trimmed to the aligned domain.

| weighting | detected | worst-best fold | FPR | AUROC |
|---|---|---|---|---|
| hand-set weights, prior fingerprint | 90.3% | 64% – 100% | 0.00% | 0.9990 |
| hand-set weights, MEASURED fingerprint | 95.3% | 77% – 100% | 0.00% | 0.9998 |
| evolutionary + phylogenetic only | 19.7% | 1% – 71% | 0.40% | 0.9489 |
| prune uninformative | 96.9% | 83% – 100% | 0.00% | 0.9998 |
| prune + reweight | 98.2% | 89% – 100% | 0.00% | 0.9999 |
| FLIP inverted + reweight | 98.7% | 93% – 100% | 0.40% | 0.9998 |

### Signals that point the wrong way against real output

| signal | inverted in | layer |
|---|---|---|
| `indel_deficit` | 6/6 folds | evolutionary |
| `rate_incongruence` | 6/6 folds | evolutionary |
| `stratum_contrast` | 3/6 folds | structural |

An inverted signal is not a dead one: an AUROC of 0.163 reversed is 0.837, more
informative than most signals the detector keeps. `apply_validation` can flip
them, and a bug meant it could not: the reweighting gain was
`max(0, 2*(AUROC-0.5))`, which is exactly zero for anything below chance, and
`-abs(0.0)` is `0.0`. So `reweight=True` silently cancelled
`flip_inverted=True`. The gain is now `2*|AUROC-0.5|` — magnitude says how
informative, sign says which way — and an inverted signal is never allowed to
keep a positive weight.

`indel_deficit` and `rate_incongruence` are inverted in **6 of 6** folds, so
that is a property of the generator rather than one backbone's quirk.
`indel_deficit` has a mechanical explanation: these backbones are crystal
structures with unresolved loops, so a design built on one is missing residues
the profile expects and reads as having *more* indels than a natural sequence,
not fewer. That is an artefact of backbone preparation, and it is a good reason
not to flip on the strength of it — a design built on a complete model could
flip the sign back.

The measurement agrees. Flipping buys +0.5 points of detection, wins **1 of 6**
paired folds, and raises the false-positive rate from 0.00% to 0.40%. That is
not a win, so `flip_inverted` stays off by default. What the fix bought is that
the option now does what it says when someone asks for it.

### The evolutionary layer earns its place against a different adversary

Earlier versions of this README reported that against composition-matched
designs the evolutionary and phylogenetic signals alone did the work, and the
composition layer contributed nothing. Against **real** ProteinMPNN output at
default sampling temperature, that reverses completely: the evolutionary and
phylogenetic signals alone detect 19.7% where the full detector detects 98.2%.

Both are true, and together they say what each layer is for. Composition does
the work against the tool as shipped. The evolutionary layer is insurance
against an adversary who has matched composition on purpose — which real
ProteinMPNN at T ≤ 0.3 does not do. Reporting only the first number would
oversell the deep layers; reporting only the second would invite someone to
delete them.

### Raising the sampling temperature, for real this time

`simulate_designed(composition_blend=...)` was the stand-in for a blunted
composition signature. Real output at T = 0.5 and 0.8 is the thing itself:
sequence recovery falls from 46% to 39%, and the composition ratios against
background drop from (E 1.6x, P 1.6x, Q 0.41x) to (Y 1.6x, P 1.4x, Q 0.72x).

Measured on 240 real ProteinMPNN designs from 6 kinase backbones (cdk2, cdk8, cmet, p38, syk, tyk2), dca couplings, M=313, Meff/M=2.88, family 3544 PF00069 sequences, alpha=0.01, natural class trimmed to the aligned domain.

| weighting | detected | worst-best fold | FPR | AUROC |
|---|---|---|---|---|
| hand-set weights, prior fingerprint | 52.5% | 12% – 78% | 0.00% | 0.9821 |
| hand-set weights, MEASURED fingerprint | 45.0% | 5% – 100% | 0.00% | 0.9899 |
| evolutionary + phylogenetic only | 27.1% | 0% – 75% | 0.40% | 0.9400 |
| prune uninformative | 69.2% | 10% – 100% | 0.74% | 0.9929 |
| prune + reweight | 57.9% | 0% – 100% | 0.67% | 0.9905 |
| FLIP inverted + reweight | 68.3% | 2% – 100% | 1.48% | 0.9858 |

Detection falls from 98.2% to roughly 60–70%. Read nothing more than that from
this table: 240 designs over six backbones is 40 per fold, the worst-to-best
fold spread runs from 0% to 100%, and differences of ten points between rows
are not resolvable at that sample size. The configuration ordering here is
**not** stable and should not be used to pick weights.

One result in it is robust, because it has a mechanism rather than a margin:
the fingerprint measured at T ≤ 0.3 *loses* to the literature prior on this set
(−7.5 points, winning 1 of 6 folds). It was measured at one temperature and
applied at another, and the composition has moved in between. Refit the
fingerprint at the temperature you expect, or use the prior, which is vague
enough not to be wrong in a specific direction.

`stratum_contrast` and `rate_incongruence` are inverted in 6 of 6 folds here,
and `equilibrium_drift` in 5 of 6 — more inversions than at low temperature,
which is consistent with the designs sitting closer to the natural
distribution and the signals' margins shrinking toward noise.



### The control earned its place: half the backbones were out of family

A Colab run at the intended configuration — 3,544 sequences, full DCA at
Meff/M = 3.14, α = 0.01 — reported **98.5% detection at 0.0% false positives**,
and the native-backbone control flagged 2 of 12 natives. Chasing that down
found a confound that neither of the checks in the code could see.

It was not coverage: designs, trimmed naturals and natives all sit at 75–80%
of the profile. It was not domain context. The six backbones split cleanly by
nearest-neighbour identity to the family:

| backbone | kinase type | nearest family member | native p-value | designs detected |
|---|---|---|---|---|
| p38 | Ser/Thr (MAPK) | 99.6% | 0.85 | 100% |
| cdk8 | Ser/Thr (CDK) | 99.6% | 0.62 | 98% |
| cdk2 | Ser/Thr (CDK) | 98.5% | 0.61 | 91% |
| tyk2 | **tyrosine** (JAK) | 38.8% | 0.0031 | 100% |
| cmet | **tyrosine** (MET) | 34.3% | 0.0031 | 100% |
| syk | **tyrosine** (SYK) | 34.3% | 0.071 | 100% |

The family is assembled by filtering whole UniProt entries to 150–400 aa, which
collects *small single-domain kinases* — overwhelmingly Ser/Thr. The family's
own members sit at 79% mean nearest-neighbour identity with a 10th percentile
of 55%. Three of the six backbones are tyrosine kinases at 34–39%: genuine
outliers in this family, on a different branch of the kinase tree. Their
designs were pinned at 100% and two of their three natives were flagged too.

So `backbone_membership()` now runs before the measurement and names the
backbones to drop, with the bar taken from the family's own identity
distribution rather than a fixed cutoff — a tight family and a diverse one have
very different notions of distant. The notebook prints the restricted rate and
says to report that one.

Restricted to the three in-family backbones, leave-one-out over three folds:

| weighting | detected | worst-best fold | FPR |
|---|---|---|---|
| hand-set weights, prior fingerprint | 83.1% | 64% – 100% | 0.00% |
| hand-set weights, MEASURED fingerprint | 93.9% | 84% – 99% | 0.00% |
| evolutionary + phylogenetic only | 13.7% | 1% – 36% | 0.40% |
| prune + reweight | **99.6%** | 99% – 100% | 0.27% |

All three natives pass the control (p = 0.61, 0.62, 0.85). The headline did not
move much — 98.2% over six backbones against
99.6% over the clean three — so the confound was
real but not load-bearing. What changed is the denominator: **the defensible
measurement now rests on three proteins**, and that is the honest error bar,
not the 540 sequences they produced.

Two smaller things from the same run. The native control was reading the native
out of *every* output folder, so "2 of 12" meant "1 of 6 counted twice" — it is
deduplicated by backbone now. And section 14 was inheriting an α its own
smaller, domain-trimmed null could not place; it adopts a reachable one and
says so.


### What simulation actually flattered — measured twice, because once was wrong

Running the notebook end to end produces both reports on the same family: one
against simulated designs, one against the real ones. Subtracting them says
which signals the simulator was propping up. Two runs now exist, and comparing
them matters more than either alone — the first was on a 433-sequence family
with mutual-information couplings, the second on 3,544 with full DCA.

| layer | run A (MI, 433) | run B (DCA, 3,544) | same direction |
|---|---|---|---|
| structural | +0.122 | +0.088 | yes |
| composition | +0.041 | +0.030 | yes |
| covariation | +0.169 | -0.036 | **no** |
| phylogenetic | -0.171 | -0.039 | yes |
| evolutionary | -0.146 | -0.087 | yes |

| signal | layer | run A | run B |
|---|---|---|---|
| `equilibrium_drift` | evolutionary | -0.286 | -0.204 |
| `indel_deficit` | evolutionary | -0.276 | -0.182 |
| `stratum_contrast` | structural | +0.172 | +0.172 |
| `ancestral_implausibility` | phylogenetic | -0.285 | -0.107 |
| `subst_implausibility` | evolutionary | -0.200 | -0.101 |
| `orphan_fold` | structural | +0.232 | +0.096 |
| `variable_llr` | structural | +0.142 | +0.095 |
| `charge_excess` | composition | +0.089 | +0.088 |
| `cys_depletion` | composition | +0.125 | +0.083 |
| `rate_incongruence` | evolutionary | -0.098 | -0.059 |

Positive means the signal does **better** on real output than on simulated, so
simulation was hiding it; negative means simulation was flattering it.

**What replicates.** Simulation flatters the evolutionary and phylogenetic
signals and hides the structural ones, in both runs and by wide margins.
`equilibrium_drift`, `ancestral_implausibility`, `indel_deficit`,
`subst_implausibility` and `rate_incongruence` are all down in both;
`orphan_fold`, `stratum_contrast`, `variable_llr`, `cys_depletion` and
`charge_excess` are all up in both. The mechanism is clear in each direction.
`simulate_designed` ignores phylogeny, so a simulated design lands nowhere near
any real relative and the history signals fire on a sequence no generator would
produce; a real redesign keeps ~46% identity to an actual protein, lands
plausibly on the tree, and they go quiet. And the structural signals read the
residual against the profile, which a simulator sampling *from that profile*
cannot generate.

**What does not replicate, and an earlier claim of mine that was wrong.** On
the first run `coupling_violation` came out +0.169 and I wrote in this file
that simulation *hides* the coupling signal. At full DCA it is −0.036. The
direction is not a property of simulation at all — it is a property of which
coupling model got fitted. On the thin family the fallback was mutual
information, which scored near chance against simulated designs (0.511) and
better against real ones (0.680); with enough sequences for DCA the signal is
strong against both (0.888 simulated, 0.852 real). So: **no direction should be
claimed for the covariation layer.** It was my second correction to this
paragraph, and the lesson is the one already in the Limits section — a single
run, however clean, does not establish an ordering.

### The native-backbone control

| backbone | held-out designs | native p-value | native flagged |
|---|---|---|---|
| cdk2 | 180 | 0.599 | no |
| cdk8 | 180 | 0.578 | no |
| cmet | 180 | 0.0142 | no |
| p38 | 180 | 0.873 | no |
| syk | 180 | 0.106 | no |
| tyk2 | 180 | 0.0142 | no |

## Limits

* **Numbers above the "Real ProteinMPNN output" section are upper bounds.**
  They were measured against `simulate_designed`, which draws from the same
  prior the detector tests for. The real-output section replaces them with
  measurements; everything before it is still a bound.
* **The real measurement is three proteins.** 1,080 designs came from six
  kinase backbones, but three are tyrosine kinases at 34–39% identity to this
  Ser/Thr-dominated family, so their detection is partly distance. The
  defensible number rests on cdk2, cdk8 and p38, and the honest error bar is
  the worst-to-best fold spread over those three, not an interval over the
  sequences they produced. Always run `backbone_membership()` first; a backbone
  that is not in the family you profiled will flatter the result.
* **The native-backbone control is the one check not to skip.** It caught the
  confound above, and the length-selection bug before it. If those natives get
  flagged, the detection rate is measuring something other than design, and
  the notebook now distinguishes the three causes: the backbone is from
  another family (it barely aligns), it is distant within the family (it
  aligns but has no close relatives), or the natural class is whole proteins
  where the designs are bare domains.
* **The composition prior disagrees with real output in direction** on D, G, P,
  W and Y. Prefer `PROTEINMPNN_MEASURED`, or better, `fit_fingerprint()` on
  your own output — and refit it per sampling temperature: the T ≤ 0.3
  fingerprint *loses* to the vague literature prior on T ≥ 0.5 output.
* **The simulator resamples positions independently**, and the consequence is
  not uniform optimism. Across two runs it *flatters* the evolutionary and
  phylogenetic signals (it ignores phylogeny entirely) and *hides* the
  structural ones (it samples from the very profile they measure the residual
  against). The covariation layer does not replicate in either direction: its
  sign depends on whether the family was thick enough for DCA or fell back to
  mutual information. Two earlier versions of this file claimed a coupling
  direction; both were drawn from a single run.
* **DCA needs data.** Above Meff/M = 2 it fits the full 20-residue model, above
  ~0.7 a reduced seven-class one, and below that it declines to mutual
  information. Which tier you get depends on the family's redundancy, not its
  size: real kinases gave Meff/n = 0.47.
* **Placement is local and approximate** — NJ on ~16 neighbours with Fitch
  reconstruction, not a full ML placement on a family-wide tree. It is
  calibrated leave-one-out, which is what makes it interpretable.
* **Sparse families produce false positives**; `sparse_family` is flagged below
  100 members.
* **Localisation is the weakest capability** — window-level AUROC ≈ 0.98, and
  the designed half of a fusion scores higher in ~90% of cases, so read the
  profile shape rather than any single window.
* **A pipeline tuned against a detector defeats that detector.** This is a
  triage aid, not a biosecurity control.
* **AUROC and detection can diverge.** A signal that adds variance without
  adding separation raises the threshold and costs detection at fixed α, which
  AUROC cannot show. Read both, and prefer `apply_validation(reweight=True)`
  over the hand-set priors.
* **Composition-matched designs stay hard, and so does hot sampling.** At
  α = 0.01 the best configuration detects about 60% of simulated
  composition-matched designs, reached by the evolutionary and phylogenetic
  signals. Against real output at T ≥ 0.5 detection is 60–70%, and there the
  layer ordering reverses: the evolutionary signals alone manage 19.7% on real
  T ≤ 0.3 output where the full detector manages 98.2%. Composition does the
  work against the tool as shipped; the evolutionary layer is insurance against
  someone who has matched composition deliberately.
* **Single-split differences of a few points are noise.** On 80 designs that is
  a handful of sequences. One claim in an earlier version of this file did not
  survive repetition; repeat before believing an ordering.
* **Signals are backwards on real output, reproducibly.** Against real
  ProteinMPNN at T ≤ 0.3, `indel_deficit` and `rate_incongruence` are inverted
  in 6 of 6 leave-one-backbone-out folds; at T ≥ 0.5, `stratum_contrast` and
  `rate_incongruence` are. `indel_deficit` has a known cause - these backbones
  are crystal structures with unresolved loops - so flipping it would fit the
  backbone preparation, not the generator. Run `cross_validate` then
  `apply_validation` on your own family before trusting any weight.
* **Simulation flattered almost everything.** Of 21 signals, 17 looked
  informative on simulated designs and 14 were on real kinases — and
  `stratum_contrast` went from 0.95 to 0.374. Treat any number measured against
  `simulate_designed` as an upper bound, and re-run `cross_validate` on the
  family you care about before trusting a weight.
* **Only one generator was actually run.** ProteinMPNN, in two trained variants
  (`vanilla` and `soluble`). LigandMPNN and ESM-IF are supported by the parser
  and untested against their real output, because their weights are not hosted
  anywhere this sandbox can reach. The notebook runs them wherever you can
  reach their weights; `load_generator_output` reads plain FASTA from any tool,
  with the per-sample metadata simply absent.
* **A detection rate at 0% false positives on 300 natural sequences is not a
  0% false-positive rate.** The smallest rate that set can resolve is 1/300.

Next, in order of value: run it on **your** family and **your** generator.
`fit_fingerprint` at the sampling temperature you expect, `extract_domains` so
both classes are the same thing, then
`cross_validate(..., designed_sequences=...)` — and read the backbones' own
native sequences as a control before believing the headline.
