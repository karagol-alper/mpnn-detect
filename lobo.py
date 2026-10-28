"""Leave-one-backbone-out: does anything learned from real designs generalise?

The diagnostic run turned up three signals that are strongly INVERTED against
real ProteinMPNN output - rate_incongruence 0.163, indel_deficit 0.202,
stratum_contrast 0.242, all with tight intervals. An AUROC of 0.163 reversed is
0.837, so these are not dead signals; they are informative ones pointing
backwards. `apply_validation` can flip them, but flipping on the strength of
the family in front of you is how a quirk gets baked into a model.

The honest test is whether a weighting derived from some backbones helps on a
backbone it has never seen. So: for each of the six, derive the report from the
other five and measure detection on the held-out one. Six folds, and the
natural side is split five ways so no set of natural sequences does two jobs.

Everything here is scoring against one profile build, because the profile does
not depend on the designs. That is what makes six folds affordable.
"""
import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mpnn_detect as md

md.LOG.setLevel(logging.ERROR)

AP = argparse.ArgumentParser()
AP.add_argument("--family", default="/home/claude/w7/kin.fasta")
AP.add_argument("--designs", nargs="+", default=["/home/claude/pmpnn_out/vanilla",
                                                 "/home/claude/pmpnn_out/soluble"])
AP.add_argument("--alpha", type=float, default=0.01)
AP.add_argument("--coupling", default="dca")
AP.add_argument("--symfrac", type=float, default=0.6)
AP.add_argument("--max-family", type=int, default=0)
AP.add_argument("--n-boot", type=int, default=300)
AP.add_argument("--extract-domains", action="store_true")
AP.add_argument("--out", default="")
AP.add_argument("--only", nargs="*", default=None,
                help="restrict to these backbones, e.g. the ones "
                     "backbone_membership() says are in family")
AP.add_argument("--profile-dir", default="",
                help="save/reuse the built profile here; the "
                     "profile does not depend on the designs, so "
                     "rebuilding it per run is pure waste")
A = AP.parse_args()


def banner(t):
    print(f"\n{t}\n{'=' * len(t)}", flush=True)


# ------------------------------------------------------------------ data ----
gens = md.load_generator_output(A.designs, include_native=True)
# One native per backbone: it is written into every run folder, so reading
# two model variants yields each native twice and would weight the control
# by how many runs happened to be loaded.
natives, _seen = [], set()
for g in gens:
    if g.is_native and g.source not in _seen:
        _seen.add(g.source)
        natives.append(g)
designs = [g for g in gens if not g.is_native]
if A.only:
    designs = [g for g in designs if g.source in set(A.only)]
    natives = [g for g in natives if g.source in set(A.only)]
    if not designs:
        raise SystemExit(f"no designs from {A.only}")
backbones = sorted({g.source for g in designs})
banner(f"{len(designs)} real designs from {len(backbones)} backbones: "
       f"{', '.join(backbones)}")
print(md.describe_generator_output(designs))

seqs = md.load_sequences(A.family, min_length=150)
seqs = [s for s in seqs if 150 <= len(s) <= 400]
if A.max_family:
    seqs = seqs[:A.max_family]
random.Random(0).shuffle(seqs)

# Five disjoint natural sets. Reusing one set for two jobs is how a model ends
# up validated on its own training data without anyone noticing.
n = len(seqs)
n_null = int(min(0.22 * n, max(60, 5.0 / A.alpha)))
n_eval = int(min(0.12 * n, max(40, n_null * 0.6)))
n_rep = n_eval
n_ref = int(0.20 * n)
n_build = n - n_null - n_eval - n_rep - n_ref
build = seqs[:n_build]
ref = seqs[n_build:n_build + n_ref]
rep_nat = seqs[n_build + n_ref:n_build + n_ref + n_rep]
null_set = seqs[n_build + n_ref + n_rep:n_build + n_ref + n_rep + n_null]
eval_set = seqs[n_build + n_ref + n_rep + n_null:][:n_eval]
print(f"\nnatural splits: build {len(build)}, reference {len(ref)}, "
      f"signal-ranking {len(rep_nat)}, null {len(null_set)}, "
      f"evaluation {len(eval_set)}")

t0 = time.time()
if A.profile_dir and os.path.exists(os.path.join(A.profile_dir, "profile.json")):
    prof = md.FamilyProfile.load(A.profile_dir)
    print(f"reused profile from {A.profile_dir}", flush=True)
else:
    prof = md.FamilyProfile("kinase", random_state=0).build(
        build, check_homogeneity=False, coupling_method=A.coupling,
        symfrac=A.symfrac)
    prof.fit_feature_reference(md.extract_features, md.PROTEINMPNN_PRIOR,
                               reference_sequences=ref)
    if A.profile_dir:
        prof.save(A.profile_dir)
print(f"profile: M={prof.hmm.M}, Meff={prof.meff:.0f}, "
      f"Meff/M={prof.meff / prof.hmm.M:.2f}, {prof.coupling['method']} "
      f"({time.time() - t0:.0f}s)", flush=True)

if A.extract_domains:
    banner("Trimming the natural class to the aligned domain")
    rep_nat = md.extract_domains(prof, rep_nat)
    null_set = md.extract_domains(prof, null_set)
    eval_set = md.extract_domains(prof, eval_set)
    print(f"ranking {len(rep_nat)}, null {len(null_set)}, "
          f"evaluation {len(eval_set)} after trimming")

# ----------------------------------------------- measured fingerprint -------
fp_measured = md.fit_fingerprint("proteinmpnn_measured",
                                 [g.sequence.seq for g in designs])
md.register_fingerprint(fp_measured)

EVO = [k for k in md.FEATURE_KEYS
       if md.SPEC_BY_KEY[k].layer in ("evolutionary", "phylogenetic")]


def signal_report(det, design_subset, seed=0):
    """Rank the signals using these designs and the ranking-only natural set."""
    rn = [det.predict(s) for s in rep_nat]
    rd = [det.predict(g.sequence) for g in design_subset]
    keys = [s.key for s in rn[0].signals]
    Aa = np.array([[s.standardized for s in r.signals] for r in rn])
    Bb = np.array([[s.standardized for s in r.signals] for r in rd])
    rng = np.random.default_rng(seed)
    sigs = []
    for i, k in enumerate(keys):
        a, b = Aa[:, i], Bb[:, i]
        boots = [md._auroc(a[rng.integers(0, len(a), len(a))],
                           b[rng.integers(0, len(b), len(b))])
                 for _ in range(A.n_boot)]
        lo, hi = md._percentile_ci(boots)
        sigs.append(md.SignalPerformance(
            key=k, auroc_mean=md._auroc(a, b), auroc_lo=lo, auroc_hi=hi,
            permutation_p=md._permutation_p(a, b, 1000, seed=seed),
            net_pull_mean=0.0, n_repeats=1))
    nan = float("nan")
    return md.ValidationReport(
        n_repeats=1, alpha=A.alpha, overall_auroc_mean=nan,
        overall_auroc_ci=(nan, nan), fpr_mean=nan, fpr_ci=(nan, nan),
        tpr_mean=nan, tpr_ci=(nan, nan), signals=sigs, seconds=0.0,
        designs_were_real=True, n_designs=len(design_subset))


def evaluate(fingerprint, validated, held_out, weights=None, **kw):
    """Fit the threshold and score the held-out backbone's designs."""
    det = md.Detector(profile=prof, fingerprint=fingerprint, alpha=A.alpha,
                      weights=dict(weights) if weights else None,
                      bias=md.DEFAULT_BIAS if weights else None)
    if weights:
        det.calibrated = False
    if validated is not None:
        det.apply_validation(validated, **kw)
    info = det.fit_null(null_set)
    alpha = A.alpha
    rec = info.get("recommended_alpha")
    if rec and rec > alpha:
        alpha = round(rec, 4)
        det.alpha = alpha
    rn = [det.predict(s) for s in eval_set]
    rd = [det.predict(g.sequence) for g in held_out]
    return {
        "tpr": sum(1 for r in rd if r.p_value <= alpha) / len(rd),
        "fpr": sum(1 for r in rn if r.p_value <= alpha) / len(rn),
        "auroc": md._auroc([r.score for r in rn], [r.score for r in rd]),
        "alpha": alpha,
    }


CONDITIONS = [
    ("hand-set weights, prior fingerprint", dict(fp="prior", mode="none")),
    ("hand-set weights, MEASURED fingerprint", dict(fp="measured", mode="none")),
    ("evolutionary + phylogenetic only", dict(fp="measured", mode="evo")),
    ("prune uninformative", dict(fp="measured", mode="prune")),
    ("prune + reweight", dict(fp="measured", mode="reweight")),
    ("FLIP inverted + reweight", dict(fp="measured", mode="flip")),
]

banner("Leave-one-backbone-out: weights from five backbones, "
       "measured on the sixth")
print("Every row is a mean over six folds; each fold's detection rate comes "
      "from a\nbackbone whose designs played no part in choosing the weights.\n")

results = {label: [] for label, _ in CONDITIONS}
fold_detail = {}
for held in backbones:
    tr = [g for g in designs if g.source != held]
    te = [g for g in designs if g.source == held]
    rep_prior = signal_report(
        md.Detector(profile=prof, fingerprint=md.PROTEINMPNN_PRIOR,
                    alpha=A.alpha), tr)
    rep_meas = signal_report(
        md.Detector(profile=prof, fingerprint=fp_measured, alpha=A.alpha), tr)
    for label, spec in CONDITIONS:
        fp = md.PROTEINMPNN_PRIOR if spec["fp"] == "prior" else fp_measured
        rep = rep_prior if spec["fp"] == "prior" else rep_meas
        if spec["mode"] == "none":
            out = evaluate(fp, None, te)
        elif spec["mode"] == "evo":
            out = evaluate(fp, None, te, weights={
                k: (v if k in EVO else 0.0)
                for k, v in md.DEFAULT_WEIGHTS.items()})
        elif spec["mode"] == "prune":
            out = evaluate(fp, rep, te, drop_uninformative=True)
        elif spec["mode"] == "reweight":
            out = evaluate(fp, rep, te, drop_uninformative=True, reweight=True)
        else:
            out = evaluate(fp, rep, te, drop_uninformative=True,
                           flip_inverted=True, reweight=True)
        results[label].append(out)
    fold_detail[held] = {"n": len(te),
                         "inverted": [s.key for s in rep_meas.signals
                                      if s.verdict.startswith("inverted")]}
    print(f"  fold {held}: {len(te)} held-out designs, "
          f"{len(fold_detail[held]['inverted'])} signals inverted in training",
          flush=True)

print()
print(f"{'weighting':<40}{'detected':>10}{'spread':>18}{'FPR':>8}{'AUROC':>8}")
print("-" * 84)
for label, _ in CONDITIONS:
    t = [r["tpr"] for r in results[label]]
    f = [r["fpr"] for r in results[label]]
    au = [r["auroc"] for r in results[label]]
    print(f"{label:<40}{np.mean(t):>10.1%}"
          f"{f'[{min(t):.0%} - {max(t):.0%}]':>18}"
          f"{np.mean(f):>8.1%}{np.mean(au):>8.4f}")
print("\nspread is worst-to-best fold, which is the error bar that matters: "
      "it is\nhow much the answer moves when the protein changes.")

# paired, because the folds are the same across conditions
banner("Paired differences across the six folds")
base = np.array([r["tpr"] for r in results["prune + reweight"]])
for label in ("FLIP inverted + reweight", "hand-set weights, MEASURED fingerprint",
              "evolutionary + phylogenetic only", "prune uninformative"):
    d = np.array([r["tpr"] for r in results[label]]) - base
    print(f"  {label + ' - (prune + reweight)':<52}{np.mean(d):>+8.1%}"
          f"   wins {int((d > 0).sum())}/{len(d)}")

fp_d = (np.array([r["tpr"] for r in results["hand-set weights, MEASURED fingerprint"]])
        - np.array([r["tpr"] for r in results["hand-set weights, prior fingerprint"]]))
print(f"\n  {'measured fingerprint - literature prior':<52}{np.mean(fp_d):>+8.1%}"
      f"   wins {int((fp_d > 0).sum())}/{len(fp_d)}")

# ------------------------------------------------- which signals inverted ---
banner("Which signals came out inverted, and in how many folds")
counts: dict = {}
for held, d in fold_detail.items():
    for k in d["inverted"]:
        counts[k] = counts.get(k, 0) + 1
if counts:
    for k, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<26}{md.SPEC_BY_KEY[k].layer:<14}{c}/{len(backbones)} folds")
    print("\nA signal inverted in every fold is a property of the generator, "
          "not of\none backbone. One inverted in a single fold is that "
          "backbone's quirk.")
else:
    print("  none")

# ------------------------------------------------- native control -----------
banner("Control: the backbones' own native sequences")
det = md.Detector(profile=prof, fingerprint=fp_measured, alpha=A.alpha)
info = det.fit_null(null_set)
alpha = max(A.alpha, info.get("recommended_alpha") or A.alpha)
det.alpha = alpha
print(f"{'backbone':<12}{'p-value':>12}{'called designed':>18}")
print("-" * 42)
for g in sorted(natives, key=lambda g: g.source):
    r = det.predict(g.sequence)
    print(f"{g.source:<12}{r.p_value:>12.3g}{str(r.p_value <= alpha):>18}")
n_flag = sum(1 for g in natives if det.predict(g.sequence).p_value <= alpha)
print(f"\n{n_flag}/{len(natives)} native backbone sequences called designed "
      f"at alpha={alpha}.")
print("These are natural proteins presented exactly as the designs are - same\n"
      "length, same domain boundaries, same missing loops. If they were being\n"
      "flagged, the detection rate above would be measuring domain context.")

if A.out:
    with open(A.out, "w") as fh:
        json.dump({"backbones": backbones, "n_designs": len(designs),
                   "alpha": A.alpha, "coupling": prof.coupling["method"],
                   "M": prof.hmm.M, "meff": prof.meff,
                   "inverted_counts": counts,
                   "measured_composition": {a: float(fp_measured.freqs[i])
                                            for i, a in enumerate(md.AA)},
                   "conditions": {label: results[label]
                                  for label, _ in CONDITIONS},
                   "native_flagged": n_flag}, fh, indent=1)
    print(f"\nwrote {A.out}")

# --------------------------------------------------------------- markdown ---
# Printed so the README tables are generated from the run rather than typed
# from it. An earlier version of the README carried a claim that did not
# survive repetition; copying numbers by hand is how that happens.
banner("README tables (generated, not transcribed)")
print(f"Measured on {len(designs)} real ProteinMPNN designs from "
      f"{len(backbones)} kinase backbones "
      f"({', '.join(backbones)}), {prof.coupling['method']} couplings, "
      f"M={prof.hmm.M}, Meff/M={prof.meff / prof.hmm.M:.2f}, "
      f"family {len(seqs)} PF00069 sequences, alpha={A.alpha}"
      + (", natural class trimmed to the aligned domain" if A.extract_domains
         else "") + ".\n")
print("| weighting | detected | worst-best fold | FPR | AUROC |")
print("|---|---|---|---|---|")
for label, _ in CONDITIONS:
    t = [r["tpr"] for r in results[label]]
    f = [r["fpr"] for r in results[label]]
    au = [r["auroc"] for r in results[label]]
    print(f"| {label} | {np.mean(t):.1%} | {min(t):.0%} – {max(t):.0%} | "
          f"{np.mean(f):.2%} | {np.mean(au):.4f} |")

print("\n| signal | inverted in | layer |")
print("|---|---|---|")
for k, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"| `{k}` | {c}/{len(backbones)} folds | "
          f"{md.SPEC_BY_KEY[k].layer} |")

print("\n| backbone | held-out designs | native p-value | native flagged |")
print("|---|---|---|---|")
for g in sorted(natives, key=lambda g: g.source):
    r = det.predict(g.sequence)
    n_held = sum(1 for d in designs if d.source == g.source)
    print(f"| {g.source} | {n_held} | {r.p_value:.3g} | "
          f"{'yes' if r.p_value <= alpha else 'no'} |")
