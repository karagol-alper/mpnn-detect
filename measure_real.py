"""Measure detection of REAL ProteinMPNN output, not simulated designs.

Every number this project has reported so far came from `simulate_designed`,
which samples from the same profile the detector scores against and applies the
same composition fingerprint the detector tests for. That makes the measurement
circular in the generous direction: the simulator cannot produce a design whose
failure mode the detector does not already model.

This script replaces that with real output. The designs are genuine ProteinMPNN
sequences, generated locally from real kinase catalytic-domain backbones, and
scored by a detector built from the user's own 3,544-sequence PF00069 family.

Two things are reported separately because they answer different questions:

  pooled     - detection over all designs, which is the headline number
  per-backbone - the same designs grouped by the backbone they came from

The pooled number alone would overstate the evidence. 540 designs from six
backbones are not 540 independent trials: ProteinMPNN conditions on structure,
so designs from one backbone share whatever that backbone does to the signals.
The effective sample size for "does this work on a new protein" is six.
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
AP.add_argument("--designs", nargs="+",
                default=["/home/claude/pmpnn_out/vanilla"])
AP.add_argument("--alpha", type=float, default=0.01)
AP.add_argument("--coupling", default="dca")
AP.add_argument("--symfrac", type=float, default=0.6)
AP.add_argument("--max-family", type=int, default=0)
AP.add_argument("--out", default="")
AP.add_argument("--drop-homologs", type=float, default=0.0,
                help="drop family members above this identity to a design's "
                     "native sequence, to remove the obvious leak")
ARGS = AP.parse_args()


def banner(text):
    print(f"\n{text}\n{'=' * len(text)}", flush=True)


# ---------------------------------------------------------------- designs ---
gens = md.load_generator_output(ARGS.designs, include_native=True)
natives = [g for g in gens if g.is_native]
designs = [g for g in gens if not g.is_native]
banner(f"Real generator output: {len(designs)} designs, "
       f"{len(natives)} native backbone sequences")
print(md.describe_generator_output(designs))

# ---------------------------------------------------------------- family ----
seqs = md.load_sequences(ARGS.family, min_length=150)
seqs = [s for s in seqs if 150 <= len(s) <= 400]
if ARGS.max_family:
    seqs = seqs[:ARGS.max_family]
banner(f"Natural family: {len(seqs)} PF00069 kinases")

# The six backbones are human kinases, so the family almost certainly contains
# their native sequences. A profile built on those sequences knows this exact
# subfamily unusually well, which flatters the measurement. Report the overlap,
# and optionally remove it.
nat_seqs = [g.sequence.seq for g in natives]
if nat_seqs:
    ident = []
    for s in seqs:
        ident.append(max(md.sequence_identity(s.seq, n) for n in nat_seqs))
    ident = np.array(ident)
    print(f"identity of family members to the design backbones' natives: "
          f"max {ident.max():.0%}, "
          f">=90%: {(ident >= 0.9).sum()}, >=60%: {(ident >= 0.6).sum()}")
    if ARGS.drop_homologs:
        keep = ident < ARGS.drop_homologs
        print(f"dropping {(~keep).sum()} members above "
              f"{ARGS.drop_homologs:.0%} identity to a backbone native")
        seqs = [s for s, k in zip(seqs, keep) if k]

# ---------------------------------------------------------------- split -----
random.Random(0).shuffle(seqs)
n = len(seqs)
nn = int(min(0.35 * n, max(60, 5 / ARGS.alpha)))
ne = int(min(0.20 * n, max(40, nn * 0.6)))
nr = int(0.23 * n)
nb = n - nn - ne - nr
build, ref = seqs[:nb], seqs[nb:nb + nr]
nul, ev = seqs[nb + nr:nb + nr + nn], seqs[nb + nr + nn:][:ne]
print(f"split: build {len(build)}, reference {len(ref)}, null {len(nul)}, "
      f"evaluation {len(ev)}")

t0 = time.time()
prof = md.FamilyProfile("kinase", random_state=0).build(
    build, check_homogeneity=False, coupling_method=ARGS.coupling,
    symfrac=ARGS.symfrac)
prof.fit_feature_reference(md.extract_features, md.PROTEINMPNN_PRIOR,
                           reference_sequences=ref)
print(f"profile: M={prof.hmm.M}, Meff={prof.meff:.0f}, "
      f"Meff/M={prof.meff / prof.hmm.M:.2f}, couplings={prof.coupling['method']} "
      f"({time.time() - t0:.0f}s)")

det = md.Detector(profile=prof, fingerprint=md.PROTEINMPNN_PRIOR,
                  alpha=ARGS.alpha)
info = det.fit_null(nul)
rec = info.get("recommended_alpha")
alpha = ARGS.alpha
if rec and rec > alpha:
    alpha = round(rec, 4)
    det.alpha = alpha
    print(f"alpha {ARGS.alpha} is below this null's floor of 1/{info['n'] + 1}; "
          f"using alpha={alpha}")

# ---------------------------------------------------------------- score -----
banner("Detection of real ProteinMPNN designs")
rn = [det.predict(s) for s in ev]
rd = {g.sequence.name: det.predict(g.sequence) for g in designs}
nat_res = [det.predict(g.sequence) for g in natives]

nat_scores = [r.score for r in rn]
des_scores = [rd[g.sequence.name].score for g in designs]
auroc = md._auroc(nat_scores, des_scores)
tpr = sum(1 for g in designs if rd[g.sequence.name].p_value <= alpha) / len(designs)
fpr = sum(1 for r in rn if r.p_value <= alpha) / len(rn)

print(f"{'':<34}{'n':>6}{'AUROC':>9}{'detected':>10}")
print("-" * 59)
print(f"{'natural kinases (evaluation set)':<34}{len(rn):>6}{'':>9}"
      f"{fpr:>9.1%}  <- false positives")
print(f"{'real ProteinMPNN designs':<34}{len(designs):>6}{auroc:>9.4f}{tpr:>10.1%}")
if nat_res:
    nat_called = sum(1 for r in nat_res if r.p_value <= alpha) / len(nat_res)
    print(f"{'the backbones own native seqs':<34}{len(nat_res):>6}{'':>9}"
          f"{nat_called:>9.1%}  <- must stay low")

# per backbone
banner("Per backbone (six structures, not 540 independent trials)")
print(f"{'backbone':<14}{'n':>5}{'detected':>10}{'median p':>12}{'recovery':>10}")
print("-" * 51)
by_src = {}
for g in designs:
    by_src.setdefault(g.source, []).append(g)
for src in sorted(by_src):
    grp = by_src[src]
    ps = np.array([rd[g.sequence.name].p_value for g in grp])
    r = [g.seq_recovery for g in grp if g.seq_recovery is not None]
    print(f"{src:<14}{len(grp):>5}{(ps <= alpha).mean():>10.1%}"
          f"{np.median(ps):>12.2g}{(np.mean(r) if r else float('nan')):>10.1%}")

# per temperature
banner("Per sampling temperature")
print(f"{'T':<8}{'n':>5}{'detected':>10}{'median p':>12}")
print("-" * 35)
by_t = {}
for g in designs:
    by_t.setdefault(g.temperature, []).append(g)
for t in sorted(by_t, key=lambda x: (x is None, x)):
    grp = by_t[t]
    ps = np.array([rd[g.sequence.name].p_value for g in grp])
    print(f"{('?' if t is None else f'{t:g}'):<8}{len(grp):>5}"
          f"{(ps <= alpha).mean():>10.1%}{np.median(ps):>12.2g}")

# ---------------------------------------------------------------- signals ---
banner("Per-signal AUROC on REAL designs")
KEYS = [s.key for s in rn[0].signals]
A = np.array([[s.standardized for s in r.signals] for r in rn])
B = np.array([[s.standardized for s in rd[g.sequence.name].signals]
              for g in designs])
rng = np.random.default_rng(0)
rows = []
for i, k in enumerate(KEYS):
    a, b = A[:, i], B[:, i]
    boots = [md._auroc(a[rng.integers(0, len(a), len(a))],
                       b[rng.integers(0, len(b), len(b))]) for _ in range(400)]
    lo, hi = md._percentile_ci(boots)
    rows.append(md.SignalPerformance(
        key=k, auroc_mean=md._auroc(a, b), auroc_lo=lo, auroc_hi=hi,
        permutation_p=md._permutation_p(a, b, 2000, seed=0),
        net_pull_mean=0.0, n_repeats=1))

print(f"{'signal':<26}{'layer':<14}{'AUROC':>7}{'interval':>17}  verdict")
print("-" * 92)
for s in sorted(rows, key=lambda s: -s.auroc_mean):
    print(f"{s.key:<26}{md.SPEC_BY_KEY[s.key].layer:<14}{s.auroc_mean:>7.3f}"
          f"  [{s.auroc_lo:.3f},{s.auroc_hi:.3f}]  {s.verdict}")
print(f"\n{sum(s.informative for s in rows)}/{len(rows)} signals informative "
      f"against real output")

# ------------------------------------------------- composition, measured ----
banner("Measured ProteinMPNN composition vs the hand-set prior")
emp = md.composition("".join(g.sequence.seq for g in designs))
prior = md.PROTEINMPNN_PRIOR.freqs
nat = md.composition("".join(s.seq for s in ev))
print(f"{'aa':<4}{'measured':>10}{'prior':>9}{'natural':>10}"
      f"{'meas/nat':>10}{'prior/nat':>11}  prior direction")
print("-" * 70)
wrong = []
for i, a in enumerate(md.AA):
    m, p, q = emp[i], prior[i], nat[i]
    d_m, d_p = m - q, p - q
    ok = "" if d_m * d_p > 0 else "  <- wrong way"
    if d_m * d_p <= 0 and abs(d_m) > 0.004:
        wrong.append(a)
    print(f"{a:<4}{m:>10.4f}{p:>9.4f}{q:>10.4f}{m / max(q, 1e-9):>10.2f}"
          f"{p / max(q, 1e-9):>11.2f}{ok}")
print(f"\nprior points the wrong way on: {', '.join(wrong) or 'nothing'}")
m = 0.5 * (prior + emp)
js = 0.5 * float(np.sum(prior * np.log(prior / m)) + np.sum(emp * np.log(emp / m)))
print(f"prior-vs-measured Jensen-Shannon divergence: {js:.4f} nats "
      f"(prior-vs-natural {0.5 * float(np.sum(prior * np.log(prior / (0.5 * (prior + nat)))) + np.sum(nat * np.log(nat / (0.5 * (prior + nat))))):.4f})")

if ARGS.out:
    payload = {
        "alpha": alpha, "auroc": auroc, "tpr": tpr, "fpr": fpr,
        "n_designs": len(designs), "n_eval": len(rn),
        "M": prof.hmm.M, "meff": prof.meff, "n_build": len(build),
        "per_backbone": {k: float(np.mean([rd[g.sequence.name].p_value <= alpha
                                           for g in v]))
                         for k, v in by_src.items()},
        "signals": [{"key": s.key, "auroc_mean": s.auroc_mean,
                     "auroc_lo": s.auroc_lo, "auroc_hi": s.auroc_hi,
                     "permutation_p": s.permutation_p,
                     "net_pull_mean": 0.0, "n_repeats": 1} for s in rows],
        "measured_composition": {a: float(emp[i]) for i, a in enumerate(md.AA)},
    }
    with open(ARGS.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {ARGS.out}")
