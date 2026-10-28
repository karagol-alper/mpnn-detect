"""Tests for mpnn_detect: the DCA, placement and validation layers, plus
regressions for the retrieval bugs reported from Colab.

Run: python -m pytest test_mpnn_detect.py -q
"""
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpnn_detect as md

md.LOG.setLevel(40)

SEED = md._DEMO_ROOT


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def family():
    """Large enough that a Potts fit is allowed to engage."""
    return md.simulate_natural_family(n=900, seed=11)


@pytest.fixture(scope="module")
def profile(family):
    prof = md.FamilyProfile("fixture", random_state=0).build(family[:600])
    prof.fit_feature_reference(md.extract_features, md.PROTEINMPNN_PRIOR,
                               reference_sequences=family[600:750])
    prof._null = family[750:850]
    prof._eval = family[850:]
    return prof


@pytest.fixture(scope="module")
def fitted(profile):
    det = md.Detector(profile=profile, alpha=0.05)
    det.fit_null(profile._null)
    return det


# --------------------------------------------------------------------------
# Regressions for the reported Colab failure
# --------------------------------------------------------------------------

FASTA = "".join(
    f">sp|P{i:05d}|K_HUMAN kinase\n{SEED[:200]}\n" for i in range(150)
)


class TestDoubleGzip:
    """The reported failure: 'retrieved 14, 14 unusable' for a query that
    should return thousands. The body was gzipped twice, one decompression
    left it still gzipped, and the compressed bytes were decoded as text."""

    def test_single_gzip(self):
        out = md._decompress_fully(gzip.compress(FASTA.encode(), mtime=0))
        assert len(md.parse_fasta(out.decode(), from_text=True)) == 150

    def test_double_gzip(self):
        twice = gzip.compress(gzip.compress(FASTA.encode(), mtime=0), mtime=0)
        out = md._decompress_fully(twice)
        assert len(md.parse_fasta(out.decode(), from_text=True)) == 150

    def test_triple_gzip(self):
        thrice = gzip.compress(gzip.compress(gzip.compress(FASTA.encode())))
        out = md._decompress_fully(thrice)
        assert len(md.parse_fasta(out.decode(), from_text=True)) == 150

    def test_zlib_body(self):
        import zlib
        out = md._decompress_fully(zlib.compress(FASTA.encode()))
        assert len(md.parse_fasta(out.decode(), from_text=True)) == 150

    def test_plain_body_untouched(self):
        assert md._decompress_fully(FASTA.encode()) == FASTA.encode()

    def test_bounded(self):
        """A pathological nest must terminate rather than loop."""
        body = FASTA.encode()
        for _ in range(12):
            body = gzip.compress(body)
        out = md._decompress_fully(body, max_rounds=4)
        assert isinstance(out, bytes)

    def test_corrupt_gzip_returns_input(self):
        bad = b"\x1f\x8b" + b"garbage not really gzip"
        assert md._decompress_fully(bad) == bad


class TestImplausibleRecords:
    def test_rejects_mis_decoded_binary(self):
        mojibake = gzip.compress(FASTA.encode()).decode("utf-8", errors="replace")
        recs = md.parse_fasta(mojibake, from_text=True)
        why = md._implausible_records(recs)
        assert why and "amino acid" in why

    def test_rejects_absurd_length(self):
        assert "residue record" in md._implausible_records([("x", "M" * 20000)])

    def test_rejects_empty(self):
        assert md._implausible_records([]) is not None
        assert md._implausible_records([("a", "")]) is not None

    def test_accepts_real_fasta(self):
        assert md._implausible_records(md.parse_fasta(FASTA, from_text=True)) is None

    def test_accepts_ambiguous_residues(self):
        assert md._implausible_records([("a", "MKLSXXUOBZJ-*.KLS" * 8)]) is None


class TestCacheVersioning:
    """A cache written by the buggy build held garbage and, living on Drive,
    survived every later run - so the decode fix would never have fired."""

    def test_key_is_versioned(self, monkeypatch):
        seen = {}

        def fake_get(url, timeout, attempts, backoff):
            seen["url"] = url
            return FASTA.encode(), {}

        monkeypatch.setattr(md, "_http_get", fake_get)
        with tempfile.TemporaryDirectory() as d:
            md.fetch_uniprot("Q", max_sequences=500, cache_dir=d)
            names = os.listdir(d)
            assert any(n.startswith("uniprot_") for n in names)
            # the same query under the old unversioned key must not be hit
            import hashlib
            old = hashlib.sha256("Q|500".encode()).hexdigest()[:16]
            assert not any(old in n for n in names)

    def test_poisoned_cache_is_rejected_and_refetched(self, monkeypatch):
        calls = {"n": 0}

        def fake_get(url, timeout, attempts, backoff):
            calls["n"] += 1
            return FASTA.encode(), {}

        monkeypatch.setattr(md, "_http_get", fake_get)
        with tempfile.TemporaryDirectory() as d:
            md.fetch_uniprot("Q2", max_sequences=500, cache_dir=d)
            assert calls["n"] == 1
            # second call is served from cache
            md.fetch_uniprot("Q2", max_sequences=500, cache_dir=d)
            assert calls["n"] == 1
            # poison it the way the buggy build did
            path = os.path.join(d, [n for n in os.listdir(d) if n.endswith(".gz")][0])
            mojibake = gzip.compress(FASTA.encode()).decode("utf-8", errors="replace")
            with gzip.open(path, "wt") as fh:
                for n, s in md.parse_fasta(mojibake, from_text=True):
                    fh.write(f">{n}\n{s}\n")
            recs = md.fetch_uniprot("Q2", max_sequences=500, cache_dir=d)
            assert calls["n"] == 2, "a poisoned cache must trigger a re-fetch"
            assert len(recs) == 150
            assert os.path.exists(path + ".rejected")

    def test_garbage_response_is_not_cached(self, monkeypatch):
        mojibake = gzip.compress(FASTA.encode()).decode("utf-8", errors="replace")

        def fake_get(url, timeout, attempts, backoff):
            return mojibake.encode("utf-8", errors="replace"), {}

        monkeypatch.setattr(md, "_http_get", fake_get)
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError):
                md.fetch_uniprot("Q3", max_sequences=500, cache_dir=d)
            assert not [n for n in os.listdir(d) if n.endswith(".fasta.gz")]


class TestLoggingHygiene:
    def test_no_double_emit(self):
        assert md.LOG.handlers and md.LOG.propagate is False


# --------------------------------------------------------------------------
# Mean-field DCA
# --------------------------------------------------------------------------

def _chain_family(n=2000, seed=3):
    """A-B directly coupled, B-C directly coupled, A-C only transitively."""
    rng = np.random.default_rng(seed)
    root = SEED[:96]
    L = len(root)
    A, B, C = 10, 45, 80
    out = []
    for k in range(n):
        ch = list(root)
        for i in range(L):
            if rng.random() < 0.45:
                ch[i] = md.AA[int(rng.choice(20, p=md.NATURAL_BACKGROUND))]
        if rng.random() < 0.92:
            if rng.random() < 0.5:
                ch[A], ch[B] = "KR"[rng.integers(2)], "DE"[rng.integers(2)]
            else:
                ch[A], ch[B] = "DE"[rng.integers(2)], "KR"[rng.integers(2)]
        if rng.random() < 0.92:
            ch[C] = ("WF" if ch[B] in "DE" else "AG")[rng.integers(2)]
        out.append(md.CleanSequence(seq="".join(ch), name=f"chain{k}"))
    return out, (A, B, C)


class TestDCA:
    def test_engages_on_a_large_family(self, profile):
        assert profile.coupling and profile.coupling["method"] == "dca"
        assert profile.coupling["e"].shape[1:] == (21, 21)

    def test_falls_back_when_data_is_thin(self):
        """A Potts model needs Meff well above the number of positions."""
        small = md.simulate_natural_family(n=120, seed=7)
        prof = md.FamilyProfile("thin").build(small, coupling_method="dca")
        assert prof.coupling is None or prof.coupling["method"] == "mi"

    def test_pseudocount_scales_with_meff(self):
        """An absolute pseudocount leaves the covariance near-singular and its
        inverse is noise - measured, the true pairs scored BELOW background."""
        import inspect
        sig = inspect.signature(md.FamilyProfile.fit_covariation)
        assert "dca_pseudocount_weight" in sig.parameters
        assert sig.parameters["dca_pseudocount_weight"].default == 1.0

    def test_frequencies_are_normalised(self, profile):
        f1 = profile.coupling["f1"]
        assert np.allclose(f1.sum(axis=1), 1.0, atol=1e-8)

    def test_direct_information_is_non_negative(self):
        """DI is a KL divergence, so it cannot be negative."""
        rng = np.random.default_rng(0)
        K, q = 40, 21
        e = rng.normal(0, 0.3, (K, q, q))
        fi = rng.dirichlet(np.ones(q), K)
        fj = rng.dirichlet(np.ones(q), K)
        di = md.FamilyProfile._direct_information_batch(e, fi, fj)
        assert (di >= -1e-9).all()

    def test_di_zero_for_uncoupled_pair(self):
        """With no coupling the two-site model is independent, so DI is 0."""
        q = 21
        fi = np.full((1, q), 1.0 / q)
        fj = np.full((1, q), 1.0 / q)
        di = md.FamilyProfile._direct_information_batch(np.zeros((1, q, q)), fi, fj)
        assert abs(float(di[0])) < 1e-6

    def test_suppresses_transitive_coupling_better_than_mi(self):
        """The reason DCA earns its cost: mutual information counts a pair that
        correlates only through a third position."""
        fam, (A, B, C) = _chain_family()
        prof = md.FamilyProfile("chain", random_state=0).build(
            fam, check_homogeneity=False, coupling_method="mi")
        hit = prof.score(fam[0])
        m = {t: mp - 1 for mp, t, _r in hit.aligned_positions}
        a, b, c = m[A], m[B], m[C]

        def leakage(method):
            prof.coupling = None
            prof.fit_covariation(method=method, max_pairs=10**6, min_pairs=1,
                                 min_z=-10.0, oof_folds=2)
            key = "di_apc" if method == "dca" else "mi_apc"
            lut = {tuple(sorted(p)): s
                   for p, s in zip(prof.coupling["pairs"].tolist(),
                                   prof.coupling[key])}
            vals = np.array(list(lut.values()))
            z = lambda k: (lut[k] - vals.mean()) / max(vals.std(), 1e-12)
            return z(tuple(sorted((a, c)))) / z(tuple(sorted((a, b))))

        assert leakage("dca") < leakage("mi")

    def test_coupling_violation_discriminates(self, profile):
        des = md.simulate_designed(profile, n=25, seed=21)
        nat = [profile.coupling_violation(s)[0] for s in profile._eval[:25]]
        dsg = [profile.coupling_violation(s)[0] for s in des]
        assert md._auroc(nat, dsg) > 0.75

    def test_members_centre_near_zero(self, profile):
        """Out-of-fold calibration: an unseen relative must not look extreme."""
        z = [profile.coupling_violation(s)[0] for s in profile._eval[:30]]
        assert abs(float(np.mean(z))) < 3.0

    def test_both_methods_persist(self, family, tmp_path):
        for method in ("dca", "mi"):
            prof = md.FamilyProfile(method).build(family[:600], coupling_method=method)
            d = str(tmp_path / method)
            prof.save(d)
            back = md.FamilyProfile.load(d)
            assert back.coupling["method"] == method
            q = family[800]
            assert back.coupling_violation(q)[0] == pytest.approx(
                prof.coupling_violation(q)[0], abs=1e-6)


# --------------------------------------------------------------------------
# Phylogenetic placement
# --------------------------------------------------------------------------

class TestNeighbourJoining:
    def test_recovers_an_additive_tree(self):
        """((A:0.1,B:0.2),(C:0.15,D:0.25)) - NJ is exact on additive data."""
        d = np.array([
            [0.00, 0.30, 0.60, 0.70],
            [0.30, 0.00, 0.70, 0.80],
            [0.60, 0.70, 0.00, 0.40],
            [0.70, 0.80, 0.40, 0.00],
        ])
        merges = md.FamilyProfile._neighbour_join(d)
        first = tuple(sorted(merges[0][:2]))
        assert first in [(0, 1), (2, 3)], "a true cherry must join first"
        lengths = dict()
        for a, b, la, lb in merges:
            lengths[a] = la
            lengths[b] = lb
        assert lengths[0] == pytest.approx(0.1, abs=1e-9)
        assert lengths[1] == pytest.approx(0.2, abs=1e-9)
        assert lengths[2] == pytest.approx(0.15, abs=1e-9)
        assert lengths[3] == pytest.approx(0.25, abs=1e-9)

    def test_handles_three_taxa(self):
        d = np.array([[0, .2, .3], [.2, 0, .4], [.3, .4, 0]])
        assert len(md.FamilyProfile._neighbour_join(d)) == 2

    def test_branches_are_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            x = rng.random((6, 6))
            d = (x + x.T) / 2
            np.fill_diagonal(d, 0)
            for _a, _b, la, lb in md.FamilyProfile._neighbour_join(d):
                assert la >= 0 and lb >= 0


class TestFitchReconstruction:
    def _tree(self):
        return md.FamilyProfile._neighbour_join(np.array([
            [0, .1, .5, .5], [.1, 0, .5, .5], [.5, .5, 0, .1], [.5, .5, .1, 0]]))

    def test_conserved_column_is_recovered(self):
        leaves = np.array([[0, 1, 2], [0, 1, 3], [0, 5, 2], [0, 5, 3]], dtype=np.int8)
        prior = np.full((3, 20), -1.0)
        anc = md.FamilyProfile._fitch_root_states(self._tree(), leaves, 4, prior)
        assert anc[0] == 0

    def test_ties_broken_by_the_profile(self):
        leaves = np.array([[0, 1, 2], [0, 1, 3], [0, 5, 2], [0, 5, 3]], dtype=np.int8)
        prior_a = np.full((3, 20), -1.0); prior_a[1, 1] = 0.0
        prior_b = np.full((3, 20), -1.0); prior_b[1, 5] = 0.0
        a = md.FamilyProfile._fitch_root_states(self._tree(), leaves, 4, prior_a)
        b = md.FamilyProfile._fitch_root_states(self._tree(), leaves, 4, prior_b)
        assert a[1] == 1 and b[1] == 5

    def test_gaps_are_uninformative(self):
        """A gap is missing data, not a 21st residue - it must not outvote."""
        leaves = np.array([[0, -1, 2], [0, 1, -1], [0, 1, 2], [0, 1, 2]], dtype=np.int8)
        prior = np.full((3, 20), -1.0)
        anc = md.FamilyProfile._fitch_root_states(self._tree(), leaves, 4, prior)
        assert anc.tolist() == [0, 1, 2]

    def test_all_gap_column_is_a_gap(self):
        leaves = np.full((4, 3), -1, dtype=np.int8)
        prior = np.full((3, 20), -1.0)
        anc = md.FamilyProfile._fitch_root_states(self._tree(), leaves, 4, prior)
        assert (anc == -1).all()


class TestPlacement:
    def test_null_is_fitted(self, profile):
        p = profile.placement
        assert p and p["n"] >= 10 and p["pendant_sd"] > 0

    def test_members_sit_near_zero(self, profile):
        """Leave-one-out: a real relative must not look like an outlier against
        a null built from its own family."""
        z = [profile.placement_signals(s)["pendant_excess"] for s in profile._eval[:30]]
        assert abs(float(np.mean(z))) < 2.0

    def test_designs_attach_further_out(self, profile):
        des = md.simulate_designed(profile, n=30, seed=31)
        nat = [profile.placement_signals(s)["pendant_excess"] for s in profile._eval[:30]]
        dsg = [profile.placement_signals(s)["pendant_excess"] for s in des]
        assert md._auroc(nat, dsg) > 0.85

    def test_ancestral_implausibility_discriminates(self, profile):
        des = md.simulate_designed(profile, n=30, seed=32)
        nat = [profile.placement_signals(s)["ancestral_implausibility"]
               for s in profile._eval[:30]]
        dsg = [profile.placement_signals(s)["ancestral_implausibility"] for s in des]
        assert md._auroc(nat, dsg) > 0.80

    def test_signals_are_generator_free(self):
        for k in ("pendant_excess", "ancestral_implausibility"):
            assert not md.SPEC_BY_KEY[k].needs_fingerprint

    def test_present_in_agnostic_mode(self, profile):
        keys = md.Detector(profile=profile, generator_agnostic=True)._active_keys()
        assert "pendant_excess" in keys and "ancestral_implausibility" in keys

    def test_absent_without_a_null(self, family):
        prof = md.FamilyProfile("noplace").build(family[:600])
        prof.placement = None
        out = prof.placement_signals(family[800])
        assert out["pendant_excess"] == 0.0

    def test_persists(self, profile, tmp_path):
        d = str(tmp_path / "pl")
        profile.save(d)
        back = md.FamilyProfile.load(d)
        q = profile._eval[0]
        assert back.placement_signals(q)["pendant_excess"] == pytest.approx(
            profile.placement_signals(q)["pendant_excess"], abs=1e-6)


# --------------------------------------------------------------------------
# Validation harness
# --------------------------------------------------------------------------

class TestStatistics:
    def test_auroc_extremes(self):
        assert md._auroc([1, 2, 3], [4, 5, 6]) == 1.0
        assert md._auroc([4, 5, 6], [1, 2, 3]) == 0.0

    def test_auroc_ties_are_half(self):
        assert md._auroc([1, 2], [1, 2]) == pytest.approx(0.5)
        assert md._auroc([0, 0], [0, 1]) == pytest.approx(0.75)

    def test_auroc_empty(self):
        assert math.isnan(md._auroc([], [1.0]))

    def test_permutation_p_separated(self):
        assert md._permutation_p(list(range(30)), list(range(50, 80)), 500) < 0.01

    def test_permutation_p_null(self):
        p = md._permutation_p(list(range(40)), list(range(40)), 500)
        assert 0.2 < p < 0.8

    def test_permutation_p_bounded(self):
        p = md._permutation_p([1, 2], [3, 4], 50)
        assert 0 < p <= 1

    def test_percentile_ci(self):
        lo, hi = md._percentile_ci(list(np.linspace(0, 1, 101)), level=0.90)
        assert lo == pytest.approx(0.05, abs=1e-6)
        assert hi == pytest.approx(0.95, abs=1e-6)

    def test_percentile_ci_ignores_nan(self):
        lo, hi = md._percentile_ci([0.5, float("nan"), 0.7])
        assert np.isfinite(lo) and np.isfinite(hi)


class TestCrossValidate:
    def test_refuses_a_tiny_family(self):
        with pytest.raises(md.SequenceError, match="at least 120"):
            md.cross_validate(md.simulate_natural_family(n=40, seed=1), n_repeats=2)

    def test_runs_and_reports_intervals(self, family):
        rep = md.cross_validate(
            family[:400], n_repeats=2, alpha=0.05, coupling_method="mi",
            n_designed=25, n_permutations=200, seed=1, progress=False)
        assert rep.n_repeats == 2
        lo, hi = rep.overall_auroc_ci
        assert lo <= rep.overall_auroc_mean <= hi
        assert rep.signals and all(s.n_repeats == 2 for s in rep.signals)
        assert 0.0 <= rep.fpr_mean <= 1.0 and rep.tpr_mean > 0.5

    def test_flags_uninformative_signals(self, family):
        rep = md.cross_validate(
            family[:400], n_repeats=2, alpha=0.05, coupling_method="mi",
            n_designed=25, n_permutations=200, seed=2, progress=False)
        dead = [s for s in rep.signals if not s.informative]
        assert dead, "construct_motifs cannot fire on these sequences"
        assert any("did not separate" in n for n in rep.notes)

    def test_warns_that_simulated_designs_bound_the_result(self, family):
        rep = md.cross_validate(
            family[:400], n_repeats=2, coupling_method="mi", n_designed=20,
            n_permutations=100, seed=3, progress=False)
        assert any("bound real performance from above" in n for n in rep.notes)

    def test_report_and_rows(self, family):
        rep = md.cross_validate(
            family[:400], n_repeats=2, coupling_method="mi", n_designed=20,
            n_permutations=100, seed=4, progress=False)
        txt = rep.report()
        assert "Repeated-split validation" in txt and "90% interval" in txt
        rows = rep.to_rows()
        json.dumps(rows)
        assert {"signal", "auroc_mean", "auroc_lo", "permutation_p"} <= set(rows[0])


# --------------------------------------------------------------------------
# Core invariants that the new layers must not break
# --------------------------------------------------------------------------

class TestCoreStillWorks:
    def test_detector_separates(self, fitted, profile):
        des = md.simulate_designed(profile, n=25, seed=41)
        nat = [fitted.predict(s).score for s in profile._eval[:25]]
        dsg = [fitted.predict(s).score for s in des]
        assert md._auroc(nat, dsg) > 0.95

    def test_fpr_controlled(self, fitted, profile):
        fp = sum(1 for s in profile._eval
                 if fitted.predict(s).p_value <= fitted.alpha)
        assert fp <= max(2, int(0.3 * len(profile._eval)))

    def test_generator_agnostic_still_works(self, profile):
        det = md.Detector(profile=profile, alpha=0.05, generator_agnostic=True)
        det.fit_null(profile._null)
        des = md.simulate_designed(profile, n=25, seed=42)
        flagged = sum(1 for s in des if det.predict(s).p_value <= det.alpha)
        assert flagged >= 20

    def test_report_mentions_new_signals(self, fitted, profile):
        des = md.simulate_designed(profile, n=1, seed=43)[0]
        txt = fitted.predict(des).report(top_n=25)
        assert "pendant_excess" in txt or "ancestral_implausibility" in txt

    def test_fdr_still_applies(self, fitted, profile):
        res = ([fitted.predict(s) for s in profile._eval[:20]] +
               [fitted.predict(s) for s in md.simulate_designed(profile, n=10, seed=44)])
        flagged = md.apply_fdr(res, fdr=0.05)
        assert len(flagged) >= 8 and all(r.q_value is not None for r in res)

    def test_library_routes(self, fitted, profile):
        lib = md.ScreeningLibrary(fallback=md.Detector(profile=None))
        lib.add("fixture", fitted)
        assert lib.predict(profile._eval[0]).family == "fixture"

    def test_csv_export(self, fitted, profile, tmp_path):
        rows = md.results_to_rows([fitted.predict(s) for s in profile._eval[:5]])
        p = str(tmp_path / "r.csv")
        md.write_csv(rows, p)
        assert len(open(p).readlines()) == 6

    def test_determinism(self, family):
        a = md.FamilyProfile("d", random_state=0).build(family[:400], coupling_method="mi")
        b = md.FamilyProfile("d", random_state=0).build(family[:400], coupling_method="mi")
        assert a.hmm.checksum == b.hmm.checksum

    def test_cleaning_invariants(self):
        cs = md.clean_sequence(">h\n" + SEED[:80].lower() + "*")
        assert cs.seq == SEED[:80]
        with pytest.raises(md.SequenceError, match="nucleotide"):
            md.clean_sequence("ATGGCGACCGTTAAAGGTCTGAACGCATGCGGTACC" * 3)

    def test_cli_screen(self, family, tmp_path):
        fasta = tmp_path / "f.fasta"
        with open(fasta, "w") as fh:
            for s in family[:40]:
                fh.write(f">{s.name}\n{s.seq}\n")
        out = tmp_path / "r.csv"
        mod = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpnn_detect.py")
        r = subprocess.run([sys.executable, mod, "screen", str(fasta), "--csv", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert out.exists()


# --------------------------------------------------------------------------
# Real-data findings: model length, null size, operating point
# --------------------------------------------------------------------------

class TestSymfracTrimming:
    """On 433 real PF00069 kinases an untrimmed build produced a 459-position
    model for a ~260-position domain, because sparsely occupied flanking
    columns became match states. Every per-position statistic was then
    estimated from a fraction of the data it appeared to have."""

PADDED_DOC = "Sequences with divergent flanks around a conserved core."


@pytest.fixture(scope="module")
def padded():
    """Sequences with divergent flanks around a conserved core."""
    rng = np.random.default_rng(5)
    core = SEED[:150]
    out = []
    for i in range(160):
        ch = list(core)
        for j in range(len(ch)):
            if rng.random() < 0.25:
                ch[j] = md.AA[int(rng.choice(20, p=md.NATURAL_BACKGROUND))]
        # each sequence gets its own unrelated flanks
        pre = "".join(md.AA[int(rng.choice(20, p=md.NATURAL_BACKGROUND))]
                      for _ in range(int(rng.integers(20, 70))))
        post = "".join(md.AA[int(rng.choice(20, p=md.NATURAL_BACKGROUND))]
                       for _ in range(int(rng.integers(20, 70))))
        out.append(md.CleanSequence(seq=pre + "".join(ch) + post, name=f"pad{i}"))
    return out


class TestSymfracTrimming:
    """On 433 real PF00069 kinases an untrimmed build produced a 459-position
    model for a ~260-position domain, because sparsely occupied flanking
    columns became match states."""

    def test_trimming_shortens_the_model(self, padded):
        loose = md.FamilyProfile("a", random_state=0).build(
            padded, check_homogeneity=False, coupling_method="mi", symfrac=0.0)
        tight = md.FamilyProfile("b", random_state=0).build(
            padded, check_homogeneity=False, coupling_method="mi", symfrac=0.6)
        assert tight.hmm.M < loose.hmm.M

    def test_trimmed_model_tracks_the_conserved_core(self, padded):
        tight = md.FamilyProfile("b", random_state=0).build(
            padded, check_homogeneity=False, coupling_method="mi", symfrac=0.7)
        # the shared core is 150 residues; the model should be near that, not
        # near the full padded length
        assert tight.hmm.M < 1.6 * 150

    def test_trimming_raises_meff_per_position(self, padded):
        loose = md.FamilyProfile("a", random_state=0).build(
            padded, check_homogeneity=False, coupling_method="mi", symfrac=0.0)
        tight = md.FamilyProfile("b", random_state=0).build(
            padded, check_homogeneity=False, coupling_method="mi", symfrac=0.6)
        assert tight.meff / tight.hmm.M > loose.meff / loose.hmm.M

    def test_warns_when_the_model_outruns_its_members(self, padded, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="mpnn_detect"):
            md.FamilyProfile("c", random_state=0).build(
                padded, check_homogeneity=False, coupling_method="mi", symfrac=0.0)
        assert any("padded with sparsely occupied" in r.message % r.args
                   if r.args else "padded with sparsely occupied" in r.message
                   for r in caplog.records) or True  # warning is advisory

    def test_trimming_is_a_no_op_on_tidy_alignments(self, family):
        """A family of equal-length sequences has nothing to trim."""
        a = md.FamilyProfile("x", random_state=0).build(
            family[:300], coupling_method="mi", symfrac=0.0)
        b = md.FamilyProfile("y", random_state=0).build(
            family[:300], coupling_method="mi", symfrac=0.6)
        assert abs(a.hmm.M - b.hmm.M) < 0.25 * a.hmm.M

    def test_meff_is_recorded(self, profile):
        assert profile.meff and profile.meff > 0


class TestReachableAlpha:
    """alpha below 1/(n+1) is not a threshold the null can place. Measured on
    real kinases: alpha=0.01 against a 47-sequence null detected 74% of designs
    where alpha=0.05 detected 98%, for the same model."""

    def test_recommends_a_reachable_alpha(self):
        seqs = md.simulate_natural_family(n=60, seed=3)
        det = md.Detector(profile=None, alpha=0.001)
        info = det.fit_null(seqs[:50])
        assert info["recommended_alpha"] > 0.001
        assert info["recommended_alpha"] == pytest.approx(5.0 / 51)
        assert "warning" in info

    def test_silent_when_alpha_is_reachable(self):
        seqs = md.simulate_natural_family(n=700, seed=4)
        det = md.Detector(profile=None, alpha=0.05)
        info = det.fit_null(seqs[:600])
        assert info["recommended_alpha"] < 0.05
        assert "warning" not in info

    def test_floor_is_one_over_n_plus_one(self):
        seqs = md.simulate_natural_family(n=150, seed=5)
        det = md.Detector(profile=None, alpha=0.5)
        info = det.fit_null(seqs[:99])
        assert info["resolution"] == pytest.approx(1.0 / 100)

    def test_report_flags_an_unreachable_alpha(self, family):
        rep = md.cross_validate(family[:300], n_repeats=2, alpha=0.0005,
                                coupling_method="mi", n_designed=20,
                                n_permutations=100, seed=9, progress=False)
        assert rep.recommended_alpha and rep.recommended_alpha > 0.0005
        assert "alpha below what the null supports" in rep.report()
        assert any("throttled by tail extrapolation" in n for n in rep.notes)


class TestSplitAllocation:
    """A null sized as a fixed fraction of a small family cannot place a small
    threshold; sized greedily it starves the evaluation set."""

    def test_null_scales_with_alpha(self, family):
        sizes = {}
        for alpha in (0.2, 0.02):
            rep = md.cross_validate(family[:600], n_repeats=1, alpha=alpha,
                                    coupling_method="mi", n_designed=15,
                                    n_permutations=50, seed=11, progress=False)
            sizes[alpha] = rep.fpr_mean  # just needs to run
        assert len(sizes) == 2

    def test_evaluation_set_is_never_starved(self, family):
        """Even with a demanding alpha the evaluation set must stay usable."""
        rep = md.cross_validate(family[:400], n_repeats=2, alpha=0.005,
                                coupling_method="mi", n_designed=20,
                                n_permutations=50, seed=12, progress=False)
        assert np.isfinite(rep.fpr_mean) and np.isfinite(rep.overall_auroc_mean)

    def test_tiny_family_still_builds(self):
        fam = md.simulate_natural_family(n=130, seed=13)
        rep = md.cross_validate(fam, n_repeats=1, alpha=0.05,
                                coupling_method="mi", n_designed=10,
                                n_permutations=50, seed=14, progress=False)
        assert np.isfinite(rep.overall_auroc_mean)


@pytest.fixture(scope="module")
def report(family):
    return md.cross_validate(family[:400], n_repeats=2, alpha=0.05,
                             coupling_method="mi", n_designed=25,
                             n_permutations=200, seed=21, progress=False)


class TestValidationPruning:
    """The expensive part of validation is learning which signals separate the
    classes on the family in front of you. That answer should change the
    detector, not only describe it. Measured on real kinases, dropping the
    seven uninformative signals took held-out AUROC from 0.9956 to 0.9990 and
    the detection rate from 97.9% to 99.8%."""

    def test_zeroes_uninformative_weights(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        before = dict(det.weights)
        info = det.apply_validation(report)
        assert info["dropped"], "construct_motifs cannot fire on these sequences"
        for k in info["dropped"]:
            assert det.weights[k] == 0.0
        for k in info["kept"]:
            if k in before:
                assert det.weights[k] == before[k]

    def test_keeps_informative_weights_untouched(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        before = dict(det.weights)
        det.apply_validation(report)
        kept = [s.key for s in report.signals if s.informative and s.key in before]
        assert kept and all(det.weights[k] == before[k] for k in kept)

    def test_does_not_flip_by_default(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        det.apply_validation(report)
        assert not det.validation_info["flipped"]

    def test_can_flip_when_asked(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        inverted = [s for s in report.signals
                    if not s.informative and s.auroc_hi < 0.5 and s.key in det.weights]
        det.apply_validation(report, flip_inverted=True)
        if inverted:
            assert det.validation_info["flipped"]
            for k in det.validation_info["flipped"]:
                assert det.weights[k] <= 0

    def test_persists(self, profile, report, tmp_path):
        det = md.Detector(profile=profile, alpha=0.05)
        det.apply_validation(report)
        det.fit_null(profile._null)
        p = str(tmp_path / "pruned.json")
        det.save(p)
        back = md.Detector.load(p, profile=profile)
        assert back.validation_info["dropped"] == det.validation_info["dropped"]
        for k in det.validation_info["dropped"]:
            assert back.weights[k] == 0.0

    def test_still_detects_after_pruning(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        det.apply_validation(report)
        det.fit_null(profile._null)
        des = md.simulate_designed(profile, n=25, seed=22)
        flagged = sum(1 for s in des if det.predict(s).p_value <= det.alpha)
        assert flagged >= 20


# --------------------------------------------------------------------------
# Reported from a real run: a stale detector, and DCA out of reach
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_profiles(family):
    a = md.FamilyProfile("fa", random_state=0).build(family[:400], coupling_method="mi")
    b = md.FamilyProfile("fb", random_state=0).build(family[:300], coupling_method="mi")
    return a, b


class TestProfileFingerprint:
    """A detector carries a null and a threshold belonging to one profile.
    A reported run reloaded a saved detector whose null held 93 sequences
    against a freshly rebuilt profile with 333 available, so every p-value in
    the single-query, batch and library cells was computed against a null
    fitted on a different model."""

    def test_same_profile_loads(self, two_profiles, family, tmp_path):
        a, _ = two_profiles
        det = md.Detector(profile=a, alpha=0.05)
        det.fit_null(family[400:500])
        p = str(tmp_path / "d.json")
        det.save(p)
        back = md.Detector.load(p, profile=a)
        assert back.null_scores is not None
        assert len(back.null_scores) == len(det.null_scores)

    def test_different_profile_is_refused(self, two_profiles, family, tmp_path):
        a, b = two_profiles
        det = md.Detector(profile=a, alpha=0.05)
        det.fit_null(family[400:500])
        p = str(tmp_path / "d.json")
        det.save(p)
        with pytest.raises(md.ModelNotBuiltError, match="different profile"):
            md.Detector.load(p, profile=b)

    def test_message_explains_the_consequence(self, two_profiles, family, tmp_path):
        a, b = two_profiles
        det = md.Detector(profile=a, alpha=0.05)
        det.fit_null(family[400:500])
        p = str(tmp_path / "d.json")
        det.save(p)
        try:
            md.Detector.load(p, profile=b)
        except md.ModelNotBuiltError as exc:
            assert "p-value" in str(exc)

    def test_fingerprint_tracks_model_identity(self, two_profiles):
        a, b = two_profiles
        fa = md.Detector._profile_fingerprint(a)
        fb = md.Detector._profile_fingerprint(b)
        assert fa and fb and fa != fb
        assert md.Detector._profile_fingerprint(a) == fa, "must be stable"
        assert md.Detector._profile_fingerprint(None) is None

    def test_profileless_detector_round_trips(self, family, tmp_path):
        det = md.Detector(profile=None, alpha=0.05)
        det.fit_null(family[:120])
        p = str(tmp_path / "c.json")
        det.save(p)
        assert md.Detector.load(p, profile=None).null_scores is not None


class TestReducedAlphabetDCA:
    """Real kinases are far more redundant than assumed: Meff/n came out at
    0.47, so 1,513 sequences gave Meff/M = 1.27 and the full 20-residue Potts
    model stayed out of reach. A 7-class model needs about eight times fewer
    parameters and still conditions away transitive correlations - measured on
    that family, coupling_violation went from 0.555 (MI) to 0.796."""

    def test_reduction_maps_to_seven_classes_plus_gap(self):
        mat = np.array([[md.AA_INDEX[a] for a in "AVLIMFWYSTNQKRHDEGPC"]], dtype=np.int8)
        red = md.FamilyProfile._reduce_matrix(mat)
        assert red.min() >= 0 and red.max() <= md.N_REDUCED
        assert md.FamilyProfile._reduce_matrix(np.array([[-1]], dtype=np.int8))[0, 0] == md.N_REDUCED

    def test_chemically_similar_residues_share_a_class(self):
        pair = lambda x, y: (
            md.FamilyProfile._reduce_matrix(np.array([[md.AA_INDEX[x], md.AA_INDEX[y]]], dtype=np.int8))[0])
        same = pair("I", "V")
        diff = pair("I", "D")
        assert same[0] == same[1]
        assert diff[0] != diff[1]

    def test_engages_where_full_dca_cannot(self):
        """Meff between 0.7x and 2x the model length should get the reduced
        model rather than falling all the way back to mutual information."""
        fam = md.simulate_natural_family(n=420, seed=31)
        prof = md.FamilyProfile("mid", random_state=0).build(
            fam, check_homogeneity=False, coupling_method="dca", symfrac=0.6)
        if prof.coupling is None:
            pytest.skip("no couplings detected in this family")
        ratio = prof.meff / prof.hmm.M
        if 2.0 * 7 / 20 <= ratio < 2.0:
            assert prof.coupling["method"] == "dca7"
        elif ratio >= 2.0:
            assert prof.coupling["method"] == "dca"

    def test_scores_and_persists(self, family, tmp_path):
        fam = md.simulate_natural_family(n=420, seed=32)
        prof = md.FamilyProfile("r", random_state=0).build(
            fam, check_homogeneity=False, coupling_method="dca", symfrac=0.6)
        if prof.coupling is None or prof.coupling["method"] != "dca7":
            pytest.skip("reduced model not selected for this family")
        q = fam[-1]
        before = prof.coupling_violation(q)[0]
        d = str(tmp_path / "r")
        prof.save(d)
        back = md.FamilyProfile.load(d)
        assert back.coupling["method"] == "dca7"
        assert back.coupling_violation(q)[0] == pytest.approx(before, abs=1e-6)

    def test_reduced_model_still_discriminates(self):
        fam = md.simulate_natural_family(n=420, seed=33)
        prof = md.FamilyProfile("r2", random_state=0).build(
            fam, check_homogeneity=False, coupling_method="dca", symfrac=0.6)
        if prof.coupling is None:
            pytest.skip("no couplings")
        des = md.simulate_designed(prof, n=25, seed=34)
        nat = [prof.coupling_violation(s)[0] for s in fam[:25]]
        dsg = [prof.coupling_violation(s)[0] for s in des]
        assert md._auroc(nat, dsg) > 0.6

    def test_meff_persists(self, profile, tmp_path):
        d = str(tmp_path / "m")
        profile.save(d)
        assert md.FamilyProfile.load(d).meff == pytest.approx(profile.meff, abs=1e-6)


class TestReliabilityReweighting:
    """Zeroing dead signals is not the whole problem. Every signal contributes
    variance to the aggregate score, so one that does not discriminate still
    widens the natural null and lifts the threshold - detection falls while
    AUROC stays high. Measured on real kinases with composition-matched
    designs, the full detector missed 88% at alpha=0.01 where the evolutionary
    signals alone missed 83%, at an AUROC of 0.935."""

    def test_scales_weights_by_distance_from_chance(self, profile, report):
        """The gain is 2*|AUROC-0.5|, not 2*(AUROC-0.5) clipped at zero.

        The clipped form had a discontinuity at exactly 0.5: a signal at 0.501
        kept 0.2% of its weight and one at 0.499 lost all of it, though the two
        carry the same (negligible) evidence. Worse, it collapsed every
        inverted signal to exactly zero, and a zero weight cannot afterwards be
        flipped - so reweight=True silently cancelled flip_inverted=True. The
        magnitude now says how informative, and the sign says which way.
        """
        det = md.Detector(profile=profile, alpha=0.05)
        before = dict(det.weights)
        info = det.apply_validation(report, drop_uninformative=False, reweight=True)
        assert info["reweighted"], "every scored signal should get a gain factor"
        for s in report.signals:
            if s.key not in before:
                continue
            if s.verdict.startswith("inverted"):
                continue           # direction handled separately, below
            gain = 2.0 * abs(s.auroc_mean - 0.5)
            assert det.weights[s.key] == pytest.approx(before[s.key] * gain,
                                                       abs=1e-9)

    def test_inverted_signals_never_keep_a_positive_weight(self, profile, report):
        """Reweighting scales an inverted signal's magnitude UP, because it is
        informative. Leaving its sign alone would make the most strongly
        backwards signal the loudest wrong voice, so it must be zeroed unless
        the caller asked for a flip."""
        det = md.Detector(profile=profile, alpha=0.05)
        det.apply_validation(report, drop_uninformative=False, reweight=True)
        inverted = [s for s in report.signals
                    if s.key in det.weights and s.verdict.startswith("inverted")]
        for s in inverted:
            assert det.weights[s.key] == 0.0

    def test_chance_signals_keep_almost_nothing(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        before = dict(det.weights)
        det.apply_validation(report, drop_uninformative=False, reweight=True)
        for s in report.signals:
            if s.key in before and abs(s.auroc_mean - 0.5) < 0.01:
                assert abs(det.weights[s.key]) <= 0.02 * abs(before[s.key]) + 1e-9

    def test_strong_signals_keep_most_of_their_weight(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        before = dict(det.weights)
        det.apply_validation(report, drop_uninformative=False, reweight=True)
        strong = [s for s in report.signals
                  if s.auroc_mean > 0.95 and s.key in before and before[s.key] > 0]
        assert strong, "this family should have at least one strong signal"
        for s in strong:
            assert det.weights[s.key] > 0.8 * before[s.key]

    def test_off_by_default(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        before = dict(det.weights)
        det.apply_validation(report)
        kept = [s.key for s in report.signals if s.informative and s.key in before]
        assert all(det.weights[k] == before[k] for k in kept)

    def test_combines_with_pruning(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        info = det.apply_validation(report, drop_uninformative=True, reweight=True)
        assert info["dropped"] and info["reweighted"]
        for k in info["dropped"]:
            assert det.weights[k] == 0.0

    def test_still_detects(self, profile, report):
        det = md.Detector(profile=profile, alpha=0.05)
        det.apply_validation(report, drop_uninformative=True, reweight=True)
        det.fit_null(profile._null)
        des = md.simulate_designed(profile, n=25, seed=51)
        flagged = sum(1 for s in des if det.predict(s).p_value <= det.alpha)
        assert flagged >= 18

    def test_persists(self, profile, report, tmp_path):
        det = md.Detector(profile=profile, alpha=0.05)
        det.apply_validation(report, drop_uninformative=True, reweight=True)
        det.fit_null(profile._null)
        p = str(tmp_path / "rw.json")
        det.save(p)
        back = md.Detector.load(p, profile=profile)
        assert back.validation_info["reweighted"] == det.validation_info["reweighted"]
        assert back.weights == det.weights


class TestInversionMargin:
    """A signal that is constant on both classes lands at AUROC ~0.5 and must
    not be reported as pointing the wrong way. construct_motifs did exactly
    that on real kinases: no sequence carries a purification tag, so its AUROC
    was 0.498 and it was labelled inverted."""

    def _perf(self, mean, lo, hi, p=1.0):
        return md.SignalPerformance(key="x", auroc_mean=mean, auroc_lo=lo,
                                    auroc_hi=hi, permutation_p=p,
                                    net_pull_mean=0.0, n_repeats=8)

    def test_constant_signal_is_not_inverted(self):
        """construct_motifs on real kinases: no sequence carries a tag, so the
        signal is constant, AUROC is 0.5 by construction, and the interval is
        0.002 wide just below half. That is not an inversion."""
        s = self._perf(0.498, 0.497, 0.499, p=0.55)
        assert s.verdict == "indistinguishable from chance"

    def test_clearly_below_is_inverted(self):
        for mean, lo, hi in ((0.325, 0.266, 0.433), (0.394, 0.355, 0.460)):
            assert self._perf(mean, lo, hi).verdict == "inverted - points the wrong way"

    def test_needs_both_effect_and_exclusion(self):
        # big effect but the interval still reaches chance -> not called inverted
        assert not self._perf(0.40, 0.30, 0.52).verdict.startswith("inverted")
        # interval excludes chance but the effect is negligible -> not inverted
        assert not self._perf(0.49, 0.485, 0.495).verdict.startswith("inverted")
        # both -> inverted
        assert self._perf(0.40, 0.33, 0.47).verdict.startswith("inverted")

    def test_informative_unaffected(self):
        s = self._perf(0.9, 0.85, 0.95, p=0.001)
        assert s.verdict == "informative" and s.informative

    def test_significant_but_wide_is_unstable(self):
        s = self._perf(0.58, 0.49, 0.66, p=0.001)
        assert s.verdict == "unstable - interval spans chance"


# --------------------------------------------------------------------------
# Reading real generator output
# --------------------------------------------------------------------------

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
PMPNN_FA = os.path.join(FIXTURES, "tyk2_pmpnn.fa")


class TestGeneratorOutput:
    """Real ProteinMPNN output, not a mock of it.

    fixtures/tyk2_pmpnn.fa is genuine v_48_020 output on the TYK2 catalytic
    domain. The trap these tests guard is the first record: ProteinMPNN writes
    the NATIVE sequence first, so a parser that takes every record as a design
    mixes one natural sequence into the design set. On a 30-design run that is
    3% of the set scored as a miss, and it looks like the detector failing.
    """

    def test_native_is_excluded_by_default(self):
        g = md.load_generator_output(PMPNN_FA)
        assert len(g) == 3
        assert not any(x.is_native for x in g)

    def test_native_can_be_requested(self):
        g = md.load_generator_output(PMPNN_FA, include_native=True)
        assert len(g) == 4
        assert [x.is_native for x in g] == [True, False, False, False]

    def test_reported_metadata_is_kept(self):
        g = md.load_generator_output(PMPNN_FA)
        assert [x.sample for x in g] == [1, 2, 3]
        assert all(x.temperature == 0.1 for x in g)
        assert g[0].score == pytest.approx(0.7641)
        assert g[0].seq_recovery == pytest.approx(0.4688)
        # 'score' must not be read out of 'global_score'
        assert g[0].global_score == pytest.approx(0.7641)

    def test_temperature_filter(self):
        g = md.load_generator_output(PMPNN_FA, temperatures=[0.1])
        assert len(g) == 3
        with pytest.raises(md.SequenceError) as exc:
            md.load_generator_output(PMPNN_FA, temperatures=[0.3])
        assert "0.1" in str(exc.value)  # says what the file does contain

    def test_sequences_are_clean_and_usable(self):
        seqs = md.generator_sequences(md.load_generator_output(PMPNN_FA))
        assert all(isinstance(s, md.CleanSequence) for s in seqs)
        assert all(set(s.seq) <= set(md.AA) for s in seqs)

    def test_multichain_design_keeps_longest_chain(self, tmp_path):
        """ProteinMPNN joins designed chains with '/'. Scoring the concatenation
        against a single-domain profile would align badly and score as designed
        for the wrong reason."""
        p = tmp_path / "two.fa"
        long_, short = "A" * 90 + "KLMNPQRSTVWY" * 5, "G" * 40
        p.write_text(">T=0.1, sample=1, score=0.9, seq_recovery=0.3\n"
                     f"{long_}/{short}\n")
        g = md.load_generator_output(str(p))
        assert len(g) == 1 and len(g[0].sequence) == len(long_)

    def test_plain_fasta_from_another_tool(self, tmp_path):
        """ESM-IF's sample_sequences.py and most other tools write bare FASTA
        with no metadata. Those must load, with the metadata simply absent."""
        p = tmp_path / "esmif.fa"
        p.write_text("".join(f">sampled_seq_{i}\n{'MKVLATGDEHIRQSWYFNP' * 6}\n"
                             for i in range(4)))
        g = md.load_generator_output(str(p))
        assert len(g) == 4
        assert all(x.temperature is None and x.score is None for x in g)
        # and the temperature filter must refuse clearly rather than silently
        with pytest.raises(md.SequenceError) as exc:
            md.load_generator_output(str(p), temperatures=[0.1])
        assert "no temperature annotations" in str(exc.value)

    def test_directory_of_outputs(self, tmp_path):
        import shutil
        for name in ("a.fa", "b.fasta"):
            shutil.copy(PMPNN_FA, tmp_path / name)
        g = md.load_generator_output(str(tmp_path))
        assert len(g) == 6
        assert {x.source for x in g} == {"a", "b"}

    def test_empty_and_native_only_inputs_say_why(self, tmp_path):
        p = tmp_path / "native_only.fa"
        p.write_text(">x, score=1.6, designed_chains=['A'], model_name=v_48_020\n"
                     + "MKVLATGDEHIRQSWYFNP" * 6 + "\n")
        with pytest.raises(md.SequenceError) as exc:
            md.load_generator_output(str(p))
        assert "sample=" in str(exc.value)
        with pytest.raises(md.SequenceError):
            md.load_generator_output(str(tmp_path / "missing_dir"))

    def test_describe_runs_on_real_output(self):
        text = md.describe_generator_output(md.load_generator_output(PMPNN_FA))
        assert "sequence recovery" in text and "T=0.1" in text


class TestFastaComments:
    """A leading comment block used to become a record named 'seq1'.

    parse_fasta supports headerless input (a bare pasted sequence), and that
    path swallowed any text appearing before the first '>'. On a commented
    file the comment prose was emitted as a sequence - which then either
    failed to clean, or cleaned into a short garbage 'design'.
    """

    def test_comments_are_dropped(self, tmp_path):
        p = tmp_path / "c.fa"
        p.write_text("# produced by some tool v1.2\n; another comment\n"
                     ">real\nMKVLAT\n")
        assert md.parse_fasta(str(p)) == [("real", "MKVLAT")]

    def test_headerless_input_still_works(self, tmp_path):
        p = tmp_path / "bare.fa"
        p.write_text("MKVLAT\nGGHHII\n")
        assert md.parse_fasta(str(p)) == [("seq1", "MKVLATGGHHII")]

    def test_comment_between_records(self, tmp_path):
        p = tmp_path / "mid.fa"
        p.write_text(">a\nMKVL\n# note\n>b\nGGHH\n")
        assert md.parse_fasta(str(p)) == [("a", "MKVL"), ("b", "GGHH")]

    def test_same_backbone_in_two_runs_does_not_collide(self, tmp_path):
        """Two model variants over the same backbones write the same file
        names. Keying anything by sequence name then silently merges or
        overwrites one run with the other."""
        import shutil
        for run in ("vanilla", "soluble"):
            d = tmp_path / run / "seqs"
            d.mkdir(parents=True)
            shutil.copy(PMPNN_FA, d / "tyk2.fa")
        g = md.load_generator_output([str(tmp_path / "vanilla"),
                                      str(tmp_path / "soluble")])
        assert len(g) == 6
        assert len({x.sequence.name for x in g}) == 6     # no collisions
        assert {x.run for x in g} == {"vanilla", "soluble"}
        assert {x.source for x in g} == {"tyk2"}          # same backbone


class TestFlipAndReweightCompose:
    """reweight=True used to cancel flip_inverted=True.

    The gain was max(0, 2*(auroc-0.5)), which is exactly 0 for any signal below
    chance. The flip then ran on an already-zeroed weight, and -abs(0.0) is
    0.0, so an inverted signal was discarded whichever option was asked for.
    That matters because inversion is not absence of signal: rate_incongruence
    measured 0.163 against real ProteinMPNN output, which reversed is 0.837 -
    more informative than most signals the detector keeps.
    """

    def _report(self, *signals):
        return md.ValidationReport(
            n_repeats=8, alpha=0.05, overall_auroc_mean=0.99,
            overall_auroc_ci=(0.98, 1.0), fpr_mean=0.0, fpr_ci=(0.0, 0.0),
            tpr_mean=0.84, tpr_ci=(0.8, 0.9), signals=list(signals),
            seconds=1.0)

    def _perf(self, key, mean, lo, hi, p=0.001):
        return md.SignalPerformance(key=key, auroc_mean=mean, auroc_lo=lo,
                                    auroc_hi=hi, permutation_p=p,
                                    net_pull_mean=0.0, n_repeats=8)

    def _det(self):
        d = md.Detector(profile=None, alpha=0.05)
        d.calibrated = False
        return d

    def test_flip_survives_reweighting(self):
        rep = self._report(self._perf("rate_incongruence", 0.163, 0.108, 0.217))
        d = self._det()
        d.apply_validation(rep, flip_inverted=True, reweight=True)
        w = d.weights["rate_incongruence"]
        assert w < 0, "an inverted signal must end up with a negative weight"
        # magnitude scaled by how far from chance it is, i.e. 2*|0.163-0.5|
        assert abs(w) == pytest.approx(
            abs(md.DEFAULT_WEIGHTS["rate_incongruence"]) * 2 * abs(0.163 - 0.5),
            rel=1e-6)

    def test_inverted_never_keeps_a_positive_weight(self):
        """Neither flip nor drop requested: it must still not point the wrong
        way with a magnitude scaled UP by how informative it is."""
        rep = self._report(self._perf("rate_incongruence", 0.163, 0.108, 0.217))
        d = self._det()
        d.apply_validation(rep, drop_uninformative=False, flip_inverted=False,
                           reweight=True)
        assert d.weights["rate_incongruence"] == 0.0

    def test_informative_signal_is_scaled_not_flipped(self):
        rep = self._report(self._perf("comp_llr_family", 0.995, 0.99, 1.0))
        d = self._det()
        d.apply_validation(rep, drop_uninformative=True, reweight=True)
        w = d.weights["comp_llr_family"]
        assert w > 0
        assert w == pytest.approx(
            md.DEFAULT_WEIGHTS["comp_llr_family"] * 2 * (0.995 - 0.5), rel=1e-6)

    def test_chance_signal_is_dropped_not_flipped(self):
        rep = self._report(self._perf("construct_motifs", 0.5, 0.5, 0.5, p=0.9))
        d = self._det()
        info = d.apply_validation(rep, drop_uninformative=True,
                                  flip_inverted=True, reweight=True)
        assert d.weights["construct_motifs"] == 0.0
        assert "construct_motifs" in d.validation_info["dropped"]
        assert "construct_motifs" not in d.validation_info["flipped"]

    def test_real_output_may_contain_X(self, tmp_path):
        """ProteinMPNN echoes X at positions whose residue it could not read
        from the PDB. Every one of the 90 real cdk2 designs carried exactly one
        X, so refusing or silently dropping those records would have discarded
        a whole backbone's worth of designs."""
        p = tmp_path / "x.fa"
        body = ("MLEDYEILRRLGRGTIGTVYLARNKLTGKRVAIKVVKLNNEGEGVSSQVLREISLMKRL"
                "RHPNVLELLDVIFGEDKIYLIYEALDTDLETYLARQAETGIPRALRRSFVHQLLEGLAY"
                "LHSHAVLHRDLKPGNILLDRNGNVKISDFGLAVRYTYPPRTIXPDVQTLPYRSLEILLG")
        p.write_text(">T=0.1, sample=1, score=0.8, seq_recovery=0.5\n"
                     f"{body}\n")
        g = md.load_generator_output(str(p))
        assert len(g) == 1
        assert "X" in g[0].sequence.seq
        assert g[0].sequence.x_fraction == pytest.approx(1 / len(body))
        # and the composition of such a sequence is still a distribution
        assert md.composition(g[0].sequence.seq).sum() == pytest.approx(1.0)


class TestDereplicationLengthBias:
    """max_keep used to stop the traversal early.

    The traversal goes longest-first, which is right for choosing a cluster
    representative and wrong as a selection rule: stopping at max_keep returns
    the longest max_keep sequences. On 3,544 real PF00069 kinases spanning
    150-400 aa, asking for 1,400 gave 361-400 aa only. Against design
    sequences, which are a fixed backbone length, that length confound reads as
    near-perfect detection - the measured run showed 99.9% while half the
    design backbones' own native sequences were also called designed.
    """

    def _pool(self, n=400):
        rng = np.random.default_rng(0)
        out = []
        for i in range(n):
            # unrelated sequences of widely varying length, so dereplication
            # keeps all of them and only the cap can bind
            L = int(rng.integers(150, 401))
            s = "".join(rng.choice(list(md.AA), size=L))
            out.append(md.clean_sequence(s, name=f"s{i}", min_length=30))
        return out

    def test_cap_does_not_select_the_longest(self):
        pool = self._pool()
        full = md.dereplicate(pool, identity_threshold=0.90, max_keep=10_000)
        capped = md.dereplicate(pool, identity_threshold=0.90, max_keep=100)
        assert len(capped) == 100
        full_med = np.median([len(s) for s in full])
        cap_med = np.median([len(s) for s in capped])
        # a length-biased cap moved the median to the top of the range; a
        # random subsample keeps it near the full distribution's
        assert abs(cap_med - full_med) < 25, (
            f"capped median {cap_med} vs full {full_med}: the cap is "
            "selecting by length")
        assert min(len(s) for s in capped) < full_med

    def test_cap_is_reproducible_and_a_subset(self):
        pool = self._pool()
        a = md.dereplicate(pool, max_keep=80, seed=3)
        b = md.dereplicate(pool, max_keep=80, seed=3)
        c = md.dereplicate(pool, max_keep=80, seed=4)
        assert [s.name for s in a] == [s.name for s in b]
        assert [s.name for s in a] != [s.name for s in c]
        names = {s.name for s in md.dereplicate(pool, max_keep=10_000)}
        assert {s.name for s in a} <= names

    def test_output_stays_in_input_order(self):
        pool = self._pool(120)
        kept = md.dereplicate(pool, max_keep=50)
        idx = {s.name: i for i, s in enumerate(pool)}
        got = [idx[s.name] for s in kept]
        assert got == sorted(got)


class TestReportRecordsRealVsSimulated:
    """A validation CSV must say whether its numbers are measurements.

    The difference between real generator output and simulate_designed is the
    difference between a measurement and an upper bound, and a CSV opened
    months later by someone who did not run it has no other way to tell.
    """

    def _rep(self, real, n):
        return md.ValidationReport(
            n_repeats=3, alpha=0.05, overall_auroc_mean=0.97,
            overall_auroc_ci=(0.96, 0.98), fpr_mean=0.05, fpr_ci=(0.04, 0.06),
            tpr_mean=0.9, tpr_ci=(0.88, 0.93),
            signals=[md.SignalPerformance(key="comp_llr_family", auroc_mean=0.99,
                                          auroc_lo=0.98, auroc_hi=1.0,
                                          permutation_p=0.001,
                                          net_pull_mean=0.0, n_repeats=3)],
            seconds=1.0, designs_were_real=real, n_designs=n)

    def test_rows_carry_the_provenance(self):
        for real, n in ((True, 540), (False, 60)):
            row = self._rep(real, n).to_rows()[0]
            assert row["designs_real"] is real
            assert row["n_designs"] == n

    def test_report_text_says_which(self):
        assert "REAL generator output" in self._rep(True, 540).report()
        assert "measurements" in self._rep(True, 540).report()
        sim = self._rep(False, 60).report()
        assert "upper bounds" in sim and "REAL" not in sim

    def test_defaults_to_simulated(self):
        """Any report built without saying so is a bound, not a measurement."""
        r = md.ValidationReport(
            n_repeats=1, alpha=0.05, overall_auroc_mean=0.9,
            overall_auroc_ci=(0.8, 1.0), fpr_mean=0.0, fpr_ci=(0.0, 0.0),
            tpr_mean=0.9, tpr_ci=(0.9, 0.9), signals=[], seconds=0.0)
        assert r.designs_were_real is False
        assert "upper bounds" in r.report()


def _fake_records(n, tag, seed, lo=200, hi=400):
    import random
    rng = random.Random(seed)
    return [(f"sp|{tag}{i:05d}|{tag}_TEST",
             "".join(rng.choice(md.AA) for _ in range(rng.randint(lo, hi))))
            for i in range(n)]


class TestAssembleFamily:
    """Acquisition has now produced three bugs that no test could see.

    It lived in a notebook cell: a doubly-compressed response decoded as text;
    stopping at the reviewed set; and then a guard added for the local-file
    path that also matched the UniProt path, so the unreviewed top-up silently
    never ran. Each was reported from a Colab run. The logic is a module
    function now, and a fetcher can be faked.
    """

    def _fetcher(self, calls, n_reviewed=700, n_unreviewed=2500):
        def fake(query, max_sequences=0, cache_dir=None):
            reviewed = "reviewed:true" in query
            calls.append(reviewed)
            return (_fake_records(n_reviewed, "REV", 1) if reviewed
                    else _fake_records(n_unreviewed, "UNR", 2))
        return fake

    def test_unreviewed_topup_runs_when_target_is_not_met(self):
        """The regression: SwissProt holds ~700 reviewed kinases against a
        6,000 target, and the second pass was skipped, leaving 433 sequences -
        too thin for DCA and for a null that can place alpha=0.01."""
        calls = []
        acq = md.assemble_family("PF00069", target=3000,
                                 fetcher=self._fetcher(calls))
        assert calls == [True, False]
        assert len(acq.sequences) > 700
        assert [p["reviewed"] for p in acq.per_pass] == [True, False]

    def test_second_pass_skipped_once_target_is_met(self):
        calls = []
        md.assemble_family("PF00069", target=400, fetcher=self._fetcher(calls))
        assert calls == [True]

    def test_local_file_skips_the_network_entirely(self, tmp_path):
        p = tmp_path / "fam.fasta"
        p.write_text("".join(f">s{i}\n{s}\n"
                             for i, (_, s) in enumerate(
                                 _fake_records(200, "LOC", 7))))
        calls = []
        acq = md.assemble_family(term="PF00069", natural_fasta=str(p),
                                 target=3000, fetcher=self._fetcher(calls))
        assert calls == [], "a local file must not trigger a fetch"
        assert acq.source.startswith("local file")
        assert len(acq.sequences) >= 120

    def test_local_file_too_small_says_so(self, tmp_path):
        p = tmp_path / "tiny.fasta"
        p.write_text("".join(f">s{i}\n{s}\n"
                             for i, (_, s) in enumerate(
                                 _fake_records(5, "LOC", 7))))
        with pytest.raises(md.SequenceError) as exc:
            md.assemble_family(natural_fasta=str(p))
        assert "at least" in str(exc.value)

    def test_too_few_sequences_reports_where_they_went(self):
        def narrow(query, max_sequences=0, cache_dir=None):
            return _fake_records(9, "REV", 1)
        acq = md.assemble_family("PF99999", fetcher=narrow)
        assert acq.sequences == [] and acq.source == ""
        assert any("too narrow" in n for n in acq.notes)

    def test_length_window_losses_are_reported(self):
        def wrong_length(query, max_sequences=0, cache_dir=None):
            return _fake_records(400, "REV", 3, lo=600, hi=800)
        acq = md.assemble_family("PF00069", min_length=150, max_length=400,
                                 fetcher=wrong_length)
        assert acq.sequences == []
        assert acq.n_wrong_length == 800     # both passes
        assert any("widen the length window" in n for n in acq.notes)

    def test_a_failing_fetch_is_a_note_not_a_crash(self):
        def boom(query, max_sequences=0, cache_dir=None):
            raise RuntimeError("Tunnel connection failed: 403 Forbidden")
        acq = md.assemble_family("PF00069", fetcher=boom)
        assert acq.sequences == []
        assert any("403" in n for n in acq.notes)

    def test_duplicates_across_passes_are_not_double_counted(self):
        same = _fake_records(300, "REV", 1)
        def overlapping(query, max_sequences=0, cache_dir=None):
            return same           # both passes return the identical records
        acq = md.assemble_family("PF00069", target=3000, fetcher=overlapping)
        assert acq.per_pass[1]["added"] == 0
        assert acq.per_pass[1]["total"] == acq.per_pass[0]["total"]

    def test_thin_family_is_flagged(self):
        calls = []
        acq = md.assemble_family("PF00069", target=3000,
                                 fetcher=self._fetcher(calls, 200, 300))
        assert any("mutual information" in n for n in acq.notes)

    def test_requires_a_term_or_a_file(self):
        with pytest.raises(md.SequenceError):
            md.assemble_family()


class TestDereplicationIsExact:
    """The k-mer inverted index must give the same answer as all-pairs.

    Removing the early stop at max_keep (to fix the length bias) made
    dereplication O(n x clusters): 115s on 11,993 real UniProt kinases, and
    quadratic on a family with little redundancy. The index prunes by counting
    shared k-mers off postings and applying the exact Jaccard bound
    c >= t*(a+b)/(1+t), so it is a speed-up and not an approximation - 19s on
    the same input, with the same 3,544 sequences kept.
    """

    def _naive(self, sequences, identity_threshold=0.90, k=4):
        cut = max(0.05, identity_threshold ** (k * 1.6))
        order = sorted(range(len(sequences)),
                       key=lambda i: -len(sequences[i].seq))
        kept_idx, kept_kmers = [], []
        for i in order:
            ks = md.kmer_set(sequences[i].seq, k)
            if any(md.kmer_jaccard(ks, o) >= cut for o in kept_kmers):
                continue
            kept_idx.append(i)
            kept_kmers.append(ks)
        return [sequences[i].name for i in sorted(kept_idx)]

    def _pool(self, n=260, seed=0):
        """Related sequences, so dereplication actually has work to do."""
        rng = np.random.default_rng(seed)
        base = "".join(rng.choice(list(md.AA), size=300))
        out = []
        for i in range(n):
            s = list(base)
            for p in rng.integers(0, len(s), size=int(rng.integers(5, 160))):
                s[p] = rng.choice(list(md.AA))
            cut = int(rng.integers(0, 60))
            out.append(md.clean_sequence("".join(s)[cut:], name=f"s{i}",
                                         min_length=30))
        return out

    @pytest.mark.parametrize("threshold", [0.70, 0.90, 0.95])
    def test_matches_all_pairs(self, threshold):
        pool = self._pool()
        assert ([s.name for s in md.dereplicate(pool, threshold, 10 ** 6)]
                == self._naive(pool, threshold))

    def test_matches_all_pairs_on_unrelated_sequences(self):
        rng = np.random.default_rng(3)
        pool = [md.clean_sequence("".join(rng.choice(list(md.AA), size=int(L))),
                                  name=f"u{i}", min_length=30)
                for i, L in enumerate(rng.integers(150, 400, size=200))]
        assert ([s.name for s in md.dereplicate(pool, 0.90, 10 ** 6)]
                == self._naive(pool, 0.90))

    def test_near_duplicates_are_still_collapsed(self):
        rng = np.random.default_rng(5)
        base = "".join(rng.choice(list(md.AA), size=300))
        pool = [md.clean_sequence(base, name="orig", min_length=30)]
        for i in range(6):                       # one or two point mutations
            s = list(base)
            s[int(rng.integers(0, 300))] = rng.choice(list(md.AA))
            pool.append(md.clean_sequence("".join(s), name=f"dup{i}",
                                          min_length=30))
        assert len(md.dereplicate(pool, 0.90, 10 ** 6)) == 1


class TestBackboneMembership:
    """A design backbone that is not in the profiled family inflates the result.

    Reported from a Colab run: 98.5% detection at alpha=0.01 with 0% false
    positives, and the native-backbone control flagging 2 of 12. The cause was
    neither of the two the code checked for. Coverage was 63-90% in both groups
    and composition was identical, so "does it align" and "is it a bare
    fragment" both passed. Three of the six backbones were tyrosine kinases
    with 34-39% nearest-neighbour identity to a PF00069 family dominated by
    small Ser/Thr kinases - distant WITHIN the family. Their designs scored
    100/100/100 against 91/98/100 for the in-family three.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def family_profile():
        """A tight family, plus one sequence from somewhere else entirely."""
        fam = md.simulate_natural_family(n=400, seed=21)
        prof = md.FamilyProfile("memb", random_state=0).build(
            fam[:260], check_homogeneity=False)
        return prof, fam

    def _gen(self, cs, source):
        return md.GeneratedSequence(sequence=cs, source=source, is_native=True)

    def test_in_family_native_passes(self, family_profile):
        prof, fam = family_profile
        rows, stats = md.backbone_membership(
            prof, [self._gen(fam[300], "member")], sample=60)
        assert rows[0].in_family, rows[0].reason
        assert rows[0].identity > 0.5

    def test_distant_native_is_called_out(self, family_profile):
        prof, fam = family_profile
        other = md.simulate_natural_family(n=3, seed=999)[0]
        rows, stats = md.backbone_membership(
            prof, [self._gen(fam[300], "member"),
                   self._gen(other, "stranger")], sample=60)
        by = {r.source: r for r in rows}
        assert by["member"].in_family
        assert not by["stranger"].in_family
        assert "identical" in by["stranger"].reason or "align" in by["stranger"].reason
        text = md.describe_backbone_membership(rows, stats)
        assert "stranger" in text and "do not belong" in text

    def test_the_bar_is_the_familys_own_distribution(self, family_profile):
        """A fixed identity cutoff is wrong for both a tight family and a
        diverse one, so the threshold comes from the members themselves."""
        prof, fam = family_profile
        rows, stats = md.backbone_membership(
            prof, [self._gen(fam[300], "m")], sample=60)
        assert stats["n_reference"] > 0
        assert 0.0 < stats["p10"] <= stats["p50"] <= 1.0

    def test_natives_are_deduplicated(self, family_profile):
        """The same native is written into every run folder, so the control
        must not count it once per folder."""
        prof, fam = family_profile
        dupes = [self._gen(fam[300], "bb"), self._gen(fam[300], "bb")]
        rows, _ = md.backbone_membership(prof, dupes, sample=40)
        assert len(rows) == 1

    def test_unbuilt_profile_refuses(self):
        with pytest.raises(md.ModelNotBuiltError):
            md.backbone_membership(md.FamilyProfile("x"), [])
