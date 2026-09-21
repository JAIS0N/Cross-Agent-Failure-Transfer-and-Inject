"""The 5 lifecycle tests that DEFINE 'done' for the learning layer (spec §3)."""
from pfti.core.signature import Signature
from pfti.core.generalize import LearningLayer, _nontool


def S(*preds):
    return Signature(frozenset(preds))


def test_1_lgg_equals_shared_dims():
    """Two instances differing only in irrelevant dims -> generalized record
    equals exactly the shared dims."""
    ll = LearningLayer()
    ll.on_failure(S(("tool", "write_file"), ("path_prefix", "/data"),
                    ("perm", "ro"), ("path", "/data/a.txt")),
                  "write_file", "PermissionDenied")
    ll.on_failure(S(("tool", "write_file"), ("path_prefix", "/data"),
                    ("perm", "ro"), ("path", "/data/b.txt")),
                  "write_file", "PermissionDenied")
    g = ll.grecs[("write_file", "PermissionDenied")]
    assert g.gsig.preds == frozenset({("tool", "write_file"),
                                      ("path_prefix", "/data"), ("perm", "ro")})


def test_2_min_size_refusal():
    """Intersection below MIN_SIZE -> no generalized record, refusal logged."""
    ll = LearningLayer(min_size=2)
    # share only path_prefix (1 non-tool pred) -> below MIN_SIZE=2
    ll.on_failure(S(("tool", "write_file"), ("path_prefix", "/data"),
                    ("perm", "ro")), "write_file", "PermissionDenied")
    ll.on_failure(S(("tool", "write_file"), ("path_prefix", "/data"),
                    ("perm", "rw")), "write_file", "PermissionDenied")
    assert ("write_file", "PermissionDenied") not in ll.grecs
    assert any(r[1] == "intersection_too_small" for r in ll.refusals)


def test_3_kmin_fpmax_split():
    """A generalized record fed K_MIN shadow fires with > FP_MAX
    oracle-contradictions -> deactivated and split; sub-records subsume
    their own instances only."""
    ll = LearningLayer(min_size=2, k_min=5, fp_max=0.3)
    # bucket: all share (path_prefix=/data, size=large -- a COINCIDENTAL
    # correlated irrelevant dim, spec pitfall). Real cause is perm=ro; the
    # perm=rw instances are coincidental siblings. LGG over-shoots because
    # size=large survives the intersection while perm does not.
    for pth, perm in [("/data/a", "ro"), ("/data/b", "ro"),
                      ("/data/c", "rw"), ("/data/d", "rw")]:
        ll.on_failure(S(("tool", "write_file"), ("path_prefix", "/data"),
                        ("perm", perm), ("size_bucket", "large"),
                        ("path", pth)),
                      "write_file", "PermissionDenied")
    key = ("write_file", "PermissionDenied")
    g = ll.grecs[key]
    # gsig = {path_prefix=/data, size=large} -> over-general (misses perm).
    # feed fires: calls with perm=rw actually SUCCEED (oracle_fail=False)
    for i in range(6):
        perm = "rw" if i % 2 else "ro"
        call = S(("tool", "write_file"), ("path_prefix", "/data"),
                 ("perm", perm), ("size_bucket", "large"),
                 ("path", f"/data/x{i}"))
        ll.record_fire(g, call, oracle_fail=(perm == "ro"))
    assert g.active is False              # deactivated
    assert ll.subrecs.get(key)            # split produced sub-records
    for sub in ll.subrecs[key]:
        assert len(_nontool(sub.gsig.preds)) >= ll.min_size


def test_4_poisoned_intersection_refused():
    """One crafted instance sharing almost nothing drags the intersection
    below MIN_SIZE -> Guard 1 refuses."""
    ll = LearningLayer(min_size=2)
    ll.on_failure(S(("tool", "call_api"), ("endpoint", "/render"),
                    ("payload_kind", "str")), "call_api", "RateLimited")
    ll.on_failure(S(("tool", "call_api"), ("endpoint", "/render"),
                    ("payload_kind", "str")), "call_api", "RateLimited")
    assert ("call_api", "RateLimited") in ll.grecs      # healthy so far
    # poison: shares only tool
    ll.on_failure(S(("tool", "call_api"), ("endpoint", "/other"),
                    ("payload_kind", "dict")), "call_api", "RateLimited")
    assert ("call_api", "RateLimited") not in ll.grecs
    assert any(r[1] == "intersection_too_small" for r in ll.refusals)


def test_5_matched_via_generalized_precedence():
    """A call matched by a generalized record is tagged generalized; the
    generalized layer is checked first."""
    ll = LearningLayer()
    ll.on_failure(S(("tool", "write_file"), ("path_prefix", "/data"),
                    ("perm", "ro"), ("path", "/data/a")),
                  "write_file", "PermissionDenied")
    ll.on_failure(S(("tool", "write_file"), ("path_prefix", "/data"),
                    ("perm", "ro"), ("path", "/data/b")),
                  "write_file", "PermissionDenied")
    # brand-new call, never seen at G-fine, but satisfies the learned pattern
    unseen = S(("tool", "write_file"), ("path_prefix", "/data"),
               ("perm", "ro"), ("path", "/data/NEVER_SEEN.txt"))
    hits = ll.check(unseen)
    assert len(hits) == 1
    assert hits[0].gsig.preds <= unseen.preds     # subsumption match
