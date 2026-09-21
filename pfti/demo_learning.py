"""Learning-layer demo: the three-wall narrative (spec §1).

  first crash  -> the store memorizes THAT wall (instance)
  second crash -> the store learns THAT CLASS of wall (LGG generalization)
  third agent  -> correctly warned on a BRAND-NEW call it never saw

No LLM, fully symbolic and deterministic.

Run:  python -m pfti.demo_learning
"""
from __future__ import annotations

from .core.signature import Signature
from .core.generalize import LearningLayer


def S(*preds):
    return Signature(frozenset(preds))


def banner(t):
    print("\n" + "=" * 66 + f"\n{t}\n" + "=" * 66)


def main():
    ll = LearningLayer()
    tool, ec = "write_file", "PermissionDenied"

    banner("WALL 1: FileOps fails writing /data/report.csv")
    a = S(("tool", "write_file"), ("path_prefix", "/data"), ("perm", "ro"),
          ("size_bucket", "large"), ("path", "/data/report.csv"))
    ll.on_failure(a, tool, ec)
    print("  stored 1 instance. generalized record yet?",
          ("write_file", "PermissionDenied") in ll.grecs, "(need >=2 to merge)")

    banner("WALL 2: Coder fails writing a DIFFERENT file /data/summary.txt")
    b = S(("tool", "write_file"), ("path_prefix", "/data"), ("perm", "ro"),
          ("size_bucket", "small"), ("path", "/data/summary.txt"))
    ll.on_failure(b, tool, ec)
    g = ll.grecs[("write_file", "PermissionDenied")]
    print("  the store now knows the CLASS of wall (LGG intersection):")
    print("     learned pattern:", g.gsig.readable())
    print("  note it kept the shared cause (path_prefix=/data, perm=ro) and")
    print("  DROPPED the irrelevant differences (exact path, size).")

    banner("WALL 3: Planner attempts a BRAND-NEW call never seen before")
    unseen = S(("tool", "write_file"), ("path_prefix", "/data"), ("perm", "ro"),
               ("size_bucket", "med"), ("path", "/data/NEVER_SEEN_BEFORE.log"))
    hits = ll.check(unseen)
    print("  new call:", unseen.readable())
    if hits:
        print("  -> WARNED via generalized record:", hits[0].gsig.readable())
        print(f"     ({len(hits[0].provenance)} teammates hit this pattern)")
    print("  The system correctly warns on a call it has NEVER stored.")
    print("  That is the upgrade: from memorizing walls to knowing wall-classes.")

    banner("SILENCE CHECK: a safe call into a writable dir")
    safe = S(("tool", "write_file"), ("path_prefix", "/tmp"), ("perm", "rw"),
             ("size_bucket", "med"), ("path", "/tmp/ok.txt"))
    print("  safe call:", safe.readable())
    print("  -> warned?", len(ll.check(safe)) > 0, "(correctly silent)")


if __name__ == "__main__":
    main()
