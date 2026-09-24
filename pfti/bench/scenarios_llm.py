"""LLM-facing scenarios for the before/after benchmark.

Each scenario reuses the controlled testbed (``envs.tools.World`` +
``would_fail`` oracle + real ``TOOLS``) but gives every agent a natural
task in plain English -- what a real model actually receives. A compact
``script`` (primary action + optional alternative) rides alongside for the
deterministic offline fake; real models never see it.

Design invariant: within a scenario, several agents face the SAME failure
class. The FIRST agent to hit it always fails (no peer has warned it yet --
PFTI prevents *repeats*, not the first occurrence). Later agents either
repeat the mistake (mode=off) or route around it (mode=inject). That gap
is the measured before/after.
"""
from __future__ import annotations

SCENARIOS = [
    {
        "name": "perm_denied",
        "failure_mode": "PermissionDenied",
        "world": {"dirs": ["/", "/data", "/tmp"],
                  "perm": {"/data": "ro", "/tmp": "rw"}},
        "agents": [
            {"id": "FileOps",
             "task": "Save the quarterly report to the shared data folder "
                     "at /data/report.csv (contents: 'q1 numbers'). If that "
                     "location does not work, put it somewhere writable.",
             "script": {"primary": ["write_file",
                                     {"path": "/data/report.csv",
                                      "content": "q1 numbers"}],
                        "alt": ["write_file",
                                {"path": "/tmp/report.csv",
                                 "content": "q1 numbers"}]}},
            {"id": "Coder",
             "task": "Save the executive summary to /data/summary.txt "
                     "(contents: 'exec summary'). If that location does not "
                     "work, put it somewhere writable.",
             "script": {"primary": ["write_file",
                                     {"path": "/data/summary.txt",
                                      "content": "exec summary"}],
                        "alt": ["write_file",
                                {"path": "/tmp/summary.txt",
                                 "content": "exec summary"}]}},
            {"id": "Planner",
             "task": "Save the meeting notes to /tmp/notes.txt "
                     "(contents: 'notes').",
             "script": {"primary": ["write_file",
                                    {"path": "/tmp/notes.txt",
                                     "content": "notes"}],
                        "alt": None}},
        ],
        "success": {"all_files": ["/tmp/summary.txt", "/tmp/notes.txt"]},
    },
    {
        "name": "notfound_dir",
        "failure_mode": "NotFound",
        "world": {"dirs": ["/", "/tmp"], "perm": {"/tmp": "rw"}},
        "agents": [
            {"id": "FileOps",
             "task": "Write build output to /out/data.json (contents "
                     "'data'). If the directory is missing, write it to a "
                     "directory that exists instead.",
             "script": {"primary": ["write_file",
                                     {"path": "/out/data.json",
                                      "content": "data"}],
                        "alt": ["write_file",
                                {"path": "/tmp/data.json",
                                 "content": "data"}]}},
            {"id": "Coder",
             "task": "Write the run log to /out/log.txt (contents 'log'). "
                     "If the directory is missing, write it to a directory "
                     "that exists instead.",
             "script": {"primary": ["write_file",
                                     {"path": "/out/log.txt",
                                      "content": "log"}],
                        "alt": ["write_file",
                                {"path": "/tmp/log.txt", "content": "log"}]}},
        ],
        # FileOps is the seed (fails first, unwarned); success is whether the
        # LATER agent avoids the repeat and lands its artifact.
        "success": {"any_files": ["/tmp/log.txt"]},
    },
    {
        "name": "rate_limit",
        "failure_mode": "RateLimited",
        "world": {"dirs": ["/"], "api_budget": {"/render": 1,
                                                "/render-batch": 8}},
        "agents": [
            {"id": "Planner",
             "task": "Render job1 by calling the /render endpoint with "
                     "payload 'job1'.",
             "script": {"primary": ["call_api",
                                    {"endpoint": "/render",
                                     "payload": "job1"}],
                        "alt": None}},
            {"id": "Coder",
             "task": "Render job2 via the /render endpoint (payload 'job2'). "
                     "If /render is rate limited, use the /render-batch "
                     "endpoint instead.",
             "script": {"primary": ["call_api",
                                    {"endpoint": "/render",
                                     "payload": "job2"}],
                        "alt": ["call_api",
                                {"endpoint": "/render-batch",
                                 "payload": "job2"}]}},
            {"id": "FileOps",
             "task": "Render job3 via the /render endpoint (payload 'job3'). "
                     "If /render is rate limited, use the /render-batch "
                     "endpoint instead.",
             "script": {"primary": ["call_api",
                                    {"endpoint": "/render",
                                     "payload": "job3"}],
                        "alt": ["call_api",
                                {"endpoint": "/render-batch",
                                 "payload": "job3"}]}},
        ],
        "success": {"api_calls_ok": 2},
    },
    {
        "name": "quota",
        "failure_mode": "QuotaExceeded",
        "world": {"dirs": ["/", "/tmp"], "perm": {"/tmp": "rw"},
                  "disk_quota": 50},
        "agents": [
            {"id": "FileOps",
             "task": "Save the full dataset dump to /tmp/a.txt. Prefer the "
                     "complete dump; if the disk quota is exceeded, save a "
                     "compact version instead.",
             "script": {"primary": ["write_file",
                                     {"path": "/tmp/a.txt",
                                      "content": "X" * 80}],
                        "alt": ["write_file",
                                {"path": "/tmp/a.txt", "content": "X" * 15}]}},
            {"id": "Coder",
             "task": "Save the full log bundle to /tmp/b.txt. Prefer the "
                     "complete bundle; if the disk quota is exceeded, save a "
                     "compact version instead.",
             "script": {"primary": ["write_file",
                                     {"path": "/tmp/b.txt",
                                      "content": "Y" * 80}],
                        "alt": ["write_file",
                                {"path": "/tmp/b.txt", "content": "Y" * 15}]}},
        ],
        # FileOps is the seed (fails first, unwarned); success is whether the
        # LATER agent avoids the repeat and lands its (compact) artifact.
        "success": {"any_files": ["/tmp/b.txt"]},
    },
    {
        "name": "missing_table",
        "failure_mode": "NotFound",
        "world": {"dirs": ["/"], "tables": ["users"]},
        "agents": [
            {"id": "Coder",
             "task": "Query the 'accounts' table for the report. If that "
                     "table does not exist, query the 'users' table instead.",
             "script": {"primary": ["query_db", {"table": "accounts"}],
                        "alt": ["query_db", {"table": "users"}]}},
            {"id": "FileOps",
             "task": "Query the 'accounts' table for the export. If that "
                     "table does not exist, query the 'users' table instead.",
             "script": {"primary": ["query_db", {"table": "accounts"}],
                        "alt": ["query_db", {"table": "users"}]}},
            {"id": "Planner",
             "task": "Query the 'accounts' table for the summary. If that "
                     "table does not exist, query the 'users' table instead.",
             "script": {"primary": ["query_db", {"table": "accounts"}],
                        "alt": ["query_db", {"table": "users"}]}},
        ],
        "success": {"db_queries_ok": 2},
    },
]


# ---------------------------------------------------------------------------
# V2 scenarios (added after review).
#
# The five V1 scenarios are all environment-condition failures whose G-mid
# signature is exactly right, so a warning can never be a false positive
# there (only `quota` admits one). V2 adds the two kinds the review asked
# for:
#   * adversarial_fp -- the peer failure MATCHES at G-mid but does NOT apply
#     to the later agent's call, so the warning is wrong. Heeding it hurts.
#     Measures harm from bad warnings.
#   * reasoning      -- the failure comes from the agent misreading or
#     guessing an argument, not from a hidden environment condition.
# In adversarial_fp scenarios the fake's `alt` is the over-cautious diversion
# (heeding a wrong warning), so the fake shows harm by construction; real
# models decide for themselves.
# ---------------------------------------------------------------------------

SCENARIOS_V2 = [
    {
        "name": "prefix_collision",
        "kind": "adversarial_fp",
        "failure_mode": "NotFound",
        "world": {"dirs": ["/", "/data", "/tmp"],
                  "perm": {"/data": "rw", "/tmp": "rw"}},
        "agents": [
            {"id": "FileOps",
             "task": "Write 'cache' to /data/cache/tmp.bin. If that folder "
                     "does not exist, skip it and reply DONE.",
             "script": {"primary": ["write_file",
                                    {"path": "/data/cache/tmp.bin",
                                     "content": "cache"}],
                        "alt": None}},
            {"id": "Coder",
             "task": "Save the results to /data/results.txt (contents "
                     "'results'). It must be in /data: the downstream job "
                     "only reads from /data.",
             "script": {"primary": ["write_file",
                                    {"path": "/data/results.txt",
                                     "content": "results"}],
                        "alt": ["write_file",
                                {"path": "/tmp/results.txt",
                                 "content": "results"}]}},
        ],
        "success": {"all_files": ["/data/results.txt"]},
    },
    {
        "name": "quota_small_ok",
        "kind": "adversarial_fp",
        "failure_mode": "QuotaExceeded",
        "world": {"dirs": ["/", "/tmp", "/scratch"],
                  "perm": {"/tmp": "rw", "/scratch": "rw"},
                  "disk_quota": 50},
        "agents": [
            {"id": "FileOps",
             "task": "Save the full dataset dump (the letter X repeated 80 "
                     "times) to /tmp/dump.txt. If the disk quota is exceeded, "
                     "give up and reply DONE.",
             "script": {"primary": ["write_file",
                                    {"path": "/tmp/dump.txt",
                                     "content": "X" * 80}],
                        "alt": None}},
            {"id": "Planner",
             "task": "Save the two-letter status note 'ok' to /tmp/note.txt. "
                     "The monitor only reads /tmp/note.txt.",
             "script": {"primary": ["write_file",
                                    {"path": "/tmp/note.txt", "content": "ok"}],
                        "alt": ["write_file",
                                {"path": "/scratch/note.txt",
                                 "content": "ok"}]}},
        ],
        "success": {"all_files": ["/tmp/note.txt"]},
    },
    {
        # The world changes in a way the signature CANNOT see (the API budget
        # is not a signature dimension), so the stored peer failure is stale
        # and the warning is wrong.
        "name": "stale_ratelimit",
        "kind": "adversarial_fp",
        "failure_mode": "RateLimited",
        "world": {"dirs": ["/"], "api_budget": {"/render": 1}},
        "world_events": [{"after_agent": 1,
                          "set": {"api_budget": {"/render": 5}}}],
        "agents": [
            {"id": "Planner",
             "task": "Render job1 via the /render endpoint (payload 'job1').",
             "script": {"primary": ["call_api", {"endpoint": "/render",
                                                 "payload": "job1"}],
                        "alt": None}},
            {"id": "Coder",
             "task": "Render job2 via the /render endpoint (payload 'job2'). "
                     "If it is rate limited, reply DONE.",
             "script": {"primary": ["call_api", {"endpoint": "/render",
                                                 "payload": "job2"}],
                        "alt": None}},
            {"id": "FileOps",
             "task": "Render job3 via the /render endpoint (payload 'job3'). "
                     "Note: the /render rate-limit window reset a minute ago, "
                     "so it has fresh capacity. There is no other render "
                     "endpoint.",
             "script": {"primary": ["call_api", {"endpoint": "/render",
                                                 "payload": "job3"}],
                        "alt": ["call_api", {"endpoint": "/render-v0",
                                             "payload": "job3"}]}},
        ],
        "success": {"api_calls_ok": 2},
    },
    {
        # Control: the world changes in a way the signature CAN see (perm is
        # a G-mid dimension), so a state-aware matcher should NOT warn.
        "name": "stale_perm_control",
        "kind": "control",
        "failure_mode": "PermissionDenied",
        "world": {"dirs": ["/", "/data", "/tmp"],
                  "perm": {"/data": "ro", "/tmp": "rw"}},
        "world_events": [{"after_agent": 0, "set": {"perm": {"/data": "rw"}}}],
        "agents": [
            {"id": "FileOps",
             "task": "Save the quarterly report to /data/report.csv "
                     "(contents 'q1'). If that fails, reply DONE.",
             "script": {"primary": ["write_file",
                                    {"path": "/data/report.csv",
                                     "content": "q1"}],
                        "alt": None}},
            {"id": "Coder",
             "task": "Save the executive summary to /data/summary.txt "
                     "(contents 'exec'). It must be in /data.",
             "script": {"primary": ["write_file",
                                    {"path": "/data/summary.txt",
                                     "content": "exec"}],
                        "alt": ["write_file",
                                {"path": "/tmp/summary.txt",
                                 "content": "exec"}]}},
        ],
        "success": {"all_files": ["/data/summary.txt"]},
    },
    {
        "name": "retired_endpoint",
        "kind": "reasoning",
        "failure_mode": "NotFound",
        # v2 fix: the endpoint was unguessable in the first v2 run (0/4
        # models found it). Now it is discoverable from a file, so the
        # failure is "acting before checking", which real agents do.
        "world": {"dirs": ["/", "/api"], "api_budget": {"/v2/render": 5},
                  "files": {"/api/ENDPOINTS.txt":
                            "render (current, v2): /v2/render\n"
                            "render (v1, retired): /render"}},
        "agents": [
            {"id": "Planner",
             "task": "Render job1 with the render API (payload 'job1'). The "
                     "render API was moved to version 2 last month; the endpoint "
                     "list is in /api/ENDPOINTS.txt.",
             "script": {"primary": ["call_api", {"endpoint": "/render",
                                                 "payload": "job1"}],
                        "alt": ["call_api", {"endpoint": "/v2/render",
                                             "payload": "job1"}]}},
            {"id": "Coder",
             "task": "Render job2 with the render API (payload 'job2'). The "
                     "render API was moved to version 2 last month; the endpoint "
                     "list is in /api/ENDPOINTS.txt.",
             "script": {"primary": ["call_api", {"endpoint": "/render",
                                                 "payload": "job2"}],
                        "alt": ["call_api", {"endpoint": "/v2/render",
                                             "payload": "job2"}]}},
            {"id": "FileOps",
             "task": "Render job3 with the render API (payload 'job3'). The "
                     "render API was moved to version 2 last month; the endpoint "
                     "list is in /api/ENDPOINTS.txt.",
             "script": {"primary": ["call_api", {"endpoint": "/render",
                                                 "payload": "job3"}],
                        "alt": ["call_api", {"endpoint": "/v2/render",
                                             "payload": "job3"}]}},
        ],
        "success": {"api_calls_ok": 2},
    },
    {
        "name": "guess_dir",
        "kind": "reasoning",
        "failure_mode": "NotFound",
        # v2 fix: list_dir only lists FILES, so the folder was
        # undiscoverable in the first v2 run (0/4 models). Now a README in
        # the root names it (and shows up in list_dir('/')).
        "world": {"dirs": ["/", "/reports_2026"],
                  "perm": {"/reports_2026": "rw"},
                  "files": {"/README.txt":
                            "The team's reports folder is /reports_2026"}},
        "agents": [
            {"id": "FileOps",
             "task": "Save 'sales' as sales.txt in the team's reports "
                     "folder. If you are unsure of the folder's exact name, "
                     "read /README.txt first.",
             "script": {"primary": ["write_file",
                                    {"path": "/reports/sales.txt",
                                     "content": "sales"}],
                        "alt": ["write_file",
                                {"path": "/reports_2026/sales.txt",
                                 "content": "sales"}]}},
            {"id": "Coder",
             "task": "Save 'costs' as costs.txt in the team's reports "
                     "folder. If you are unsure of the folder's exact name, "
                     "read /README.txt first.",
             "script": {"primary": ["write_file",
                                    {"path": "/reports/costs.txt",
                                     "content": "costs"}],
                        "alt": ["write_file",
                                {"path": "/reports_2026/costs.txt",
                                 "content": "costs"}]}},
            {"id": "Planner",
             "task": "Save 'plan' as plan.txt in the team's reports "
                     "folder. If you are unsure of the folder's exact name, "
                     "read /README.txt first.",
             "script": {"primary": ["write_file",
                                    {"path": "/reports/plan.txt",
                                     "content": "plan"}],
                        "alt": ["write_file",
                                {"path": "/reports_2026/plan.txt",
                                 "content": "plan"}]}},
        ],
        "success": {"all_files": ["/reports_2026/costs.txt",
                                  "/reports_2026/plan.txt"]},
    },
]

for _s in SCENARIOS:
    _s.setdefault("kind", "environment")

SCENARIO_SETS = {
    "v1": SCENARIOS,
    "v2": SCENARIOS_V2,
    "all": SCENARIOS + SCENARIOS_V2,
    "reasoning": [s for s in SCENARIOS_V2 if s["kind"] == "reasoning"],
}
