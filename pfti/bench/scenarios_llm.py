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
