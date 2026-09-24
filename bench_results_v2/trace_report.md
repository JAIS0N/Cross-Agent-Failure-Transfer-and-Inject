# What happens after a warning (from saved traces)

## Agent response to each warning

| model | correct warnings: rerouted / repeated / stopped | false warnings: rerouted / repeated / stopped |
|---|--:|--:|
| llama3.1:8b | 2 / 1 / 3 | 0 / 1 / 1 |
| qwen2.5:14b | 8 / 0 / 2 | 2 / 1 / 0 |
| qwen2.5:3b | 9 / 2 / 0 | 0 / 2 / 0 |
| qwen2.5:7b | 11 / 0 / 1 | 0 / 2 / 0 |

## Regressions: off succeeded, inject failed (same model × scenario × repeat)

| model | scenario | rep | cause | warnings (correct?, next action) | agent's last words after warning |
|---|---|--:|---|---|---|
| llama3.1:8b | missing_table | 0 | after correct warning(s) only | FileOps:TP,stopped; Planner:TP,stopped | Let me call the query_db tool again with the 'users' table. |
| llama3.1:8b | stale_ratelimit | 0 | after FALSE-positive warning | FileOps:FP,stopped | !! PEER FAILURE WARNING Agent Coder attempted call_api with a matching pattern (arg_types=endpoint:str, endpoint=/status, tool=call_api) at step 2 and failed with RateLimited: "rate limit exhausted for /status". Consider: (a) verifying the  |

## Summary

- **llama3.1:8b**: inject fixed 0 off-failures; regressions: 1 after correct warning(s) only, 1 after FALSE-positive warning
- **qwen2.5:14b**: inject fixed 0 off-failures; regressions: none
- **qwen2.5:3b**: inject fixed 0 off-failures; regressions: none
- **qwen2.5:7b**: inject fixed 1 off-failures; regressions: none