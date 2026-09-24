# What happens after a warning (from saved traces)

## Agent response to each warning

| model | correct warnings: rerouted / repeated / stopped | false warnings: rerouted / repeated / stopped |
|---|--:|--:|
| llama3.1:8b | 5 / 1 / 0 | 2 / 1 / 0 |

## Regressions: off succeeded, inject failed (same model × scenario × repeat)

| model | scenario | rep | cause | warnings (correct?, next action) | agent's last words after warning |
|---|---|--:|---|---|---|

## Summary

- **llama3.1:8b**: inject fixed 0 off-failures; regressions: none