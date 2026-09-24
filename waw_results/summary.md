# Who&When real-LLM before/after (real Ollama)

_2026-09-21 14:25:11 - 36 real code-execution transfer-opportunity cases from 184 Who&When logs - repeats=1 - 3123.7s_


Each case: the real conversation is replayed up to the agent turn that proposed the failing code; the model regenerates that turn with no warning (off) and with a PFTI peer-failure warning built from the earlier real failure (inject). Both code blocks are actually executed and graded against the real historical error.


| model | fail rate off -> inject | same-error rate off -> inject | code parsed off / inject | engaged with warning |
|---|---|---|---|---|
| qwen2.5:3b | 77% -> 71% | 9% -> 10% | 35/36 / 31/36 | 50% |
| qwen2.5:7b | 78% -> 64% | 15% -> 7% | 27/36 / 28/36 | 47% |
| qwen2.5:14b | 77% -> 77% | 12% -> 15% | 26/36 / 26/36 | 39% |
| llama3.1:8b | 66% -> 61% | 0% -> 3% | 29/36 / 31/36 | 44% |

Unsafe snippets skipped (never executed; denylist for file deletion / shells / system changes): qwen2.5:3b off=0 inject=1, qwen2.5:7b off=0 inject=0, qwen2.5:14b off=0 inject=0, llama3.1:8b off=0 inject=1


**fail rate**: regenerated code failed at all when executed. **same-error rate**: failed with the same error class as the real historical failure (a true repeat). Lower is better for both.


Caveat: the original tool environment (downloaded files, web state) is not reproduced, so some regenerated code fails for environment reasons in BOTH conditions; the off vs inject difference is the comparison, not the absolute rate.
