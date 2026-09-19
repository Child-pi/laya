---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: onnx
tags:
  - onnx
  - onnxruntime
  - decision-model
  - system-one
  - jev
  - modernbert
---

# Laya — ONNX export

ONNX export of [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (ModernBERT-large
encoder + Laya's decision head) for use with [`@receptron/laya`](https://www.npmjs.com/package/@receptron/laya)
from Node.js / TypeScript, or with ONNX Runtime directly.

| File | Contents |
|---|---|
| `laya.onnx` / `laya.onnx.data` | graph + fp32 weights (English checkpoint, 421M parameters) |
| `laya_config.json` | `max_len`, `head_max_len` and the per-cardinality temperatures from `rl_agent_config.json` |
| `tokenizer/` | the checkpoint's tokenizer |

Inputs: `input_ids` [B,L] int64, `attention_mask` [B,L] int64, `marker_pos` [B,K] int64, `marker_mask` [B,K] bool, `qtype` [B] int64.
Outputs: `logits` [B,K] float32 (uncalibrated; masked slots = -1e4), `act_probs` [B,2] float32.

Built with [`export/export_onnx.py`](https://github.com/receptron/laya/blob/main/export/export_onnx.py);
max logit difference vs. the PyTorch reference ≈ 1e-5.

```ts
import { Laya } from "@receptron/laya";
const laya = await Laya.load(); // downloads this bundle on first use
```

Weights are Convai Innovations' and remain under Apache 2.0. Export code: MIT, https://github.com/receptron/laya
