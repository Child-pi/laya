#!/usr/bin/env python3
"""
Snake Game (貪吃蛇) powered by Laya System-1 Decision Agent.
Can be executed directly via Python or in Google Colab.
"""

import os
import sys
import json
import math
import time
import random
import numpy as np

try:
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
except ImportError:
    print("Please install requirements: pip install onnxruntime huggingface_hub transformers numpy")
    sys.exit(1)


class PythonLayaAgent:
    """Python implementation of Laya System-1 Decision Engine."""
    QTYPES = {"choice": 0, "score": 1, "noul": 2}

    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "laya_config.json")) as f:
            self.config = json.load(f)
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.cls_id = self.tok.cls_token_id or self.tok.convert_tokens_to_ids("[CLS]")
        self.sep_id = self.tok.sep_token_id or self.tok.convert_tokens_to_ids("[SEP]")
        self.mask_id = self.tok.mask_token_id or self.tok.convert_tokens_to_ids("[MASK]")
        self.pad_id = self.tok.pad_token_id or self.tok.convert_tokens_to_ids("[PAD]")
        
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "laya.onnx"),
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def _render_options(self, qtype, criteria):
        if qtype == "choice":
            return [f"{k}: {v}" if v else k for k, v in criteria.items()]
        elif qtype == "score":
            return [f"level {i}: {c}" for i, c in enumerate(criteria)]
        else:
            return ["false: no, the statement does not hold", "true: yes, the statement holds"]

    def _build_sequence(self, state, qtype, instructions, criteria):
        max_len = self.config["max_len"]
        head_max_len = self.config["head_max_len"]
        
        opts = self._render_options(qtype, criteria)
        head_text = f"{qtype} question: {instructions}"
        head_ids = self.tok.encode(head_text, add_special_tokens=False)
        
        opt_ids = [[self.mask_id] + self.tok.encode(" " + o, add_special_tokens=False)[:48] for o in opts]
        total_opts = sum(len(o) for o in opt_ids)
        opt_budget = head_max_len - total_opts
        if opt_budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        
        head_ids = head_ids[:max(8, opt_budget)]
        seq = [self.cls_id] + head_ids + [self.sep_id]
        markers = []
        for o in opt_ids:
            markers.append(len(seq))
            seq.extend(o)
        seq.append(self.sep_id)
        
        room = max(0, max_len - len(seq) - 1)
        state_str = json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state
        st_ids = self.tok.encode(state_str, add_special_tokens=False)[:room]
        seq.extend(st_ids)
        seq.append(self.sep_id)
        
        return seq[:max_len], [m for m in markers if m < max_len]

    def system_one(self, state, questions):
        qids = list(questions.keys())
        items = []
        for qid in qids:
            q = questions[qid]
            qtype = q["type"]
            ins = q["instructions"]
            crit = q.get("criteria")
            ids, markers = self._build_sequence(state, qtype, ins, crit)
            items.append({
                "qid": qid,
                "qtype": qtype,
                "qtype_num": self.QTYPES[qtype],
                "crit": crit,
                "ids": ids,
                "markers": markers
            })
        
        n = len(items)
        L = max(len(it["ids"]) for it in items)
        K = max(len(it["markers"]) for it in items)
        
        input_ids = np.full((n, L), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((n, L), dtype=np.int64)
        marker_pos = np.zeros((n, K), dtype=np.int64)
        marker_mask = np.zeros((n, K), dtype=bool)
        qtype_arr = np.zeros((n,), dtype=np.int64)
        
        for i, it in enumerate(items):
            seq_len = len(it["ids"])
            input_ids[i, :seq_len] = it["ids"]
            attention_mask[i, :seq_len] = 1
            m_len = len(it["markers"])
            marker_pos[i, :m_len] = it["markers"]
            marker_mask[i, :m_len] = True
            qtype_arr[i] = it["qtype_num"]
            
        logits, act_probs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": qtype_arr,
        })
        
        answers = {}
        for i, it in enumerate(items):
            qid = it["qid"]
            k = len(it["markers"])
            qtype = it["qtype"]
            
            sz_b = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
            b_key = f"{qtype}:{sz_b}"
            temp = self.config["temperature_by_options"].get(b_key, self.config["temperature"][it["qtype_num"]])
            
            raw_logits = logits[i, :k] / temp
            e = np.exp(raw_logits - np.max(raw_logits))
            p = (e / np.sum(e)).tolist()
            
            if k < 2:
                conf = 1.0
            else:
                ent = -sum(x * math.log(max(x, 1e-12)) for x in p)
                conf = 1.0 - ent / math.log(k)
                
            if qtype == "choice":
                keys = list(it["crit"].keys())
                best_idx = int(np.argmax(p))
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[best_idx],
                    "probabilities": {keys[j]: round(p[j], 4) for j in range(len(keys))},
                    "confidence": round(conf, 4)
                }
            elif qtype == "score":
                ev = sum(j * p[j] for j in range(len(p)))
                answers[qid] = {
                    "type": "score",
                    "score": round(ev, 4),
                    "probabilities": {str(j): round(p[j], 4) for j in range(len(p))},
                    "confidence": round(conf, 4)
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(p[1], 4)
                }
        return {"answers": answers}


class SnakeGame:
    DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    OPPOSITES = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        mid_x, mid_y = width // 2, height // 2
        self.snake = [(mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y)]
        self.score = 0
        self.steps = 0
        self.alive = True
        self.last_direction = "RIGHT"
        self.food = self._spawn_food()

    def _spawn_food(self):
        empty = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in self.snake]
        return random.choice(empty) if empty else (-1, -1)

    def is_collision(self, pt):
        x, y = pt
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return pt in self.snake[:-1]

    def get_state(self):
        hx, hy = self.snake[0]
        fx, fy = self.food
        analysis = {}
        for d, (dx, dy) in self.DELTAS.items():
            nxt = (hx + dx, hy + dy)
            collides = self.is_collision(nxt)
            is_reverse = (d == self.OPPOSITES[self.last_direction]) and len(self.snake) > 1
            dist = abs(nxt[0] - fx) + abs(nxt[1] - fy)
            if collides or is_reverse:
                analysis[d] = f"DEADLY: {'reverse into body' if is_reverse else 'collision'}"
            else:
                analysis[d] = f"safe path, distance to food = {dist}"
        
        return {
            "game": "Snake",
            "grid": f"{self.width}x{self.height}",
            "head": [hx, hy],
            "food": [fx, fy],
            "steps": self.steps,
            "score": self.score,
            "options_analysis": analysis
        }

    def step(self, direction):
        if not self.alive:
            return False, False
        dx, dy = self.DELTAS[direction]
        hx, hy = self.snake[0]
        new_head = (hx + dx, hy + dy)
        
        if self.is_collision(new_head):
            self.alive = False
            return False, False
            
        self.snake.insert(0, new_head)
        self.last_direction = direction
        self.steps += 1
        
        ate_food = (new_head == self.food)
        if ate_food:
            self.score += 1
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        return True, ate_food

    def render_ascii(self):
        grid = [[" · " for _ in range(self.width)] for _ in range(self.height)]
        if self.food != (-1, -1):
            grid[self.food[1]][self.food[0]] = " 🍎"
        for bx, by in self.snake[1:]:
            grid[by][bx] = " 🟩"
        hx, hy = self.snake[0]
        grid[hy][hx] = " 🟢" if self.alive else " 💥"
        
        border = "+" + "---" * self.width + "+"
        lines = [border] + ["|" + "".join(row) + "|" for row in grid] + [border]
        return "\n".join(lines)


def main():
    print("📥 Loading Laya model from Hugging Face...")
    model_dir = snapshot_download("receptron/laya-onnx")
    agent = PythonLayaAgent(model_dir)
    print("✅ Model loaded successfully!")

    game = SnakeGame(8, 8)
    print(game.render_ascii())

    for _ in range(30):
        if not game.alive:
            break
        state = game.get_state()
        t0 = time.perf_counter()
        res = agent.system_one(state, {
            "next_move": {
                "type": "choice",
                "instructions": "Pick the safest direction towards food.",
                "criteria": state["options_analysis"]
            },
            "danger_level": {
                "type": "score",
                "instructions": "Assess immediate collision risk.",
                "criteria": ["safe open", "mild hazard", "high risk", "death trap"]
            },
            "viable_path": {
                "type": "noul",
                "instructions": "Is there a viable open path?"
            }
        })
        dt = (time.perf_counter() - t0) * 1000
        mv = res["answers"]["next_move"]["choice"]
        conf = res["answers"]["next_move"]["confidence"]
        danger = res["answers"]["danger_level"]["score"]
        print(f"\nStep {game.steps + 1} ({dt:.1f}ms) -> Move: {mv} (Conf: {conf*100:.1f}%) | Danger: {danger:.2f}/3")
        game.step(mv)
        print(game.render_ascii())
        time.sleep(0.3)


if __name__ == "__main__":
    main()
