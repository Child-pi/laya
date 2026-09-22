"""
🐍 Laya System-1 Decision Agent: Gradio Web UI for Snake Game (貪吃蛇)
Optimized for Google Colab with robust queue handling, zero-flicker DOM diffing,
and timeout-prevention (supporting both gr.Timer and non-blocking streaming).
"""

import os
import sys
import json
import math
import time
import random
import numpy as np

try:
    import gradio as gr
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
except ImportError:
    print("Dependencies missing. In Colab run: !pip install gradio onnxruntime huggingface_hub transformers numpy")
    sys.exit(1)


# ==========================================
# 1. Laya System-1 Model Inference Engine
# ==========================================
class LayaAgent:
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
            ids, markers = self._build_sequence(state, q["type"], q["instructions"], q.get("criteria"))
            items.append({
                "qid": qid,
                "qtype": q["type"],
                "qtype_num": self.QTYPES[q["type"]],
                "crit": q.get("criteria"),
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
            input_ids[i, :len(it["ids"])] = it["ids"]
            attention_mask[i, :len(it["ids"])] = 1
            marker_pos[i, :len(it["markers"])] = it["markers"]
            marker_mask[i, :len(it["markers"])] = True
            qtype_arr[i] = it["qtype_num"]
        logits, _ = self.session.run(None, {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "marker_pos": marker_pos, "marker_mask": marker_mask, "qtype": qtype_arr
        })
        answers = {}
        for i, it in enumerate(items):
            qid, k = it["qid"], len(it["markers"])
            sz_b = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
            temp = self.config["temperature_by_options"].get(f"{it['qtype']}:{sz_b}", self.config["temperature"][it["qtype_num"]])
            raw_logits = logits[i, :k] / temp
            e = np.exp(raw_logits - np.max(raw_logits))
            p = (e / np.sum(e)).tolist()
            conf = 1.0 if k < 2 else 1.0 - (-sum(x * math.log(max(x, 1e-12)) for x in p)) / math.log(k)
            if it["qtype"] == "choice":
                keys = list(it["crit"].keys())
                answers[qid] = {
                    "choice": keys[int(np.argmax(p))],
                    "probabilities": {keys[j]: round(p[j], 4) for j in range(len(keys))},
                    "confidence": round(conf, 4)
                }
            elif it["qtype"] == "score":
                answers[qid] = {
                    "score": round(sum(j * p[j] for j in range(len(p))), 4),
                    "confidence": round(conf, 4)
                }
            else:
                answers[qid] = {"noul": round(p[1], 4)}
        return answers


# ==========================================
# 2. Snake Game Logic
# ==========================================
class SnakeGame:
    DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    OPPOSITES = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        mid_x, mid_y = self.width // 2, self.height // 2
        self.snake = [(mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y)]
        self.score = 0
        self.steps = 0
        self.alive = True
        self.last_direction = "RIGHT"
        self.food = self._spawn_food()
        return self

    def _spawn_food(self):
        empty = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in self.snake]
        return random.choice(empty) if empty else (-1, -1)

    def is_collision(self, pt):
        x, y = pt
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return pt in self.snake[:-1]

    def get_perceptual_state(self):
        hx, hy = self.snake[0]
        fx, fy = self.food
        analysis = {}
        for d, (dx, dy) in self.DELTAS.items():
            nxt = (hx + dx, hy + dy)
            collides = self.is_collision(nxt)
            is_reverse = (d == self.OPPOSITES[self.last_direction]) and len(self.snake) > 1
            dist = abs(nxt[0] - fx) + abs(nxt[1] - fy)
            if collides or is_reverse:
                analysis[d] = f"DEADLY: {'reverse' if is_reverse else 'collision'}"
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


# ==========================================
# 3. HTML/SVG UI Generator (Zero Flicker)
# ==========================================
def render_board_html(game, answers=None, latency_ms=0.0):
    cell_size = 40
    w_px = game.width * cell_size
    h_px = game.height * cell_size

    grid_cells = ""
    for y in range(game.height):
        for x in range(game.width):
            pt = (x, y)
            left = x * cell_size
            top = y * cell_size
            if pt == game.snake[0]:
                bg = "#10b981" if game.alive else "#ef4444"
                content = "👀" if game.alive else "💥"
                border_radius = "10px"
                shadow = "box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);" if game.alive else ""
            elif pt in game.snake:
                bg = "linear-gradient(135deg, #34d399, #059669)"
                content = ""
                border_radius = "8px"
                shadow = ""
            elif pt == game.food:
                bg = "radial-gradient(circle, #f87171, #dc2626)"
                content = "🍎"
                border_radius = "50%"
                shadow = "box-shadow: 0 0 14px rgba(239, 68, 68, 0.8);"
            else:
                bg = "#1e293b"
                content = ""
                border_radius = "6px"
                shadow = ""

            grid_cells += f"""
            <div style="position: absolute; left: {left}px; top: {top}px; width: {cell_size - 4}px; height: {cell_size - 4}px;
                        background: {bg}; border-radius: {border_radius}; display: flex; align-items: center; justify-content: center;
                        font-size: 20px; user-select: none; transition: all 0.15s ease-in-out; {shadow}">
                {content}
            </div>
            """

    if answers:
        choice = answers["next_move"]["choice"]
        conf = answers["next_move"]["confidence"] * 100
        probs = answers["next_move"]["probabilities"]
        danger = answers["danger_level"]["score"]
        viability = answers["viable_path"]["noul"] * 100

        prob_bars = ""
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            p = probs.get(d, 0.0) * 100
            is_best = (d == choice)
            bar_color = "#10b981" if is_best else "#475569"
            badge = "<span style='color: #4ade80; font-weight: bold;'>★</span>" if is_best else ""
            prob_bars += f"""
            <div style="margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #f1f5f9;">
                    <span>{badge} {d}</span>
                    <span>{p:.1f}%</span>
                </div>
                <div style="background: #334155; height: 7px; border-radius: 4px; overflow: hidden; margin-top: 2px;">
                    <div style="width: {p}%; height: 100%; background: {bar_color}; transition: width 0.2s ease;"></div>
                </div>
            </div>
            """
        status_badge = "<span style='background: #065f46; color: #34d399; padding: 4px 10px; border-radius: 9999px; font-size: 13px; font-weight: 600;'>🟢 存活中</span>" if game.alive else "<span style='background: #991b1b; color: #fca5a5; padding: 4px 10px; border-radius: 9999px; font-size: 13px; font-weight: 600;'>💥 遊戲結束</span>"
    else:
        choice = "等待中"
        conf = 0
        danger = 0
        viability = 100
        prob_bars = "<div style='color: #94a3b8; font-size: 13px;'>點擊下方按鈕開始模擬</div>"
        status_badge = "<span style='background: #334155; color: #94a3b8; padding: 4px 10px; border-radius: 9999px; font-size: 13px;'>⏳ 就緒</span>"

    html = f"""
    <div style="background: #0f172a; border-radius: 16px; padding: 20px; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4); max-width: 800px; margin: 0 auto;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 12px;">
            <div>
                <h2 style="margin: 0; font-size: 20px; color: #38bdf8; display: flex; align-items: center; gap: 8px;">
                    🐍 Laya System-1 貪吃蛇決策儀表板
                </h2>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 3px;">無自回歸延遲 • 單次 Forward Pass 預測全決策</div>
            </div>
            <div>{status_badge}</div>
        </div>

        <div style="display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap;">
            <div style="position: relative; width: {w_px}px; height: {h_px}px; background: #0b1120; border-radius: 12px; padding: 2px; border: 2px solid #334155;">
                {grid_cells}
            </div>

            <div style="flex: 1; min-width: 260px; display: flex; flex-direction: column; gap: 12px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div style="background: #1e293b; padding: 10px; border-radius: 8px; border-left: 4px solid #38bdf8;">
                        <div style="font-size: 12px; color: #94a3b8;">當前得分</div>
                        <div style="font-size: 22px; font-weight: 700; color: #fbbf24;">🍎 {game.score}</div>
                    </div>
                    <div style="background: #1e293b; padding: 10px; border-radius: 8px; border-left: 4px solid #a855f7;">
                        <div style="font-size: 12px; color: #94a3b8;">存活步數</div>
                        <div style="font-size: 22px; font-weight: 700; color: #c084fc;">👣 {game.steps}</div>
                    </div>
                </div>

                <div style="background: #1e293b; padding: 14px; border-radius: 12px; border: 1px solid #334155;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 13px; color: #94a3b8; font-weight: 600;">⚡ System-1 即時決策</span>
                        <span style="font-size: 11px; background: #0284c7; color: #e0f2fe; padding: 2px 8px; border-radius: 6px;">{latency_ms:.1f} ms</span>
                    </div>
                    <div style="font-size: 15px; margin-bottom: 10px;">
                        方向: <b style="color: #4ade80; font-size: 18px;">{choice}</b> 
                        <span style="font-size: 12px; color: #94a3b8; margin-left: 6px;">(信心度: {conf:.1f}%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #cbd5e1; margin-bottom: 10px; background: #0f172a; padding: 6px 10px; border-radius: 8px;">
                        <span>危險指數: <b>{danger:.2f}/3.0</b></span>
                        <span>路徑可行性: <b>{viability:.1f}%</b></span>
                    </div>
                    <div style="font-size: 12px; color: #94a3b8; margin-bottom: 6px; font-weight: 600;">各方向機率分佈 (Softmax):</div>
                    {prob_bars}
                </div>
            </div>
        </div>
    </div>
    """
    return html


# ==========================================
# 4. Gradio Interface Construction
# ==========================================
class ControlState:
    is_playing = False

def build_gradio_app():
    print("📥 載入 Laya 模型中...")
    model_dir = snapshot_download("receptron/laya-onnx", allow_patterns=["laya.onnx", "laya.onnx.data", "laya_config.json", "tokenizer/*"])
    agent = LayaAgent(model_dir)
    print("✅ Laya 模型準備就緒！")

    game = SnakeGame(8, 8)
    ctrl = ControlState()

    has_timer = hasattr(gr, "Timer")

    with gr.Blocks(title="Laya 貪吃蛇 - System-1 Decision Agent") as demo:
        board_display = gr.HTML(value=render_board_html(game))

        with gr.Row():
            btn_start = gr.Button("▶ 開始自動遊玩 (Auto Play)", variant="primary", scale=2)
            btn_step = gr.Button("⏭ 單步決策 (Step)", variant="secondary", scale=1)
            btn_pause = gr.Button("⏸ 暫停 (Pause)", scale=1)
            btn_reset = gr.Button("🔄 重新開局 (Reset)", scale=1)

        with gr.Row():
            speed_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.35, step=0.05, label="⏱ 步進間隔速度 (秒)")

        with gr.Accordion("🔍 檢視 Laya 模型輸入與原始輸出 (Debug State)", open=False):
            state_json = gr.JSON(label="最新感知狀態與決策結果")

        def do_one_step():
            if not game.alive:
                return render_board_html(game), {"status": "Game Over"}
            state = game.get_perceptual_state()
            t0 = time.perf_counter()
            answers = agent.system_one(state, {
                "next_move": {
                    "type": "choice",
                    "instructions": "Pick the safest direction towards food and away from obstacles.",
                    "criteria": state["options_analysis"]
                },
                "danger_level": {
                    "type": "score",
                    "instructions": "Assess current collision danger.",
                    "criteria": ["safe", "caution", "danger", "deadly"]
                },
                "viable_path": {
                    "type": "noul",
                    "instructions": "Is there a safe viable path to advance?"
                }
            })
            latency = (time.perf_counter() - t0) * 1000
            mv = answers["next_move"]["choice"]
            game.step(mv)
            html = render_board_html(game, answers, latency)
            debug_info = {"perceptual_state": state, "laya_answers": answers, "latency_ms": latency}
            return html, debug_info

        def on_reset():
            ctrl.is_playing = False
            game.reset()
            return render_board_html(game), {"status": "Reset", "score": 0}

        def on_pause():
            ctrl.is_playing = False
            return render_board_html(game), {"status": "Paused"}

        # Safe streaming auto-play with cooperative flag and limited yield duration
        def auto_play_loop(delay):
            ctrl.is_playing = True
            max_steps_per_run = 60
            steps = 0
            while ctrl.is_playing and game.alive and steps < max_steps_per_run:
                html, debug_info = do_one_step()
                steps += 1
                yield html, debug_info
                time.sleep(delay)
            ctrl.is_playing = False
            yield render_board_html(game), {"status": "Idle / Stopped"}

        # Use gr.Timer if available in modern Gradio, otherwise fall back to cooperative generator
        if has_timer:
            timer = gr.Timer(value=0.35, active=False)
            def timer_tick():
                if not game.alive:
                    return render_board_html(game), {"status": "Game Over"}, gr.Timer(active=False)
                html, debug_info = do_one_step()
                if not game.alive:
                    return html, debug_info, gr.Timer(active=False)
                return html, debug_info, gr.Timer(active=True)

            timer.tick(fn=timer_tick, outputs=[board_display, state_json, timer])
            btn_start.click(lambda: gr.Timer(active=True), outputs=[timer])
            btn_pause.click(lambda: gr.Timer(active=False), outputs=[timer])
            btn_reset.click(fn=on_reset, outputs=[board_display, state_json])
            speed_slider.change(lambda v: gr.Timer(value=v), inputs=[speed_slider], outputs=[timer])
            btn_step.click(fn=do_one_step, outputs=[board_display, state_json])
        else:
            btn_step.click(fn=do_one_step, outputs=[board_display, state_json])
            btn_reset.click(fn=on_reset, outputs=[board_display, state_json])
            btn_start.click(fn=auto_play_loop, inputs=[speed_slider], outputs=[board_display, state_json])
            btn_pause.click(fn=on_pause, outputs=[board_display, state_json])

    # Enable queue with concurrency limit to prevent proxy timeouts
    demo.queue(default_concurrency_limit=5)
    return demo


def launch_in_colab():
    app = build_gradio_app()
    # In Colab: server_name='0.0.0.0' allows external forwarding, inline=True embeds inside the notebook output
    app.launch(share=True, inline=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_in_colab()
