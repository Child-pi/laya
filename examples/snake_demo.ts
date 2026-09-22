/**
 * Snake Game (貪吃蛇) using Laya as a System-1 Decision Agent
 *
 * In this demo, Laya acts as the "brain" (System-1 instinctive decision maker).
 * At each game tick, the current board perception (head, food, surrounding hazards, distances)
 * is passed to Laya. In a single forward pass, Laya evaluates:
 *  1. `move` (choice): The next direction (UP, DOWN, LEFT, RIGHT)
 *  2. `safety_score` (score): Risk assessment of current surroundings (0..3)
 *  3. `food_opportunity` (noul): Probability that a safe path to food is open
 */

import { Laya } from "../src/index.js";

type Point = [number, number];
type Direction = "UP" | "DOWN" | "LEFT" | "RIGHT";

const DELTAS: Record<Direction, Point> = {
  UP: [0, -1],
  DOWN: [0, 1],
  LEFT: [-1, 0],
  RIGHT: [1, 0],
};

const OPPOSITES: Record<Direction, Direction> = {
  UP: "DOWN",
  DOWN: "UP",
  LEFT: "RIGHT",
  RIGHT: "LEFT",
};

export class SnakeGame {
  width: number;
  height: number;
  snake: Point[];
  food: Point;
  score: number = 0;
  steps: number = 0;
  alive: boolean = true;
  lastDirection: Direction = "RIGHT";

  constructor(width = 8, height = 8) {
    this.width = width;
    this.height = height;
    const midX = Math.floor(width / 2);
    const midY = Math.floor(height / 2);
    this.snake = [
      [midX, midY],
      [midX - 1, midY],
      [midX - 2, midY],
    ];
    this.food = this.spawnFood();
  }

  private spawnFood(): Point {
    const empty: Point[] = [];
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        if (!this.snake.some(([sx, sy]) => sx === x && sy === y)) {
          empty.push([x, y]);
        }
      }
    }
    if (empty.length === 0) return [-1, -1];
    return empty[Math.floor(Math.random() * empty.length)]!;
  }

  isCollision(p: Point): boolean {
    const [x, y] = p;
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) return true;
    return this.snake.slice(0, -1).some(([sx, sy]) => sx === x && sy === y);
  }

  step(dir: Direction): { alive: boolean; ateFood: boolean } {
    if (!this.alive) return { alive: false, ateFood: false };

    const head = this.snake[0]!;
    const [dx, dy] = DELTAS[dir];
    const newHead: Point = [head[0] + dx, head[1] + dy];

    if (this.isCollision(newHead)) {
      this.alive = false;
      return { alive: false, ateFood: false };
    }

    this.snake.unshift(newHead);
    this.lastDirection = dir;
    this.steps++;

    const ateFood = newHead[0] === this.food[0] && newHead[1] === this.food[1];
    if (ateFood) {
      this.score++;
      this.food = this.spawnFood();
    } else {
      this.snake.pop();
    }

    return { alive: true, ateFood };
  }

  /**
   * Builds the perceptual state for Laya's System-1 model.
   */
  getPerceptualState() {
    const head = this.snake[0]!;
    const [hx, hy] = head;
    const [fx, fy] = this.food;

    const directions: Direction[] = ["UP", "DOWN", "LEFT", "RIGHT"];
    const lookahead: Record<string, { safe: boolean; distance_to_food: number; hazard: string }> = {};

    for (const d of directions) {
      const [dx, dy] = DELTAS[d];
      const nextPos: Point = [hx + dx, hy + dy];
      const collides = this.isCollision(nextPos);
      const isReverse = d === OPPOSITES[this.lastDirection] && this.snake.length > 1;
      const dist = Math.abs(nextPos[0] - fx) + Math.abs(nextPos[1] - fy);

      lookahead[d] = {
        safe: !collides && !isReverse,
        distance_to_food: dist,
        hazard: collides ? "WALL_OR_BODY_COLLISION" : isReverse ? "REVERSE_COLLISION" : "NONE",
      };
    }

    const relX = fx > hx ? "EAST/RIGHT" : fx < hx ? "WEST/LEFT" : "ALIGNED";
    const relY = fy > hy ? "SOUTH/DOWN" : fy < hy ? "NORTH/UP" : "ALIGNED";

    return {
      game: "Snake",
      board_size: `${this.width}x${this.height}`,
      current_step: this.steps,
      score: this.score,
      snake_length: this.snake.length,
      head_position: [hx, hy],
      food_position: [fx, fy],
      food_relative_location: `${relY}, ${relX}`,
      options_analysis: {
        UP: lookahead.UP.safe
          ? `safe path, distance to food = ${lookahead.UP.distance_to_food}`
          : `DEADLY: ${lookahead.UP.hazard}`,
        DOWN: lookahead.DOWN.safe
          ? `safe path, distance to food = ${lookahead.DOWN.distance_to_food}`
          : `DEADLY: ${lookahead.DOWN.hazard}`,
        LEFT: lookahead.LEFT.safe
          ? `safe path, distance to food = ${lookahead.LEFT.distance_to_food}`
          : `DEADLY: ${lookahead.LEFT.hazard}`,
        RIGHT: lookahead.RIGHT.safe
          ? `safe path, distance to food = ${lookahead.RIGHT.distance_to_food}`
          : `DEADLY: ${lookahead.RIGHT.hazard}`,
      },
    };
  }

  renderAscii(): string {
    const grid: string[][] = Array.from({ length: this.height }, () =>
      Array.from({ length: this.width }, () => " · ")
    );

    // Render food
    if (this.food[0] >= 0 && this.food[1] >= 0) {
      grid[this.food[1]]![this.food[0]] = " 🍎";
    }

    // Render snake body
    for (let i = 1; i < this.snake.length; i++) {
      const [bx, by] = this.snake[i]!;
      grid[by]![bx] = " 🟩";
    }

    // Render head
    if (this.snake.length > 0) {
      const [hx, hy] = this.snake[0]!;
      grid[hy]![hx] = this.alive ? " 🟢" : " 💥";
    }

    const border = "+" + "---".repeat(this.width) + "+";
    const rows = grid.map((row) => "|" + row.join("") + "|");
    return [border, ...rows, border].join("\n");
  }
}

async function main() {
  console.log("🎮 Initializing Laya System-1 Decision Model for Snake (貪吃蛇)...");
  
  // Loads Laya (downloads from Hugging Face receptron/laya-onnx if not cached)
  const laya = await Laya.load({
    onProgress: ({ file, received, total }) => {
      if (total) {
        process.stderr.write(`\r[Model Download] ${file}: ${((received / total) * 100).toFixed(1)}% `);
      }
    },
  });
  console.log("\n✅ Laya loaded successfully!\n");

  const game = new SnakeGame(8, 8);
  console.log("Initial Board:\n" + game.renderAscii() + "\n");

  const maxSteps = 25;
  while (game.alive && game.steps < maxSteps) {
    const state = game.getPerceptualState();

    const t0 = performance.now();
    // Hand Laya the board state and 3 typed questions in ONE forward pass
    const result = await laya.systemOne(state, {
      next_move: {
        type: "choice",
        instructions: "As the snake controller, pick the safest direction that makes progress toward food and avoids death.",
        criteria: {
          UP: state.options_analysis.UP,
          DOWN: state.options_analysis.DOWN,
          LEFT: state.options_analysis.LEFT,
          RIGHT: state.options_analysis.RIGHT,
        },
      },
      danger_level: {
        type: "score",
        instructions: "How hazardous is the current position to the snake?",
        criteria: ["safe", "caution", "high risk", "trapped"],
      },
      has_clear_path: {
        type: "noul",
        instructions: "Is there a safe viable path to advance?",
      },
    });
    const latency = (performance.now() - t0).toFixed(1);

    const chosenDirection = result.answers.next_move.choice as Direction;
    const confidence = (result.answers.next_move.confidence * 100).toFixed(1);
    const dangerScore = result.answers.danger_level.score.toFixed(2);
    const hasPathProb = (result.answers.has_clear_path.noul * 100).toFixed(1);

    console.log(`--- Step ${game.steps + 1} (${latency} ms) ---`);
    console.log(`Decision: ${chosenDirection} (Confidence: ${confidence}%) | Danger Score: ${dangerScore}/3 | Safe Path: ${hasPathProb}%`);
    console.log(`Probabilities:`, result.answers.next_move.probabilities);

    const stepResult = game.step(chosenDirection);
    console.log(game.renderAscii());

    if (stepResult.ateFood) {
      console.log(`🎉 Ate food! Current Score: ${game.score}`);
    }

    if (!stepResult.alive) {
      console.log("💀 Game Over! Snake crashed.");
      break;
    }

    // Small delay for viewing
    await new Promise((r) => setTimeout(r, 400));
  }

  console.log(`\nGame finished. Final Score: ${game.score}, Total Steps: ${game.steps}`);
  await laya.close();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}
