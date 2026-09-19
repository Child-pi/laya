import { Laya } from "../src/index.js";

// Pass { modelDir: "./onnx" } to use a local export instead of the Hugging Face bundle.
const laya = await Laya.load({
  modelDir: process.env.LAYA_MODEL_DIR,
  onProgress: ({ file, received, total }) => {
    if (total) process.stderr.write(`\r${file}: ${((received / total) * 100).toFixed(0)}%   `);
  },
});

const state = {
  subject: "Refund not received",
  body: "I cancelled my subscription two weeks ago and I still have not received my refund. This is the third time I am writing. If this is not resolved I will dispute the charge with my bank.",
};

const t0 = performance.now();
const result = await laya.systemOne(state, {
  department: {
    type: "choice",
    instructions: "Which team should handle this ticket?",
    criteria: { billing: "payments, refunds, invoices", support: "product help and bugs", sales: "new purchases and upgrades" },
  },
  urgency: {
    type: "score",
    instructions: "How urgent is this ticket?",
    criteria: ["not urgent", "somewhat urgent", "urgent", "critical"],
  },
  churn_risk: { type: "noul", instructions: "Is the customer likely to cancel or dispute?" },
});
console.log(`${(performance.now() - t0).toFixed(0)} ms`);
console.log(JSON.stringify(result, null, 1));

// the answer types follow the question types
console.log("→", result.answers.department.choice, result.answers.urgency.score, result.answers.churn_risk.noul);
await laya.close();
