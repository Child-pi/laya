import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import sonarjs from "eslint-plugin-sonarjs";
import prettierRecommended from "eslint-plugin-prettier/recommended";

export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", "onnx/**", "export/**"] },

  {
    linterOptions: {
      reportUnusedDisableDirectives: "error",
      // An `eslint-disable` comment removes a violation from the linter and from any count of it
      // alike — an exemption nothing can see afterwards.
      noInlineConfig: true,
    },
  },

  js.configs.recommended,
  tseslint.configs.recommendedTypeChecked,
  sonarjs.configs.recommended,
  // Last: it switches off every rule that would argue with the formatter.
  prettierRecommended,

  {
    languageOptions: {
      globals: globals.node,
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
  },

  {
    rules: {
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-non-null-assertion": "error",
      "@typescript-eslint/consistent-type-assertions": ["error", { assertionStyle: "never" }],
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^__" }],
      "@typescript-eslint/prefer-readonly": "error",
      "no-console": "off",
    },
  },

  {
    rules: {
      "no-var": "error",
      "prefer-const": "error",
      "no-param-reassign": "error",
      "no-else-return": ["error", { allowElseIf: false }],
      "sonarjs/no-collapsible-if": "error",
    },
  },

  {
    rules: {
      "max-lines": ["error", { max: 300, skipBlankLines: true, skipComments: true }],
      "max-lines-per-function": ["error", { max: 60, skipBlankLines: true, skipComments: true }],
      complexity: ["error", 15],
      "max-depth": ["error", 4],
      "max-nested-callbacks": ["error", 4],
      "max-params": ["error", 6],
      "sonarjs/cognitive-complexity": ["error", 15],
    },
  },

  {
    // A spec's arrange block repeats by nature, and its length is not a comprehension problem.
    files: ["test/**/*.ts"],
    rules: {
      "max-lines": "off",
      "max-lines-per-function": "off",
      "sonarjs/no-duplicate-string": "off",
      // node:test's `describe`/`it` return promises the runner owns. Awaiting them is wrong, and
      // `void`-prefixing every block would be noise on every line of every spec.
      "@typescript-eslint/no-floating-promises": "off",
    },
  },

  {
    // The drain backlog. Every rule below is violated by code that predates this config, so it is
    // reported without failing the build while the backlog is worked down. A rule leaves this block
    // the moment its last violation is gone — and a rule that is NOT listed here is an error, so
    // new code is held to the full set. Nothing here is a judgement that the rule is wrong.
    rules: {
      "@typescript-eslint/consistent-type-assertions": ["warn", { assertionStyle: "never" }],
      "@typescript-eslint/no-unnecessary-type-assertion": "warn",
      "@typescript-eslint/no-base-to-string": "warn",
      "@typescript-eslint/prefer-readonly": "warn",
      "@typescript-eslint/require-await": "warn",
      "sonarjs/no-nested-conditional": "warn",
      "sonarjs/cognitive-complexity": ["warn", 15],
      "sonarjs/super-linear-regex": "warn",
      "sonarjs/reduce-initial-value": "warn",
      complexity: ["warn", 15],
      "max-lines-per-function": ["warn", { max: 60, skipBlankLines: true, skipComments: true }],
    },
  },

  {
    // Not part of the TypeScript program: the config file cannot reference itself.
    files: ["eslint.config.js"],
    languageOptions: { parserOptions: { projectService: false } },
    ...tseslint.configs.disableTypeChecked,
  },
);
