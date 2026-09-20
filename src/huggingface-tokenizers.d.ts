/**
 * `@huggingface/tokenizers` is published as `"type": "module"` but its .d.ts files use extensionless
 * relative imports, which `moduleResolution: NodeNext` rejects (TS2834). `skipLibCheck` hides that
 * error and every export degrades to `any`, so the tokenizer calls here type-check against nothing.
 * Signatures below are copied from the package's own declarations; delete this once upstream ships
 * types that resolve.
 */
declare module "@huggingface/tokenizers" {
  export interface Encoding {
    ids: number[];
    tokens: string[];
    attention_mask: number[];
    token_type_ids?: number[];
  }

  export interface EncodeOptions {
    text_pair?: string | null;
    add_special_tokens?: boolean;
    return_token_type_ids?: boolean | null;
  }

  export class Tokenizer {
    constructor(tokenizer: object, config: object);
    encode(text: string, options?: EncodeOptions): Encoding;
    token_to_id(token: string): number | undefined;
    id_to_token(id: number): string | undefined;
  }
}
