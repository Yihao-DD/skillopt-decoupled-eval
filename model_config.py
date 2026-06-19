"""Single-model API switch for the WHOLE pipeline (one model does everything).

The recipient tests our effect on different closed-source model APIs. To keep every
run clean — ONE model doing optimizer-reflection AND target-rollouts end-to-end (never
mixing two providers) — both the training driver (`run.py`) and the decoupled-selection
eval (`tools/posthoc_select.py`) resolve their model from the SAME place: this module.

Switch APIs by editing THREE lines in `.env`:

    MODEL_PROVIDER=deepseek          # a preset below that fills the base URL
    MODEL_API_KEY=sk-xxxxxxxx
    MODEL_NAME=deepseek-chat         # the model id at that provider

For any other OpenAI-compatible host, set `MODEL_PROVIDER=custom` and `MODEL_BASE_URL=...`.
Every supported provider is reached through the OpenAI-compatible Chat Completions API,
so the same code path serves all of them — no per-provider branching anywhere else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# provider preset -> OpenAI-compatible base URL
PRESETS: dict[str, str] = {
    "openai":      "https://api.openai.com/v1",
    "deepseek":    "https://api.deepseek.com",
    "qwen":        "https://dashscope.aliyuncs.com/compatible-mode/v1",  # Alibaba DashScope (compat mode)
    "moonshot":    "https://api.moonshot.cn/v1",                         # Kimi
    "zhipu":       "https://open.bigmodel.cn/api/paas/v4",               # GLM
    "together":    "https://api.together.xyz/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "custom":      "",   # must supply MODEL_BASE_URL
}


@dataclass(frozen=True)
class ModelConfig:
    """The single model used for both optimizer and target roles."""
    provider: str
    base_url: str
    api_key: str
    model: str

    def masked_key(self) -> str:
        k = self.api_key
        return "(empty)" if not k else (f"{k[:4]}***{k[-2:]}" if len(k) > 8 else "***")


def load_dotenv(path: str | os.PathLike) -> dict[str, str]:
    """Parse a KEY=VALUE .env file (split on first '='; strip quotes/comments)."""
    out: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return out
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def resolve(dotenv_path: str | os.PathLike | None = None,
            extra: dict[str, str] | None = None) -> ModelConfig:
    """Resolve the single model from `.env` (+ process env, which overrides).

    Precedence: explicit `extra` > process env (`os.environ`) > `.env` file.
    Raises a clear, actionable error if anything required is missing.
    """
    merged: dict[str, str] = {}
    if dotenv_path is not None:
        merged.update(load_dotenv(dotenv_path))
    merged.update({k: v for k, v in os.environ.items() if k.startswith("MODEL_")})
    if extra:
        merged.update(extra)

    provider = (merged.get("MODEL_PROVIDER") or "").strip().lower()
    base_url = (merged.get("MODEL_BASE_URL") or "").strip()
    api_key = (merged.get("MODEL_API_KEY") or "").strip()
    model = (merged.get("MODEL_NAME") or "").strip()

    if not provider and not base_url:
        raise SystemExit(
            "[model] Set MODEL_PROVIDER in .env (one of: "
            + ", ".join(k for k in PRESETS if k != "custom")
            + ") — or MODEL_PROVIDER=custom with MODEL_BASE_URL."
        )
    if not base_url:
        if provider not in PRESETS:
            raise SystemExit(
                f"[model] Unknown MODEL_PROVIDER={provider!r}. Known: "
                + ", ".join(PRESETS) + ". For others use MODEL_PROVIDER=custom + MODEL_BASE_URL."
            )
        base_url = PRESETS[provider]
        if not base_url:  # custom with no MODEL_BASE_URL
            raise SystemExit("[model] MODEL_PROVIDER=custom requires MODEL_BASE_URL=... in .env")
    if not api_key:
        raise SystemExit("[model] Set MODEL_API_KEY in .env (the API key for your chosen provider).")
    if not model:
        raise SystemExit("[model] Set MODEL_NAME in .env (the model id, e.g. gpt-4o / deepseek-chat / qwen-plus).")

    return ModelConfig(provider=provider or "custom", base_url=base_url, api_key=api_key, model=model)


if __name__ == "__main__":  # quick check: `python model_config.py` prints the resolved model (key masked)
    import sys
    repo_root = Path(__file__).resolve().parent
    cfg = resolve(dotenv_path=repo_root / ".env")
    print(f"provider={cfg.provider}  base_url={cfg.base_url}  model={cfg.model}  key={cfg.masked_key()}",
          file=sys.stderr)
