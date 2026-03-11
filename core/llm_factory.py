from llama_index.core.llms import LLM


class LLMFactory:
    """Creates LlamaIndex LLM instances from a provider string."""

    @staticmethod
    def create(
        provider: str,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model_path: str = "",
        n_ctx: int = 8192,
    ) -> LLM:
        provider = provider.lower()

        if provider == "openai":
            from llama_index.llms.openai import OpenAI

            return OpenAI(
                model=model or "gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "gemini":
            from llama_index.llms.gemini import Gemini

            return Gemini(
                model=model or "models/gemini-2.0-flash",
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider in ("llama_cpp", "llama", "qwen"):
            from llama_index.llms.llama_cpp import LlamaCPP

            if not model_path:
                raise ValueError(f"model_path is required for provider '{provider}'")

            return _build_llama_cpp(
                model_path=model_path,
                target_ctx=n_ctx,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}'")


def _build_llama_cpp(
    model_path: str,
    target_ctx: int,
    max_tokens: int,
    temperature: float,
) -> LLM:
    """Try to load llama.cpp at target context; fall back if VRAM is insufficient."""
    from llama_index.llms.llama_cpp import LlamaCPP

    for n_ctx in (target_ctx, 8192, 4096, 2048):
        try:
            return LlamaCPP(
                model_path=model_path,
                context_window=n_ctx,
                max_new_tokens=min(max_tokens, n_ctx // 2),
                model_kwargs={"n_gpu_layers": -1},
                temperature=temperature,
                verbose=False,
            )
        except Exception as e:
            print(f"Failed to init LlamaCPP with n_ctx={n_ctx}: {e}")

    raise RuntimeError("Could not initialize LlamaCPP; try a smaller n_ctx.")
