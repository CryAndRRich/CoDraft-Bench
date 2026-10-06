DEFAULTS = dict(
    backend="vllm",
    dtype="half",
    tensor_parallel=2,
    max_model_len=16384,
    system="native",
    chat_kwargs=None,
    gated=False,
    gpu_memory_utilization=0.90,
    batch_size=10,
    workers=16,
    rescue_penalty=1.15,
)

LLMS = {
    "qwen2.5-7b": dict(model="Qwen/Qwen2.5-7B-Instruct"),
    "qwen3-8b": dict(model="Qwen/Qwen3-8B", chat_kwargs={"enable_thinking": False}),
    "llama3.1-8b": dict(model="meta-llama/Llama-3.1-8B-Instruct", gated=True),
    "nemotron-nano-8b": dict(
        model="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        system="detailed thinking off",
    ),
    "gemini-2.5-flash": dict(
        model="gemini-2.5-flash",
        backend="api",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        batch_size=30,
        workers=1,
    ),
}


def safe_tag(tag: str) -> str:
    return tag.split("/")[-1].lower().replace("_", "-")


def get_llm(tag: str) -> dict:
    if tag in LLMS:
        spec = {**DEFAULTS, **LLMS[tag]}
    elif "/" in tag:
        spec = {**DEFAULTS, "model": tag}
    else:
        raise KeyError(
            f"Unknown LLM {tag!r}. Known tags are {sorted(LLMS)}, "
            f'or pass a Hugging Face id such as "Qwen/Qwen2.5-7B-Instruct".'
        )
    spec["tag"] = safe_tag(tag)
    return spec


def describe() -> str:
    lines = [f"{'tag':18s} {'backend':7s} {'tp':>2s}  model"]
    for tag in LLMS:
        spec = {**DEFAULTS, **LLMS[tag]}
        tp = str(spec["tensor_parallel"]) if spec["backend"] == "vllm" else "-"
        gated = "  (gated)" if spec["gated"] else ""
        lines.append(f"{tag:18s} {spec['backend']:7s} {tp:>2s}  {spec['model']}{gated}")
    return "\n".join(lines)
