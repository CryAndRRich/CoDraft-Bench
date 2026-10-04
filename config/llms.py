"""Every LLM the enrichment and the LLM classifier can use, keyed by a short tag.

The notebook (scripts/llm.ipynb) only picks a tag; how a model is served lives here, so two
LLM runs differ only in what this table says. An HF id that is not in the table also works,
with the defaults below.

backend
  vllm     served on the Kaggle GPUs by vLLM's OpenAI-compatible server
  api      a hosted OpenAI-compatible endpoint (Gemini); needs an API key

Served on a Kaggle T4 x2: fp16 only (Turing has no bf16), so a 7-9B model in fp16 takes
both cards (tensor_parallel=2). A model that needs bf16 to be stable must be tested first.

system     how the chat template takes the system prompt
  native   as a system message
  merge    the template has no system role (Gemma 2): prepended to the user message
  "<text>" the model expects this exact system prompt (Nemotron's reasoning switch); ours
           is prepended to the user message instead
chat_kwargs  passed to the chat template, e.g. Qwen3's thinking switch
gated        the HF repo needs an accepted licence and HF_TOKEN
"""

DEFAULTS = dict(backend="vllm", dtype="half", tensor_parallel=2, max_model_len=16384,
                quantization=None, revision=None, system="native", chat_kwargs=None,
                gated=False, gpu_memory_utilization=0.90, enforce_eager=False,
                batch_size=10, workers=16)

LLMS = {
    # ---- the candidates (one per family; pick 2-3 after the smoke test) --------------
    "qwen2.5-7b":   dict(model="Qwen/Qwen2.5-7B-Instruct"),
    "qwen3-8b":     dict(model="Qwen/Qwen3-8B", chat_kwargs={"enable_thinking": False}),
    "llama3.1-8b":  dict(model="meta-llama/Llama-3.1-8B-Instruct", gated=True),
    # Gemma 2's template rejects a system role. vLLM warns that Gemma 2 can overflow in
    # fp16; the smoke test shows whether it does on this prompt.
    # Its context is 8k, so batches are halved: ten terms of a class with a long heading
    # (class 9) plus their answers would not fit.
    "gemma2-9b":    dict(model="google/gemma-2-9b-it", gated=True, system="merge",
                         max_model_len=8192, batch_size=5),
    # Llama-3.1 architecture; reasoning is switched off through the system prompt.
    "nemotron-nano-8b": dict(model="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                             system="detailed thinking off"),

    # ---- the original enrichment source, through the same client ---------------------
    "gemini-2.5-flash": dict(model="gemini-2.5-flash", backend="api",
                             base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                             api_key_env="GEMINI_API_KEY", batch_size=30, workers=1),
}


def get_llm(tag):
    """The serving spec for a registry tag, or for a bare HF id with the defaults."""
    if tag in LLMS:
        spec = {**DEFAULTS, **LLMS[tag]}
    elif "/" in tag:
        spec = {**DEFAULTS, "model": tag}
    else:
        raise KeyError(f"unknown LLM {tag!r}; known tags are {sorted(LLMS)}, "
                       f"or pass a Hugging Face id such as 'Qwen/Qwen2.5-7B-Instruct'")
    spec["tag"] = safe_tag(tag)
    return spec


def safe_tag(tag):
    """A tag usable in file and folder names: 'Qwen/Qwen2.5-7B-Instruct' -> 'qwen2.5-7b-instruct'."""
    return tag.split("/")[-1].lower().replace("_", "-")


def describe():
    lines = [f"{'tag':18s} {'backend':7s} {'tp':>2s}  model"]
    for t in LLMS:
        s = {**DEFAULTS, **LLMS[t]}
        tp = str(s["tensor_parallel"]) if s["backend"] == "vllm" else "-"
        lines.append(f"{t:18s} {s['backend']:7s} {tp:>2s}  {s['model']}"
                     + ("  (gated)" if s["gated"] else ""))
    return "\n".join(lines)
