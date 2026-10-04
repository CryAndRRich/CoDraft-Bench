"""One client for every LLM: a model served by vLLM on the local GPUs, or a hosted
OpenAI-compatible API (Gemini). Both are called through the openai package, and every
answer is constrained to a JSON schema, so a reply either parses or is retried.
"""
import copy
import json
import os
import subprocess
import sys
import time
import urllib.request

import openai


class VLLMServer:
    """vLLM's OpenAI-compatible server in a child process, for the length of a with-block.

    The server's own output goes to log_path; if it dies while loading, the tail of that log
    is raised so the notebook shows why.
    """

    def __init__(self, spec, port=8000, log_path="vllm.log", seed=0, startup_timeout=3600):
        self.spec, self.port, self.log_path = spec, port, log_path
        self.seed, self.startup_timeout = seed, startup_timeout
        self.proc = None
        self.base_url = f"http://127.0.0.1:{port}/v1"

    def command(self):
        s = self.spec
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", s["model"], "--served-model-name", s["tag"],
               "--port", str(self.port), "--dtype", s["dtype"],
               "--tensor-parallel-size", str(s["tensor_parallel"]),
               "--max-model-len", str(s["max_model_len"]),
               "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
               "--seed", str(self.seed)]
        if s.get("revision"):
            cmd += ["--revision", s["revision"]]
        if s.get("quantization"):
            cmd += ["--quantization", s["quantization"]]
        if s.get("enforce_eager"):
            cmd.append("--enforce-eager")
        return cmd

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        cmd = self.command()
        print("$", " ".join(cmd))
        self._log = open(self.log_path, "w", buffering=1)
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                self._log.close()
                raise RuntimeError(f"vLLM exited with code {self.proc.returncode} while loading "
                                   f"{self.spec['model']}. Errors in {self.log_path}:\n"
                                   + self.errors() + "\n\nLast lines:\n" + self.tail(15))
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > self.startup_timeout:
                self.__exit__(None, None, None)
                raise TimeoutError(f"vLLM not ready after {self.startup_timeout}s; see {self.log_path}")
            time.sleep(5)
        print(f"vLLM ready in {time.time() - t0:.0f}s: {self.spec['model']} as '{self.spec['tag']}'")
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_log", None) and not self._log.closed:
            self._log.close()

    def tail(self, n=40):
        try:
            return "".join(open(self.log_path).readlines()[-n:])
        except Exception:
            return "(no log)"

    def errors(self, n=25):
        """The lines that name an error, first ones first. When a vLLM worker dies, the API
        server's own traceback at the end only says "see root cause above"; the cause is in
        the worker's lines further up."""
        try:
            lines = open(self.log_path).readlines()
        except Exception:
            return "(no log)"
        keep, seen = [], set()
        for line in lines:
            msg = line.split(")", 1)[-1].strip() if line.startswith("(") else line.strip()
            if any(w in msg for w in ("Error", "error:", "Exception", "CUDA out of memory",
                                       "not supported", "does not support", "Unsupported")) \
                    and "Engine core initialization failed" not in msg and msg not in seen:
                seen.add(msg)
                keep.append(line.rstrip())
        return "\n".join(keep[:n]) or "(no error lines; see the full log)"


def strict_schema(model, max_items=None, min_items=None):
    """The JSON schema of a pydantic model, with array lengths bounded.

    Greedy decoding under a schema can repeat list items until it runs out of tokens; a
    bound on every array stops that. max_items / min_items map a property name to its
    maximum / minimum length.
    """
    schema = copy.deepcopy(model.model_json_schema())
    max_items, min_items = max_items or {}, min_items or {}

    def walk(node):
        if isinstance(node, dict):
            for name, sub in node.get("properties", {}).items():
                if sub.get("type") == "array":
                    if name in max_items:
                        sub["maxItems"] = max_items[name]
                    if name in min_items:
                        sub["minItems"] = min_items[name]
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(schema)
    return schema


class LLMClient:
    """Chat completions constrained to a JSON schema, at temperature 0."""

    # Transport errors in a row (across all threads) before the run is stopped: a dead
    # server would otherwise make every remaining request wait through its retries.
    MAX_CONSECUTIVE_ERRORS = 20

    def __init__(self, spec, base_url, api_key="EMPTY", seed=0, timeout=600):
        self.spec, self.seed = spec, seed
        self.consecutive_errors = 0
        self.model = spec["tag"] if spec["backend"] == "vllm" else spec["model"]
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout,
                                    max_retries=0)

    def messages(self, system, user):
        mode = self.spec.get("system", "native")
        if mode == "native":
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if mode == "merge":
            return [{"role": "user", "content": f"{system}\n\n{user}"}]
        # a fixed system prompt the model requires (e.g. a reasoning switch)
        return [{"role": "system", "content": mode}, {"role": "user", "content": f"{system}\n\n{user}"}]

    def json(self, system, user, response_model, max_tokens, schema=None, attempts=4):
        """(parsed object or None, info). info carries token counts and the last error.

        A reply cut off at max_tokens is retried with twice the budget; a transport error
        is retried after a pause. Greedy decoding gives the same reply to the same prompt,
        so an answer that fails validation is not retried.
        """
        schema = schema or response_model.model_json_schema()
        kw = dict(model=self.model, messages=self.messages(system, user), temperature=0,
                  max_tokens=max_tokens,
                  response_format={"type": "json_schema",
                                   "json_schema": {"name": response_model.__name__,
                                                   "schema": schema, "strict": True}})
        if self.spec["backend"] == "vllm":
            kw["seed"] = self.seed
            if self.spec.get("chat_kwargs"):
                kw["extra_body"] = {"chat_template_kwargs": self.spec["chat_kwargs"]}
        info = {"prompt_tokens": 0, "completion_tokens": 0, "error": None, "calls": 0}
        limit = self.spec.get("max_model_len", 8192)
        for attempt in range(attempts):
            info["calls"] += 1
            try:
                r = self.client.chat.completions.create(**kw)
            except openai.BadRequestError as e:
                # e.g. prompt + max_tokens over the context length: the same request fails again
                info["error"] = f"BadRequestError: {e}"[:300]
                return None, info
            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"[:300]
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(f"{self.consecutive_errors} requests in a row failed; the "
                                       f"LLM endpoint looks down. Last error: {info['error']}")
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            self.consecutive_errors = 0
            if r.usage:
                info["prompt_tokens"] += r.usage.prompt_tokens or 0
                info["completion_tokens"] += r.usage.completion_tokens or 0
            choice = r.choices[0]
            if choice.finish_reason == "length":
                info["error"] = f"cut off at {kw['max_tokens']} tokens"
                if kw["max_tokens"] * 2 > limit:
                    return None, info
                kw["max_tokens"] *= 2
                continue
            try:
                info["error"] = None
                return response_model.model_validate_json(choice.message.content or ""), info
            except Exception as e:
                info["error"] = f"invalid reply: {type(e).__name__}: {e}"[:300]
                return None, info
        return None, info


def make_client(spec, base_url=None, seed=0):
    """The client for a registry spec: the local server, or the hosted API with its key."""
    if spec["backend"] == "vllm":
        return LLMClient(spec, base_url, api_key="EMPTY", seed=seed)
    key = os.environ.get(spec.get("api_key_env", ""), "")
    if not key:
        raise RuntimeError(f"set {spec.get('api_key_env')} to use {spec['model']}")
    return LLMClient(spec, spec["base_url"], api_key=key, seed=seed)


def served_revision(spec):
    """The commit of the HF repo the model was loaded from, so a run names its exact weights."""
    if spec["backend"] != "vllm":
        return None
    try:
        from huggingface_hub import model_info
        return model_info(spec["model"], revision=spec.get("revision")).sha
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def package_versions():
    out = {"python": sys.version.split()[0]}
    for pkg in ("vllm", "torch", "transformers", "openai", "pydantic"):
        try:
            out[pkg] = __import__(pkg).__version__
        except Exception:
            out[pkg] = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        out["gpus"] = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception:
        out["gpus"] = []
    return out


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)
