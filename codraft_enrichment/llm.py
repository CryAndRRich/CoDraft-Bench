import copy
import os
import subprocess
import sys
import time
import urllib.request

import openai
from pydantic import BaseModel

ERROR_WORDS = (
    "Error",
    "error:",
    "Exception",
    "CUDA out of memory",
    "not supported",
    "does not support",
    "Unsupported",
)


class VLLMServer:
    def __init__(
        self,
        spec: dict,
        port: int = 8000,
        log_path: str = "vllm.log",
        seed: int = 0,
        startup_timeout: int = 3600,
    ) -> None:
        self.spec = spec
        self.port = port
        self.log_path = log_path
        self.seed = seed
        self.startup_timeout = startup_timeout
        self.proc = None
        self.log = None
        self.base_url = f"http://127.0.0.1:{port}/v1"

    def command(self) -> list[str]:
        s = self.spec
        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            s["model"],
            "--served-model-name",
            s["tag"],
            "--port",
            str(self.port),
            "--dtype",
            s["dtype"],
            "--tensor-parallel-size",
            str(s["tensor_parallel"]),
            "--max-model-len",
            str(s["max_model_len"]),
            "--gpu-memory-utilization",
            str(s["gpu_memory_utilization"]),
            "--seed",
            str(self.seed),
        ]

    def __enter__(self) -> "VLLMServer":
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        cmd = self.command()
        print("$", " ".join(cmd))
        self.log = open(self.log_path, "w", buffering=1)
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                self.log.close()
                raise RuntimeError(
                    f"vLLM exited with code {self.proc.returncode} while loading {self.spec['model']}. "
                    f"Errors in {self.log_path}:\n{self.errors()}\n\nLast lines:\n{self.tail(15)}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > self.startup_timeout:
                self.__exit__()
                raise TimeoutError(
                    f"vLLM is not ready after {self.startup_timeout}s. See {self.log_path}."
                )
            time.sleep(5)
        print(
            f'vLLM ready in {time.time() - t0:.0f}s: {self.spec["model"]} as "{self.spec["tag"]}"'
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log and not self.log.closed:
            self.log.close()

    def tail(self, n: int = 40) -> str:
        try:
            with open(self.log_path) as fh:
                return "".join(fh.readlines()[-n:])
        except Exception:
            return "(no log)"

    def errors(self, n: int = 25) -> str:
        try:
            with open(self.log_path) as fh:
                lines = fh.readlines()
        except Exception:
            return "(no log)"
        keep, seen = [], set()
        for line in lines:
            msg = line.split(")", 1)[-1].strip() if line.startswith("(") else line.strip()
            if (
                any(w in msg for w in ERROR_WORDS)
                and "Engine core initialization failed" not in msg
                and msg not in seen
            ):
                seen.add(msg)
                keep.append(line.rstrip())
        return "\n".join(keep[:n]) or "(no error lines; see the full log)"


def strict_schema(
    model: type[BaseModel],
    max_items: dict | None = None,
    min_items: dict | None = None,
) -> dict:
    schema = copy.deepcopy(model.model_json_schema())
    max_items, min_items = max_items or {}, min_items or {}

    def walk(node: object) -> None:
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
    MAX_CONSECUTIVE_ERRORS = 20

    def __init__(
        self,
        spec: dict,
        base_url: str,
        api_key: str = "EMPTY",
        seed: int = 0,
        timeout: int = 600,
    ) -> None:
        self.spec = spec
        self.seed = seed
        self.consecutive_errors = 0
        self.model = spec["tag"] if spec["backend"] == "vllm" else spec["model"]
        self.client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0
        )

    def messages(self, system: str, user: str) -> list[dict]:
        mode = self.spec["system"]
        if mode == "native":
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return [
            {"role": "system", "content": mode},
            {"role": "user", "content": f"{system}\n\n{user}"},
        ]

    def json(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel],
        max_tokens: int,
        schema: dict | None = None,
        attempts: int = 4,
        sampling: dict | None = None,
    ) -> tuple[BaseModel | None, dict]:
        kw = dict(
            model=self.model,
            messages=self.messages(system, user),
            temperature=0,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema or response_model.model_json_schema(),
                    "strict": True,
                },
            },
        )
        if self.spec["backend"] == "vllm":
            kw["seed"] = self.seed
            extra = dict(sampling or {})
            if self.spec["chat_kwargs"]:
                extra["chat_template_kwargs"] = self.spec["chat_kwargs"]
            if extra:
                kw["extra_body"] = extra
        info = {"prompt_tokens": 0, "completion_tokens": 0, "error": None, "calls": 0}
        for attempt in range(attempts):
            info["calls"] += 1
            try:
                r = self.client.chat.completions.create(**kw)
            except (openai.BadRequestError, openai.APITimeoutError) as e:
                info["error"] = f"{type(e).__name__}: {e}"[:300]
                return None, info
            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"[:300]
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(
                        f"{self.consecutive_errors} requests in a row failed; the LLM endpoint "
                        f"looks down. Last error: {info['error']}"
                    ) from e
                time.sleep(min(60, 5 * 2**attempt))
                continue
            self.consecutive_errors = 0
            if r.usage:
                info["prompt_tokens"] += r.usage.prompt_tokens or 0
                info["completion_tokens"] += r.usage.completion_tokens or 0
            choice = r.choices[0]
            if choice.finish_reason == "length":
                info["error"] = f"cut off at {kw['max_tokens']} tokens"
                if kw["max_tokens"] * 2 > self.spec["max_model_len"]:
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


def make_client(spec: dict, base_url: str | None = None, seed: int = 0) -> LLMClient:
    if spec["backend"] == "vllm":
        return LLMClient(spec, base_url, seed=seed)
    key = os.environ.get(spec["api_key_env"], "")
    if not key:
        raise RuntimeError(f"Set {spec['api_key_env']} to use {spec['model']}.")
    return LLMClient(spec, spec["base_url"], api_key=key, seed=seed)


def served_revision(spec: dict) -> str | None:
    if spec["backend"] != "vllm":
        return None
    try:
        from huggingface_hub import model_info

        return model_info(spec["model"]).sha
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def package_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for pkg in ("vllm", "torch", "transformers", "openai", "pydantic"):
        try:
            out[pkg] = __import__(pkg).__version__
        except Exception:
            out[pkg] = None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        out["gpus"] = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    except Exception:
        out["gpus"] = []
    return out
