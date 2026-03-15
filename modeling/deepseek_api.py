from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI
from tqdm import tqdm


@dataclass
class DeepSeekJob:
    job_id: str
    prompt: str
    status: str = "queued"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class DeepSeekConfig:
    api_key: str = "sk-3f2e7fe4ae6e4d588c619bbff9837dac"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.1
    max_output_tokens: int = 8192
    max_concurrent_jobs: int = 8
    save_path: Optional[str] = None


class DeepSeekAPIClient:
    """
    Helper to submit prompts to the DeepSeek Chat Completion API in parallel,
    optionally persist all raw responses, and provide simple aggregated stats.
    """

    def __init__(self, config: DeepSeekConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    # ------------------------------------------------------------------ #
    # Single-call helper
    # ------------------------------------------------------------------ #
    def _call_completion(self, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            stream=False,
        )
        return response.model_dump()

    # ------------------------------------------------------------------ #
    # Batch execution
    # ------------------------------------------------------------------ #
    def run_prompts(self, prompts: Sequence[str], show_progress: bool = True) -> List[DeepSeekJob]:
        jobs = [DeepSeekJob(job_id=str(idx), prompt=prompt) for idx, prompt in enumerate(prompts)]
        
        # 只在show_progress=True时显示进度条
        progress = tqdm(total=len(prompts), desc="DeepSeek requests", unit="prompt", disable=not show_progress)

        def handle_future(job: DeepSeekJob, future):
            try:
                payload = future.result()
                job.status = "completed"
                job.result = payload
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
            progress.update(1)

        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_jobs) as executor:
            futures = {
                executor.submit(self._call_completion, job.prompt): job
                for job in jobs
            }
            for future in as_completed(futures):
                handle_future(futures[future], future)

        progress.close()
        return jobs

    # ------------------------------------------------------------------ #
    # Analysis helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_text(result_payload: Dict[str, Any]) -> str:
        if not result_payload:
            return ""
        choices = result_payload.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
        return json.dumps(result_payload, ensure_ascii=False)

    @staticmethod
    def analyze_results(jobs: Sequence[DeepSeekJob]) -> Dict[str, Any]:
        completed = [job for job in jobs if job.status == "completed" and job.result]
        failed = [job for job in jobs if job.status != "completed"]
        lengths = [len(DeepSeekAPIClient.extract_text(job.result)) for job in completed]
        return {
            "total_jobs": len(jobs),
            "completed": len(completed),
            "failed_or_other": len(failed),
            "avg_text_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_text_length": min(lengths) if lengths else 0,
            "max_text_length": max(lengths) if lengths else 0,
        }

    # ------------------------------------------------------------------ #
    # Persistence + convenience wrappers
    # ------------------------------------------------------------------ #
    def run_and_save(self, prompts: Sequence[str], show_progress: bool = True) -> Tuple[List[DeepSeekJob], Dict[str, Any]]:
        jobs = self.run_prompts(prompts, show_progress=show_progress)
        stats = self.analyze_results(jobs)

        if self.config.save_path:
            payload = {
                "config": {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens,
                },
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "status": job.status,
                        "prompt": job.prompt,
                        "result": job.result,
                        "error": job.error,
                    }
                    for job in jobs
                ],
                "analysis": stats,
            }
            os.makedirs(os.path.dirname(self.config.save_path), exist_ok=True)
            with open(self.config.save_path, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        return jobs, stats

    def run_prompts_to_texts(self, prompts: Sequence[str], show_progress: bool = True) -> List[str]:
        jobs, _ = self.run_and_save(prompts, show_progress=show_progress)
        outputs: List[str] = []
        for job in jobs:
            if job.status == "completed" and job.result:
                outputs.append(self.extract_text(job.result))
            else:
                outputs.append("")
        return outputs


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def load_prompts(path: str) -> List[str]:
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(item) for item in data]
        raise ValueError("JSON prompts file must contain a list of strings.")
    prompts: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek API batch runner")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts file (JSON list or text file).")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY"), help="DeepSeek API key.")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com", help="DeepSeek API base URL.")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model identifier.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_output_tokens", type=int, default=512)
    parser.add_argument("--max_concurrent_jobs", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="deepseek_results.json")

    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Please provide a DeepSeek API key via --api_key or DEEPSEEK_API_KEY env var.")

    prompts = load_prompts(args.prompts)
    config = DeepSeekConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        max_concurrent_jobs=args.max_concurrent_jobs,
        save_path=args.save_path,
    )

    client = DeepSeekAPIClient(config)
    _, stats = client.run_and_save(prompts)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


