#!/usr/bin/env python3
"""Master orchestrator with breakpoint resume support.

Stages:
  1. download_data   — fetch MS MARCO passages + QA pairs
  2. embed           — dense (BGE GPU) + sparse (BM25)
  3. index           — push vectors to Qdrant
  4. evaluate        — run all 6 strategies through RAGAS

Usage:
  D:\\Qwen 2.5 7B\\env\\python.exe scripts\\run_all.py          # full pipeline
  D:\\Qwen 2.5 7B\\env\\python.exe scripts\\run_all.py --stage embed  # single stage
"""

import json
import sys
import time
import subprocess
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # also search ./scripts/
from config import STATE_FILE

SCRIPTS_DIR = Path(__file__).parent
PYTHON = sys.executable

STAGES = ["download_data", "embed", "index", "evaluate"]


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"current_stage": None, "stages": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def run_script(script_name, extra_args=None):
    script_path = SCRIPTS_DIR / script_name
    cmd = [PYTHON, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"[pipeline] Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["RAG_API_KEY"] = os.getenv("RAG_API_KEY", "")
    env["RAG_BASE_URL"] = os.getenv("RAG_BASE_URL", "https://api.aipaibox.com")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[pipeline] FAILED: {script_name} (exit {result.returncode})")
        sys.exit(result.returncode)


def stage_download(state):
    run_script("download_data.py")
    state["stages"]["download_data"] = {"done": True, "ts": time.time()}
    save_state(state)


def stage_embed(state):
    from embed import run as embed_run
    stage_state = state["stages"].get("embed", {"dense_offset": 0, "sparse_done": False})
    embed_run(state=stage_state)
    state["stages"]["embed"] = stage_state
    save_state(state)


def stage_index(state):
    from index_qdrant import run as index_run
    stage_state = state["stages"].get("index", {"dense_indexed": False})
    index_run(state=stage_state)
    state["stages"]["index"] = stage_state
    save_state(state)


def stage_evaluate(state):
    from evaluate import run as eval_run
    stage_state = state["stages"].get("evaluate", {})
    eval_run(state=stage_state)
    state["stages"]["evaluate"] = stage_state
    save_state(state)


def ensure_docker():
    import subprocess as sp
    result = sp.run(["docker", "ps"], capture_output=True, text=True)
    if "qdrant" not in result.stdout:
        compose_file = SCRIPTS_DIR.parent / "docker-compose.yml"
        print("[pipeline] Starting Qdrant container...")
        sp.run(["docker", "compose", "-f", str(compose_file), "up", "-d"], check=True)
        time.sleep(3)
        print("[pipeline] Qdrant started")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agentic RAG Pipeline")
    parser.add_argument("--stage", choices=STAGES, default=None,
                        help="Run specific stage only")
    parser.add_argument("--reset", action="store_true",
                        help="Reset pipeline state")
    parser.add_argument("--skip-docker", action="store_true",
                        help="Skip Docker startup")
    args = parser.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        print("[pipeline] State reset")

    state = load_state()

    if not args.skip_docker:
        ensure_docker()

    # Determine stages to run
    if args.stage:
        target_idx = STAGES.index(args.stage)
        stages_to_run = [args.stage]
    else:
        target_idx = 0
        # Skip completed stages
        for i, s in enumerate(STAGES):
            if not state["stages"].get(s, {}).get("done"):
                target_idx = i
                break
        else:
            target_idx = len(STAGES)
        stages_to_run = STAGES[target_idx:]

    if not stages_to_run:
        print("[pipeline] All stages complete. Nothing to do.")
        return

    print(f"[pipeline] Stages to run: {stages_to_run}")
    state["current_stage"] = stages_to_run[0]
    save_state(state)

    stage_funcs = {
        "download_data": stage_download,
        "embed": stage_embed,
        "index": stage_index,
        "evaluate": stage_evaluate,
    }

    for stage_name in stages_to_run:
        print(f"\n{'#'*60}")
        print(f"# STAGE: {stage_name}")
        print(f"{'#'*60}")
        state["current_stage"] = stage_name
        save_state(state)
        stage_funcs[stage_name](state)
        state["stages"][stage_name]["done"] = True
        state["current_stage"] = None
        save_state(state)

    print("\n[pipeline] All stages complete!")
    print(f"[pipeline] Check results in: evaluation/summary.json")


if __name__ == "__main__":
    main()
