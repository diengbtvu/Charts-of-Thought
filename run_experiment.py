#!/usr/bin/env python3
"""
Charts-of-Thought Experiment Runner
Chạy lại thực nghiệm với model tùy chọn.

Usage:
  python run_experiment.py --model gpt-4o --experiment modified_vlat --prompt cot --runs 3
  python run_experiment.py --model claude-sonnet-4-20250514 --experiment original_vlat --prompt cot
  python run_experiment.py --model gemini-2.5-pro --experiment kim_visqa --prompt cot
"""

import argparse
import base64
import json
import os
import re
import time

import pandas as pd


# ============================================================
# .env LOADER (minimal, no dependency)
# ============================================================

def load_env(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    # Normalize alternate key names to what the SDKs expect
    if "OPENAI_API_KEY" not in os.environ and os.environ.get("OPEN_AI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]
    if "GOOGLE_API_KEY" not in os.environ and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


load_env()

# ============================================================
# PROMPT TEMPLATES
# ============================================================

SYSTEM_PROMPT = "You are an assistant, skilled in reading and interpreting visually represented data."

COT_PROMPT = (
    'I am about to show you a graph and ask you a multiple-choice question about that graph. \n\n'
    'Task 1: Data Extraction and Table Creation: First, explicitly list ALL numerical values you can identify on both axes, '
    'then create a structured table using markdown syntax that includes ALL data points you identified above with appropriate column headers with units. \n \n'
    'Task 2: Sort the data: Sort the data in descending order by the numerical values. \n \n'
    'Task 3: Data Verification and Error Handling: Double-check if your table matches ALL elements in the graph by comparing each value in your table with the graph '
    'and updating your table with correct values, verify the sorting is correct, and before proceeding, confirm all corrections have been made and use ONLY the corrected data for analysis. \n \n'
    'Task 4: Question Analysis: Using ONLY the verified data in your table, compare EACH value individually with the reference value, '
    'for "less than" comparisons mark ALL values that are even slightly below the reference, for "greater than" comparisons mark ALL values that are even slightly above the reference, '
    'and show each comparison on a new line. \n\n'
    'Provide your reasoning with specific references to table values. \n\n'
    'End with: "Correct Answer: ". Just write the value, nothing else. Do not write anything after this. \n\n'
    "Let's solve this step by step."
)

GENERIC_PROMPT = (
    'I am about to show you a graph and ask you a multiple-choice question about that graph. '
    'Please analyze the graph and answer the question.\n\n'
    'End with: "Correct Answer: ". Just write the value, nothing else. Do not write anything after this.'
)


# ============================================================
# MODEL CALLERS
# ============================================================

def call_openai(model, system_prompt, text_prompt, image_b64):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=5000,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]}
        ]
    )
    return response.choices[0].message.content


def call_anthropic(model, system_prompt, text_prompt, image_b64):
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=5000,
        temperature=0.0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}}
            ]}
        ]
    )
    return response.content[0].text


def call_gemini(model, system_prompt, text_prompt, image_b64):
    from google import genai
    client = genai.Client()
    config = {"temperature": 0.0, "max_output_tokens": 5000}
    # Disable hidden "thinking" on 2.5+ models so it doesn't consume the output budget
    if "2.5" in model:
        config["thinking_config"] = {"thinking_budget": 0}
    response = client.models.generate_content(
        model=model,
        contents=[
            {"parts": [
                {"text": system_prompt + "\n\n" + text_prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_b64}}
            ]}
        ],
        config=config
    )
    return response.text


def get_caller(model_name):
    """Detect provider from model name and return caller function."""
    if "gpt" in model_name or "o1" in model_name or "o3" in model_name or "o4" in model_name:
        return call_openai
    elif "claude" in model_name:
        return call_anthropic
    elif "gemini" in model_name:
        return call_gemini
    else:
        raise ValueError(f"Cannot detect provider for model '{model_name}'. Use model names containing 'gpt', 'claude', or 'gemini'.")


# ============================================================
# EXPERIMENT CONFIGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = {
    "modified_vlat": {
        "cot": {
            "csv": os.path.join(BASE_DIR, "1_Modified VLAT/2_Charts-of-Thought Prompt/VLAT Questions.csv"),
            "images": os.path.join(BASE_DIR, "1_Modified VLAT/2_Charts-of-Thought Prompt/Images"),
        },
        "generic": {
            "csv": os.path.join(BASE_DIR, "1_Modified VLAT/1_Generic Prompt/VLAT Questions.csv"),
            "images": os.path.join(BASE_DIR, "1_Modified VLAT/1_Generic Prompt/Images"),
        },
    },
    "original_vlat": {
        "cot": {
            "csv": os.path.join(BASE_DIR, "2_Original VLAT/Charts-of-Thought Prompt/VLAT Questions.csv"),
            "images": os.path.join(BASE_DIR, "2_Original VLAT/Charts-of-Thought Prompt/Images"),
        },
    },
    "kim_visqa": {
        "cot": {
            "csv": os.path.join(BASE_DIR, "3_Kim_VisQA/Charts-of-Thought Prompt/Questions.csv"),
            "images": os.path.join(BASE_DIR, "3_Kim_VisQA/Charts-of-Thought Prompt/Images"),
        },
    },
}


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(text):
    match = re.search(r"Correct Answer:\s*([^\n]*)", text)
    if match:
        return match.group(1).strip()
    return ""


def detection_prefix(detections, vis):
    """Chèn dữ liệu detection (stage 1) vào prompt nếu có."""
    if not detections:
        return ""
    info = detections.get(vis, "")
    if not info:
        return ""
    return (
        "\n\nThe following information was auto-extracted from the chart by a detector: "
        "an underlying data table, plus detected text labels with their pixel positions "
        "[left,top,right,bottom]. Use it as a reference to read values and locate elements, "
        "but always verify against the image:\n" + info
    )


# ============================================================
# MAIN
# ============================================================

def run_vlat(model, caller, prompt_template, csv_path, images_dir, output_dir, run_id, limit=None, detections=None):
    """Run VLAT-style experiment (Modified or Original VLAT)."""
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    results = []

    for idx, row in df.iterrows():
        vis = str(row.get("vis", ""))
        question_text = str(row.get("question: ", ""))
        options = str(row.get("option:", ""))
        correct = str(row.get("correct", "")).strip()

        image_path = os.path.join(images_dir, vis + ".png")
        if not os.path.exists(image_path):
            print(f"  [SKIP] Image not found: {image_path}")
            results.append(["", 0, "N/A"])
            continue

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        text = prompt_template + detection_prefix(detections, vis) + "\n\n" + question_text + " " + options

        t0 = time.perf_counter()
        try:
            response_text = call_with_retry(caller, model, SYSTEM_PROMPT, text, image_b64)
            elapsed = time.perf_counter() - t0
            answer = extract_answer(response_text)
            is_correct = answer.strip().upper() == correct.strip().upper() if answer else "N/A"
        except Exception as e:
            elapsed = time.perf_counter() - t0
            answer = f"Error: {e}"
            is_correct = "N/A"

        results.append([answer, elapsed, is_correct])
        status = "✓" if is_correct is True else ("✗" if is_correct is False else "?")
        print(f"  [{status}] Q{idx+1}: {answer[:60]}")

        time.sleep(2)

    results_df = pd.DataFrame(results, columns=["response", "time", "correct_bool"])
    results_df.index = range(1, len(results_df) + 1)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model}_run{run_id}_{int(time.time())}.csv")
    results_df.to_csv(out_file, index_label="id")
    print(f"  Saved: {out_file}")
    return out_file


def run_kim_visqa(model, caller, prompt_template, csv_path, images_dir, output_dir, run_id, limit=None, detections=None):
    """Run Kim VisQA experiment."""
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    results = []

    for idx, row in df.iterrows():
        vis = str(row.get("vis", ""))
        question_text = str(row.get("question", ""))
        correct = str(row.get("correct", "")).strip()

        image_path = os.path.join(images_dir, vis + ".png")
        if not os.path.exists(image_path):
            print(f"  [SKIP] Image not found: {image_path}")
            results.append(["", 0, "", False])
            continue

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        text = prompt_template + detection_prefix(detections, vis) + "\n\n" + question_text

        t0 = time.perf_counter()
        try:
            response_text = call_with_retry(caller, model, SYSTEM_PROMPT, text, image_b64)
            elapsed = time.perf_counter() - t0
            answer = extract_answer(response_text)
            is_correct = compare_answers(answer, correct) if answer else False
        except Exception as e:
            elapsed = time.perf_counter() - t0
            response_text = f"Error: {e}"
            answer = ""
            is_correct = False

        results.append([response_text, elapsed, answer, is_correct])
        status = "✓" if is_correct else "✗"
        print(f"  [{status}] Q{idx+1}: {answer[:60]}")

        time.sleep(2)

        # Intermediate save every 50
        if (idx + 1) % 50 == 0:
            interim_df = pd.DataFrame(results, columns=["response", "time", "response_clean", "correct_bool"])
            interim_df.index = range(1, len(interim_df) + 1)
            interim_file = os.path.join(output_dir, f"{model}_interim_{int(time.time())}.csv")
            interim_df.to_csv(interim_file, index_label="id")

    results_df = pd.DataFrame(results, columns=["response", "time", "response_clean", "correct_bool"])
    results_df.index = range(1, len(results_df) + 1)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model}_run{run_id}_{int(time.time())}.csv")
    results_df.to_csv(out_file, index_label="id")
    print(f"  Saved: {out_file}")
    return out_file


def compare_answers(extracted, correct):
    """5% tolerance for numeric, exact match for text."""
    try:
        ext_num = float(extracted.replace("%", "").strip())
        cor_num = float(str(correct).replace("%", "").strip())
        return abs(ext_num - cor_num) <= cor_num * 0.05
    except (ValueError, TypeError):
        return extracted.strip().lower() == str(correct).strip().lower()


def call_with_retry(caller, model, system, text, image_b64, max_retries=3):
    for attempt in range(max_retries):
        try:
            return caller(model, system, text, image_b64)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 20 * (attempt + 1)
                print(f"    Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description="Charts-of-Thought Experiment Runner")
    parser.add_argument("--model", required=True, help="Model name (e.g. gpt-4o, claude-sonnet-4-20250514, gemini-2.5-pro)")
    parser.add_argument("--experiment", required=True, choices=["modified_vlat", "original_vlat", "kim_visqa"])
    parser.add_argument("--prompt", default="cot", choices=["cot", "generic"])
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions (for testing)")
    parser.add_argument("--output-dir", default=None, help="Output directory (auto-generated if not set)")
    parser.add_argument("--detections", default=None, help="Path to detections.json cache from detect_charts.py (stage 1)")
    args = parser.parse_args()

    # Validate experiment+prompt combo
    exp_config = EXPERIMENTS.get(args.experiment, {})
    if args.prompt not in exp_config:
        available = list(exp_config.keys())
        parser.error(f"Prompt '{args.prompt}' not available for '{args.experiment}'. Available: {available}")

    config = exp_config[args.prompt]
    prompt_template = COT_PROMPT if args.prompt == "cot" else GENERIC_PROMPT
    caller = get_caller(args.model)

    output_dir = args.output_dir or os.path.join(BASE_DIR, "results", args.experiment, args.prompt, args.model)

    detections = None
    if args.detections:
        with open(args.detections) as f:
            detections = json.load(f)

    print(f"{'='*60}")
    print(f"Model:      {args.model}")
    print(f"Experiment: {args.experiment}")
    print(f"Prompt:     {args.prompt}")
    print(f"Runs:       {args.runs}")
    print(f"Limit:      {args.limit or 'all questions'}")
    print(f"Detections: {args.detections or 'none (image only)'}")
    print(f"Output:     {output_dir}")
    print(f"{'='*60}\n")

    runner = run_kim_visqa if args.experiment == "kim_visqa" else run_vlat

    for run_id in range(1, args.runs + 1):
        print(f"\n--- Run {run_id}/{args.runs} ---")
        runner(args.model, caller, prompt_template, config["csv"], config["images"], output_dir, run_id, args.limit, detections)
        if run_id < args.runs:
            print("  Waiting 10s before next run...")
            time.sleep(10)

    print(f"\n{'='*60}")
    print("All runs complete!")


if __name__ == "__main__":
    main()
