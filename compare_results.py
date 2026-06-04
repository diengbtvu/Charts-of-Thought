#!/usr/bin/env python3
"""
So sánh accuracy giữa các pipeline / model.
Đọc các CSV kết quả (cột correct_bool) và in bảng accuracy.

Dùng:
  # So sánh 2 thư mục OLD vs NEW
  python compare_results.py results/compare/OLD_gpt-4o results/compare/NEW_gpt-4o

  # Đặt nhãn cho dễ đọc
  python compare_results.py \
      "OLD=results/compare/OLD_gpt-4o" \
      "NEW=results/compare/NEW_gpt-4o"

  # So sánh từng câu (chỉ ra câu nào OLD sai mà NEW đúng và ngược lại)
  python compare_results.py OLD=... NEW=... --per-question
"""

import argparse
import glob
import os
import sys

import pandas as pd


def latest_csv(path):
    """Nếu path là thư mục -> lấy CSV mới nhất; nếu là file -> dùng luôn."""
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
        if not files:
            return None
        return sorted(files, key=os.path.getmtime)[-1]
    return path if os.path.exists(path) else None


def load(path):
    f = latest_csv(path)
    if not f:
        return None, None
    df = pd.read_csv(f)
    # correct_bool có thể là True/False/"N/A" -> chuẩn hóa về bool
    df["is_correct"] = df["correct_bool"].astype(str).str.upper().eq("TRUE")
    return df, f


def parse_target(arg):
    """Hỗ trợ cú pháp 'LABEL=path' hoặc chỉ 'path'."""
    if "=" in arg and not os.path.exists(arg):
        label, path = arg.split("=", 1)
        return label, path
    return os.path.basename(arg.rstrip("/")), arg


def main():
    ap = argparse.ArgumentParser(description="So sánh accuracy giữa các pipeline/model")
    ap.add_argument("targets", nargs="+", help="Các thư mục/file kết quả (hỗ trợ LABEL=path)")
    ap.add_argument("--per-question", action="store_true", help="In so sánh từng câu (cần đúng 2 targets)")
    args = ap.parse_args()

    loaded = []
    for t in args.targets:
        label, path = parse_target(t)
        df, f = load(path)
        if df is None:
            print(f"[BỎ QUA] không tìm thấy CSV cho: {path}")
            continue
        loaded.append((label, df, f))

    if not loaded:
        print("Không có dữ liệu để so sánh.")
        sys.exit(1)

    # --- Bảng accuracy tổng ---
    print(f"\n{'='*64}")
    print(f"{'PIPELINE / MODEL':<24}{'ĐÚNG':>8}{'TỔNG':>8}{'ACCURACY':>12}")
    print(f"{'-'*64}")
    for label, df, f in loaded:
        correct = int(df["is_correct"].sum())
        total = len(df)
        acc = correct / total * 100 if total else 0
        print(f"{label:<24}{correct:>8}{total:>8}{acc:>11.1f}%")
    print(f"{'='*64}")
    for label, df, f in loaded:
        print(f"  {label:<20} <- {f}")

    # --- So sánh từng câu (2 targets) ---
    if args.per_question:
        if len(loaded) != 2:
            print("\n--per-question cần đúng 2 targets.")
            return
        (la, da, _), (lb, db, _) = loaded
        n = min(len(da), len(db))
        flips_win, flips_lose = [], []
        for i in range(n):
            a_ok = bool(da.iloc[i]["is_correct"])
            b_ok = bool(db.iloc[i]["is_correct"])
            if a_ok != b_ok:
                qid = da.iloc[i].get("id", i + 1)
                try:
                    qid = int(qid)
                except (ValueError, TypeError):
                    pass
                if b_ok and not a_ok:
                    flips_win.append(qid)
                else:
                    flips_lose.append(qid)
        print(f"\n{'-'*64}")
        print(f"Câu {lb} ĐÚNG mà {la} SAI  (cải thiện): {flips_win}")
        print(f"Câu {la} ĐÚNG mà {lb} SAI  (thụt lùi) : {flips_lose}")
        net = len(flips_win) - len(flips_lose)
        print(f"Thay đổi ròng: {'+' if net >= 0 else ''}{net} câu")


if __name__ == "__main__":
    main()
