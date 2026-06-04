#!/usr/bin/env python3
"""
compare_results.py — So sánh kết quả A/B giữa các pipeline.

So sánh các thư mục kết quả do run_experiment.py sinh ra (mỗi thư mục chứa
1+ file CSV `*_run*_*.csv` với cột: id, response, time, correct_bool).

Mục tiêu chính: đo tác động của `--detections` (pipeline CŨ ảnh-only vs
MỚI ảnh+detection) trên cùng một bộ câu hỏi và cùng prompt.

Cách dùng:
  python compare_results.py "OLD=results/compare/OLD_gpt-4o" \
                            "NEW=results/compare/NEW_gpt-4o" \
                            --per-question

  # Có thể truyền >2 nhãn để xem bảng accuracy tổng hợp:
  python compare_results.py A=dirA B=dirB C=dirC

  # Hiển thị kèm nội dung câu hỏi / đáp án đúng:
  python compare_results.py OLD=... NEW=... --per-question \
        --questions "1_Modified VLAT/2_Charts-of-Thought Prompt/VLAT Questions.csv"

Xuất ra:
  1) Bảng accuracy mỗi nhãn (số câu đúng / tổng / %), kèm accuracy từng run.
  2) --per-question: với 2 nhãn đầu tiên (BASELINE vs CANDIDATE) liệt kê
     - CẢI THIỆN: CANDIDATE đúng mà BASELINE sai
     - THỤT LÙI : CANDIDATE sai mà BASELINE đúng
     - THAY ĐỔI RÒNG = (số cải thiện) − (số thụt lùi)
"""

import argparse
import glob
import os
import sys

import pandas as pd


# ------------------------------------------------------------
# Đọc / chuẩn hoá
# ------------------------------------------------------------

def to_status(v):
    """Chuẩn hoá giá trị correct_bool -> True / False / None(=N/A)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes", "correct"):
        return True
    if s in ("false", "0", "0.0", "no", "incorrect"):
        return False
    # "n/a", "nan", "", ...
    return None


def find_run_csvs(path):
    """Trả về danh sách CSV trong một thư mục kết quả (hoặc chính file CSV)."""
    if os.path.isfile(path) and path.endswith(".csv"):
        return [path]
    if not os.path.isdir(path):
        return []
    # Ưu tiên các file có dạng *_run*_*.csv, nếu không có thì lấy mọi .csv
    runs = sorted(glob.glob(os.path.join(path, "*_run*_*.csv")))
    if not runs:
        runs = sorted(glob.glob(os.path.join(path, "*.csv")))
    return runs


def load_label(path):
    """
    Đọc tất cả run trong một nhãn.

    Trả về dict:
      per_q[id] = list[bool|None]    (trạng thái đúng theo từng run)
      answers[id] = str             (đáp án model ở run cuối, để hiển thị)
      run_acc = list[(filename, correct, scored, total)]
    """
    csvs = find_run_csvs(path)
    if not csvs:
        raise FileNotFoundError(f"Không tìm thấy CSV kết quả trong: {path}")

    per_q = {}
    answers = {}
    run_acc = []

    for csv in csvs:
        df = pd.read_csv(csv)
        if "correct_bool" not in df.columns:
            print(f"  [WARN] bỏ qua {csv}: thiếu cột 'correct_bool'", file=sys.stderr)
            continue
        # id: dùng cột 'id' nếu có, nếu không dùng số thứ tự 1..N
        ids = df["id"] if "id" in df.columns else pd.RangeIndex(1, len(df) + 1)

        correct = scored = 0
        for qid, row in zip(ids, df.itertuples(index=False)):
            row_d = row._asdict()
            st = to_status(row_d.get("correct_bool"))
            per_q.setdefault(qid, []).append(st)
            if "response" in row_d and pd.notna(row_d["response"]):
                answers[qid] = str(row_d["response"])
            if st is not None:
                scored += 1
                if st:
                    correct += 1
        run_acc.append((os.path.basename(csv), correct, scored, len(df)))

    return {"per_q": per_q, "answers": answers, "run_acc": run_acc, "csvs": csvs}


def question_correct(statuses):
    """Tổng hợp nhiều run cho 1 câu -> True/False/None bằng biểu quyết đa số.

    None (N/A) bị loại khỏi biểu quyết; nếu toàn N/A -> None.
    Hoà -> coi là sai (False) cho thận trọng.
    """
    valid = [s for s in statuses if s is not None]
    if not valid:
        return None
    trues = sum(1 for s in valid if s)
    falses = len(valid) - trues
    if trues > falses:
        return True
    if falses > trues:
        return False
    return False  # hoà -> thận trọng coi là sai


def label_accuracy(data):
    """(correct, total, pct) tổng hợp theo câu (đa số run)."""
    per_q = data["per_q"]
    total = len(per_q)
    correct = sum(1 for st in per_q.values() if question_correct(st) is True)
    pct = (100.0 * correct / total) if total else 0.0
    return correct, total, pct


# ------------------------------------------------------------
# Hiển thị
# ------------------------------------------------------------

def load_questions(questions_path):
    """Đọc file câu hỏi VLAT để hiển thị item/câu hỏi/đáp án đúng (tuỳ chọn)."""
    if not questions_path or not os.path.isfile(questions_path):
        return {}
    df = pd.read_csv(questions_path)
    meta = {}
    # cột 'id' trong file câu hỏi khớp với 'id' trong file kết quả
    for i, row in df.iterrows():
        qid = row["id"] if "id" in df.columns else (i + 1)
        meta[qid] = {
            "item": str(row.get("item", "")).strip(),
            "question": str(row.get("question: ", row.get("question", ""))).strip(),
            "correct": str(row.get("correct", "")).strip(),
        }
    return meta


def fmt_q(qid, qmeta):
    info = qmeta.get(qid)
    if not info:
        return f"Q{qid}"
    item = info.get("item") or ""
    qtext = info.get("question") or ""
    head = f"Q{qid}" + (f" [{item}]" if item else "")
    if qtext:
        qtext = qtext if len(qtext) <= 80 else qtext[:77] + "..."
        head += f": {qtext}"
    return head


def print_accuracy_table(labels):
    print("=" * 64)
    print("BẢNG ACCURACY")
    print("=" * 64)
    name_w = max(8, max(len(name) for name, _ in labels))
    print(f"{'Pipeline'.ljust(name_w)} | {'Đúng':>5} | {'Tổng':>5} | {'Accuracy':>9}")
    print(f"{'-' * name_w}-+-{'-'*5}-+-{'-'*5}-+-{'-'*9}")
    for name, data in labels:
        correct, total, pct = label_accuracy(data)
        print(f"{name.ljust(name_w)} | {correct:>5} | {total:>5} | {pct:>8.1f}%")
    print()

    # Accuracy từng run (nếu có nhiều run)
    multi = any(len(d["run_acc"]) > 1 for _, d in labels)
    if multi:
        print("Accuracy theo từng run:")
        for name, data in labels:
            for fn, c, scored, tot in data["run_acc"]:
                pct = (100.0 * c / tot) if tot else 0.0
                print(f"  [{name}] {fn}: {c}/{tot} = {pct:.1f}%")
        print()


def print_per_question(base_name, base, cand_name, cand, qmeta):
    base_q = base["per_q"]
    cand_q = cand["per_q"]
    base_ans = base["answers"]
    cand_ans = cand["answers"]

    # Tập câu chung (theo id) để so sánh sạch
    common = sorted(set(base_q) & set(cand_q), key=lambda x: (isinstance(x, str), x))
    only_base = set(base_q) - set(cand_q)
    only_cand = set(cand_q) - set(base_q)

    improvements, regressions, agree_right, agree_wrong = [], [], [], []
    for qid in common:
        b = question_correct(base_q[qid])
        c = question_correct(cand_q[qid])
        if c is True and b is not True:
            improvements.append(qid)
        elif b is True and c is not True:
            regressions.append(qid)
        elif b is True and c is True:
            agree_right.append(qid)
        else:
            agree_wrong.append(qid)

    print("=" * 64)
    print(f"SO SÁNH THEO CÂU:  BASELINE={base_name}  vs  CANDIDATE={cand_name}")
    print("=" * 64)
    print(f"Số câu so sánh chung: {len(common)}")
    if only_base or only_cand:
        if only_base:
            print(f"  (chỉ có ở {base_name}: {sorted(only_base)})")
        if only_cand:
            print(f"  (chỉ có ở {cand_name}: {sorted(only_cand)})")
    print(f"  Cả hai ĐÚNG : {len(agree_right)}")
    print(f"  Cả hai SAI  : {len(agree_wrong)}")
    print()

    def dump(title, ids):
        print(f"{title}: {len(ids)}")
        for qid in ids:
            print("  " + fmt_q(qid, qmeta))
            ca = qmeta.get(qid, {}).get("correct", "")
            extra = f"  (đáp án đúng: {ca})" if ca else ""
            print(f"    {base_name} → {base_ans.get(qid, '')!r}")
            print(f"    {cand_name} → {cand_ans.get(qid, '')!r}{extra}")
        print()

    dump(f"✅ CẢI THIỆN ({cand_name} đúng, {base_name} sai)", improvements)
    dump(f"❌ THỤT LÙI ({cand_name} sai, {base_name} đúng)", regressions)

    net = len(improvements) - len(regressions)
    sign = "+" if net > 0 else ""
    print("-" * 64)
    print(f"THAY ĐỔI RÒNG (net): {sign}{net} câu  "
          f"({len(improvements)} cải thiện − {len(regressions)} thụt lùi)")
    bc, bt, bp = label_accuracy(base)
    cc, ct, cp = label_accuracy(cand)
    print(f"Accuracy: {base_name} {bp:.1f}%  →  {cand_name} {cp:.1f}%  "
          f"(Δ {cp - bp:+.1f} điểm %)")
    print("-" * 64)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_label_arg(arg):
    # Hỗ trợ cả "TÊN=đường_dẫn" lẫn "đường_dẫn" (tự suy ra nhãn từ tên thư mục).
    if "=" in arg and not os.path.exists(arg):
        name, _, path = arg.partition("=")
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise argparse.ArgumentTypeError(f"Nhãn không hợp lệ: {arg!r}")
        return name, path
    path = arg.strip()
    name = os.path.basename(path.rstrip("/")) or path
    return name, path


def main():
    ap = argparse.ArgumentParser(
        description="So sánh kết quả A/B giữa các thư mục output của run_experiment.py")
    ap.add_argument("labels", nargs="+", type=parse_label_arg,
                    help='Một hoặc nhiều nhãn dạng "TÊN=đường_dẫn_thư_mục_kết_quả"')
    ap.add_argument("--per-question", action="store_true",
                    help="Liệt kê chi tiết câu cải thiện / thụt lùi giữa 2 nhãn đầu")
    ap.add_argument("--questions", default=None,
                    help="(tuỳ chọn) đường dẫn 'VLAT Questions.csv' để hiển thị nội dung câu hỏi")
    args = ap.parse_args()

    # Nạp dữ liệu từng nhãn
    labels = []
    for name, path in args.labels:
        try:
            data = load_label(path)
        except FileNotFoundError as e:
            print(f"[LỖI] {e}", file=sys.stderr)
            sys.exit(1)
        n_runs = len(data["run_acc"])
        n_q = len(data["per_q"])
        print(f"Đã nạp [{name}] từ {path}: {n_runs} run, {n_q} câu.")
        labels.append((name, data))
    print()

    print_accuracy_table(labels)

    if args.per_question:
        if len(labels) < 2:
            print("[!] Cần ít nhất 2 nhãn để so sánh theo câu.", file=sys.stderr)
        else:
            qmeta = load_questions(args.questions)
            (base_name, base), (cand_name, cand) = labels[0], labels[1]
            print_per_question(base_name, base, cand_name, cand, qmeta)


if __name__ == "__main__":
    main()
