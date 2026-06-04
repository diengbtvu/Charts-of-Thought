#!/usr/bin/env python3
"""
make_report_charts.py — Vẽ biểu đồ báo cáo cho A/B test.

Đọc kết quả pipeline CŨ (ảnh-only) và MỚI (ảnh + detection) do
run_experiment.py sinh ra, rồi xuất các biểu đồ PNG dùng cho báo cáo:

  1) accuracy_comparison.png  — so sánh accuracy OLD vs NEW (bar).
  2) outcome_breakdown.png    — phân rã 30 câu thành 4 nhóm
                                (cả hai đúng / cải thiện / thụt lùi / cả hai sai).
  3) per_question_grid.png    — lưới đúng/sai từng câu cho OLD và NEW.
  4) net_change.png           — biểu đồ cải thiện (+) vs thụt lùi (−) và net.

Cách dùng:
  python make_report_charts.py \
      "OLD=results/compare/OLD_gpt-4o" \
      "NEW=results/compare/NEW_gpt-4o" \
      --questions "1_Modified VLAT/2_Charts-of-Thought Prompt/VLAT Questions.csv" \
      --title "GPT-4o · Modified VLAT (CoT) · 30 câu" \
      --out report
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # không cần màn hình
import matplotlib.pyplot as plt

# Tái sử dụng logic đọc / chấm điểm từ compare_results.py
from compare_results import (
    load_label,
    question_correct,
    label_accuracy,
    load_questions,
)

# Bảng màu nhất quán cho toàn báo cáo
C_OLD = "#9e9e9e"       # xám: pipeline cũ
C_NEW = "#1f77b4"       # xanh dương: pipeline mới
C_BOTH_RIGHT = "#2e7d32"   # xanh lá đậm
C_IMPROVE = "#66bb6a"      # xanh lá nhạt
C_REGRESS = "#ef5350"      # đỏ
C_BOTH_WRONG = "#bdbdbd"   # xám nhạt
C_RIGHT = "#2e7d32"
C_WRONG = "#e53935"


def categorize(base, cand):
    """Phân loại từng câu: improve / regress / both_right / both_wrong."""
    common = sorted(set(base["per_q"]) & set(cand["per_q"]),
                    key=lambda x: (isinstance(x, str), x))
    cats = {"both_right": [], "improve": [], "regress": [], "both_wrong": []}
    for qid in common:
        b = question_correct(base["per_q"][qid]) is True
        c = question_correct(cand["per_q"][qid]) is True
        if b and c:
            cats["both_right"].append(qid)
        elif (not b) and c:
            cats["improve"].append(qid)
        elif b and (not c):
            cats["regress"].append(qid)
        else:
            cats["both_wrong"].append(qid)
    return common, cats


def chart_accuracy(base_name, base, cand_name, cand, suptitle, out_path):
    bc, bt, bp = label_accuracy(base)
    cc, ct, cp = label_accuracy(cand)

    fig, ax = plt.subplots(figsize=(6, 5))
    names = [f"{base_name}\n(image-only)", f"{cand_name}\n(image + detection)"]
    vals = [bp, cp]
    colors = [C_OLD, C_NEW]
    bars = ax.bar(names, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.6)

    for bar, correct, total, pct in zip(bars, [bc, cc], [bt, ct], vals):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1.5,
                f"{pct:.1f}%\n({correct}/{total})",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    delta = cp - bp
    ax.annotate(f"Δ +{delta:.1f} pts" if delta >= 0 else f"Δ {delta:.1f} pts",
                xy=(1, cp), xytext=(0.5, max(bp, cp) + 12),
                ha="center", fontsize=12, fontweight="bold",
                color=(C_BOTH_RIGHT if delta >= 0 else C_WRONG))

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(100, max(vals) + 20))
    ax.set_title("Accuracy comparison: OLD vs NEW", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=0.98, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart_outcome_breakdown(cats, total, suptitle, out_path):
    order = [
        ("Both correct", "both_right", C_BOTH_RIGHT),
        ("Improved (NEW right, OLD wrong)", "improve", C_IMPROVE),
        ("Regressed (NEW wrong, OLD right)", "regress", C_REGRESS),
        ("Both wrong", "both_wrong", C_BOTH_WRONG),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.subplots_adjust(top=0.78, bottom=0.40, left=0.04, right=0.98)
    bar_h = 0.6
    left = 0
    for label, key, color in order:
        n = len(cats[key])
        if n == 0:
            left += n
            continue
        ax.barh(0, n, height=bar_h, left=left, color=color, edgecolor="white",
                label=f"{label}: {n}")
        ax.text(left + n / 2, 0, str(n), ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if key != "both_wrong" else "black")
        left += n

    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"Number of questions (total {total})", labelpad=6)
    ax.set_title("Per-question outcome breakdown", fontsize=13,
                 fontweight="bold", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42),
              ncol=2, frameon=False, fontsize=10,
              columnspacing=2.5, handletextpad=0.8)
    if suptitle:
        fig.text(0.5, 0.95, suptitle, ha="center", va="top",
                 fontsize=10, color="#555")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart_per_question_grid(base_name, base, cand_name, cand, common, qmeta, out_path):
    n = len(common)
    fig_w = max(8, n * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, 2.8))

    for col, qid in enumerate(common):
        b = question_correct(base["per_q"][qid]) is True
        c = question_correct(cand["per_q"][qid]) is True
        # hàng trên = OLD, hàng dưới = NEW
        ax.add_patch(plt.Rectangle((col, 1), 1, 1, facecolor=C_RIGHT if b else C_WRONG,
                                   edgecolor="white"))
        ax.add_patch(plt.Rectangle((col, 0), 1, 1, facecolor=C_RIGHT if c else C_WRONG,
                                   edgecolor="white"))
        label = str(qid)
        ax.text(col + 0.5, -0.25, label, ha="center", va="top", fontsize=7, rotation=90)

    ax.set_xlim(0, n)
    ax.set_ylim(-0.6, 2)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels([f"{base_name} (image-only)", f"{cand_name} (image+detect)"])
    ax.set_xticks([])
    ax.set_title("Per-question correctness  (green = correct, red = wrong)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Question (id)", labelpad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_net_change(cats, out_path):
    imp = len(cats["improve"])
    reg = len(cats["regress"])
    net = imp - reg

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["Improved", "Regressed", "Net change"],
                  [imp, -reg, net],
                  color=[C_IMPROVE, C_REGRESS,
                         C_BOTH_RIGHT if net >= 0 else C_WRONG],
                  edgecolor="black", linewidth=0.6, width=0.6)
    for bar, v in zip(bars, [imp, -reg, net]):
        off = 0.15 if v >= 0 else -0.15
        ax.text(bar.get_x() + bar.get_width() / 2, v + off,
                f"{v:+d}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=12, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Number of questions")
    ax.set_title("Impact of detection (net change)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_label_arg(arg):
    if "=" not in arg:
        raise argparse.ArgumentTypeError(f"Nhãn phải dạng TÊN=đường_dẫn: {arg!r}")
    name, _, path = arg.partition("=")
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser(description="Vẽ biểu đồ báo cáo cho A/B test")
    ap.add_argument("labels", nargs=2, type=parse_label_arg,
                    help='2 nhãn: "OLD=dir" "NEW=dir" (nhãn đầu = baseline)')
    ap.add_argument("--questions", default=None, help="Đường dẫn VLAT Questions.csv")
    ap.add_argument("--title", default="", help="Tiêu đề phụ cho mọi biểu đồ")
    ap.add_argument("--out", default="report", help="Thư mục lưu PNG (mặc định: report)")
    args = ap.parse_args()

    (base_name, base_path), (cand_name, cand_path) = args.labels
    base = load_label(base_path)
    cand = load_label(cand_path)
    qmeta = load_questions(args.questions)

    os.makedirs(args.out, exist_ok=True)
    common, cats = categorize(base, cand)
    total = len(common)

    chart_accuracy(base_name, base, cand_name, cand, args.title,
                   os.path.join(args.out, "accuracy_comparison.png"))
    chart_outcome_breakdown(cats, total, args.title,
                            os.path.join(args.out, "outcome_breakdown.png"))
    chart_per_question_grid(base_name, base, cand_name, cand, common, qmeta,
                            os.path.join(args.out, "per_question_grid.png"))
    chart_net_change(cats, os.path.join(args.out, "net_change.png"))

    bc, bt, bp = label_accuracy(base)
    cc, ct, cp = label_accuracy(cand)
    print(f"Đã vẽ 4 biểu đồ vào '{args.out}/':")
    for f in ["accuracy_comparison.png", "outcome_breakdown.png",
              "per_question_grid.png", "net_change.png"]:
        print("  -", os.path.join(args.out, f))
    print(f"\nTóm tắt: {base_name} {bc}/{bt} ({bp:.1f}%)  →  "
          f"{cand_name} {cc}/{ct} ({cp:.1f}%)  | "
          f"cải thiện {len(cats['improve'])}, thụt lùi {len(cats['regress'])}, "
          f"net {len(cats['improve']) - len(cats['regress']):+d}")


if __name__ == "__main__":
    main()
