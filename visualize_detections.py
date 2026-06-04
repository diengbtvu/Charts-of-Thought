#!/usr/bin/env python3
"""
visualize_detections.py — Minh hoạ kết quả stage-1 (detection) cho slide.

Đọc detections.json (do detect_charts.py sinh ra) cùng ảnh biểu đồ gốc, rồi
xuất ảnh minh hoạ gồm 2 phần cạnh nhau:

  - TRÁI : ảnh biểu đồ + các hộp bao (bounding box) text do EasyOCR phát hiện,
           kèm nhãn toạ độ [left,top,right,bottom].
  - PHẢI : bảng dữ liệu (data table) DePlot trích xuất tự động từ biểu đồ.

Dùng cho slide để cho thấy pipeline "ảnh -> layout + toạ độ + bảng dữ liệu".

Cách dùng:
  # Vẽ một vài ví dụ tiêu biểu
  python visualize_detections.py \
      --detections "1_Modified VLAT/2_Charts-of-Thought Prompt/detections.json" \
      --images "1_Modified VLAT/2_Charts-of-Thought Prompt/Images" \
      --out report/detection_examples \
      --only VLAT_a VLAT_b VLAT_g

  # Vẽ tất cả
  python visualize_detections.py --detections ... --images ... --out ...
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

OCR_LINE = re.compile(r'"(?P<text>.*)"\s*@\s*\[(?P<l>\d+),(?P<t>\d+),(?P<r>\d+),(?P<b>\d+)\]')

# Bảng màu cho box (xoay vòng) để dễ phân biệt
BOX_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
              "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
              "#9A6324", "#800000", "#000075", "#a9a9a9", "#808000"]


def parse_entry(text):
    """Tách 1 entry detections.json -> (rows_bảng, list_ocr).

    rows_bảng: list[list[str]] các ô của bảng DePlot.
    list_ocr : list[(text, (l,t,r,b))].
    """
    table_part, ocr_part = "", ""
    if "Detected text with positions" in text:
        head, _, ocr_part = text.partition("Detected text with positions")
    else:
        head = text
    # Bảng dữ liệu
    if "Data table (auto-extracted):" in head:
        table_part = head.split("Data table (auto-extracted):", 1)[1]
    else:
        table_part = head

    # DePlot dùng <0x0A> ngăn hàng, ' | ' ngăn cột
    rows = []
    for raw_row in table_part.split("<0x0A>"):
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        cells = [c.strip() for c in raw_row.split("|")]
        rows.append(cells)

    # OCR
    ocr = []
    for m in OCR_LINE.finditer(ocr_part):
        box = (int(m["l"]), int(m["t"]), int(m["r"]), int(m["b"]))
        ocr.append((m["text"], box))
    return rows, ocr


def draw_one(stem, entry_text, image_path, out_path):
    rows, ocr = parse_entry(entry_text)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    fig = plt.figure(figsize=(18, 7.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.9, 1.0], wspace=0.12)

    # ---- CỘT 1: ảnh + bounding boxes ----
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_title(f"Detected layout & coordinates — {stem}.png\n"
                     f"({len(ocr)} text boxes, EasyOCR)",
                     fontsize=12, fontweight="bold")
    for i, (txt, (l, t, r, b)) in enumerate(ocr):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        ax_img.add_patch(Rectangle((l, t), r - l, b - t,
                                   fill=False, edgecolor=color, linewidth=1.8))
        ax_img.text(l, max(t - 3, 8), str(i + 1), fontsize=8, fontweight="bold",
                    color="white", ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", fc=color, ec="none"))
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.set_xlabel("x (pixels)")
    ax_img.set_ylabel("y (pixels)")

    # ---- CỘT 2: danh sách OCR text + toạ độ ----
    ax_ocr = fig.add_subplot(gs[0, 1])
    ax_ocr.axis("off")
    ax_ocr.set_title("Detected text → [l, t, r, b]", fontsize=11,
                     fontweight="bold", loc="left")
    max_ocr = 30
    fs = 8.5 if len(ocr) <= 22 else 7.0
    lines = []
    for i, (txt, box) in enumerate(ocr[:max_ocr]):
        show = txt if len(txt) <= 18 else txt[:17] + "…"
        lines.append(f"{i+1:>2}. \"{show}\" {list(box)}")
    if len(ocr) > max_ocr:
        lines.append(f"   … (+{len(ocr) - max_ocr} more)")
    ax_ocr.text(0.0, 0.98, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=fs, transform=ax_ocr.transAxes)

    # ---- CỘT 3: bảng dữ liệu DePlot ----
    ax_tbl = fig.add_subplot(gs[0, 2])
    ax_tbl.axis("off")
    ax_tbl.set_title("Auto-extracted data table (DePlot)", fontsize=11,
                     fontweight="bold", loc="left")
    if rows:
        ncol = max(len(r) for r in rows)
        norm = [r + [""] * (ncol - len(r)) for r in rows]
        max_rows = 18
        shown = norm[:max_rows]
        tbl = ax_tbl.table(cellText=shown, cellLoc="left",
                           bbox=[0.0, 0.0, 1.0, 0.94])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.0 if len(shown) <= 14 else 6.8)
        for (rr, _cc), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if rr == 0:
                cell.set_facecolor("#1f77b4")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
        if len(norm) > max_rows:
            ax_tbl.text(0.0, -0.02, f"… (+{len(norm) - max_rows} more rows)",
                        va="top", ha="left", fontsize=8, style="italic",
                        transform=ax_tbl.transAxes)

    fig.suptitle("Stage-1 detection example  ·  chart → layout + coordinates + data table",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(ocr), len(rows)


def main():
    ap = argparse.ArgumentParser(description="Minh hoạ detection stage-1 cho slide")
    ap.add_argument("--detections", required=True, help="đường dẫn detections.json")
    ap.add_argument("--images", required=True, help="thư mục ảnh biểu đồ gốc")
    ap.add_argument("--out", default="report/detection_examples", help="thư mục lưu ảnh minh hoạ")
    ap.add_argument("--only", nargs="*", default=None,
                    help="chỉ vẽ các stem này (vd: VLAT_a VLAT_b). Bỏ trống = vẽ tất cả")
    args = ap.parse_args()

    with open(args.detections) as f:
        det = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    stems = args.only if args.only else sorted(det.keys())

    made = 0
    for stem in stems:
        if stem not in det or not det[stem]:
            print(f"[SKIP] {stem}: không có dữ liệu detection")
            continue
        image_path = os.path.join(args.images, stem + ".png")
        if not os.path.exists(image_path):
            print(f"[SKIP] {stem}: không thấy ảnh {image_path}")
            continue
        out_path = os.path.join(args.out, f"{stem}_detection.png")
        n_box, n_row = draw_one(stem, det[stem], image_path, out_path)
        print(f"[OK] {out_path}  ({n_box} boxes, {n_row} table rows)")
        made += 1

    print(f"\nĐã tạo {made} ảnh minh hoạ trong '{args.out}/'.")


if __name__ == "__main__":
    main()
