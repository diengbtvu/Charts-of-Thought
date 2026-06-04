#!/usr/bin/env python3
"""
STAGE 1 — Chart Detection (LOCAL / GPU)
Chuyển ảnh biểu đồ -> dữ liệu có cấu trúc (bảng + OCR text), lưu cache JSON.
Chạy file NÀY trên máy GPU mạnh; output cache rồi mang về cho run_experiment.py.

Cài trên máy GPU:
  pip install torch transformers pillow
  pip install easyocr            # tùy chọn (OCR tiêu đề/legend)

Dùng:
  python detect_charts.py --experiment modified_vlat --prompt cot
  python detect_charts.py --images-dir "path/to/Images" --out detections.json
"""

import argparse
import glob
import json
import os

# Lấy lại cấu hình đường dẫn ảnh từ runner (tránh lặp config)
from run_experiment import EXPERIMENTS

DEPLOT_PROMPT = "Generate underlying data table of the figure below:"


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_deplot(device):
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    proc = Pix2StructProcessor.from_pretrained("google/deplot")
    model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(device)
    return proc, model


def load_ocr():
    """EasyOCR là tùy chọn; trả None nếu chưa cài."""
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=True)
    except Exception as e:
        print(f"[OCR] bỏ qua EasyOCR ({e})")
        return None


def detect_image(proc, model, device, ocr, image_path):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    # 1) Chart -> bảng dữ liệu
    inputs = proc(images=img, text=DEPLOT_PROMPT, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=512)
    table = proc.decode(out[0], skip_special_tokens=True)

    # 2) OCR text + tọa độ (tùy chọn) — giống grounding/bbox trong L9
    ocr_text = ""
    if ocr is not None:
        lines = []
        for bbox, txt, conf in ocr.readtext(image_path):  # detail=1 (mặc định)
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            box = [min(xs), min(ys), max(xs), max(ys)]  # left,top,right,bottom
            lines.append(f'"{txt}" @ [{box[0]},{box[1]},{box[2]},{box[3]}]')
        ocr_text = "\n".join(lines)

    parts = ["Data table (auto-extracted):\n" + table]
    if ocr_text:
        parts.append("Detected text with positions [left,top,right,bottom] in pixels:\n" + ocr_text)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Stage 1 - Chart detection (local GPU)")
    ap.add_argument("--experiment", choices=list(EXPERIMENTS.keys()))
    ap.add_argument("--prompt", default="cot")
    ap.add_argument("--images-dir", help="Ghi đè thư mục ảnh (thay cho --experiment)")
    ap.add_argument("--out", help="File cache JSON output")
    ap.add_argument("--no-ocr", action="store_true", help="Tắt OCR, chỉ chạy DePlot")
    args = ap.parse_args()

    # Xác định thư mục ảnh + file output
    if args.images_dir:
        images_dir = args.images_dir
        out_path = args.out or "detections.json"
    else:
        if not args.experiment:
            ap.error("Cần --experiment hoặc --images-dir")
        cfg = EXPERIMENTS[args.experiment][args.prompt]
        images_dir = cfg["images"]
        out_path = args.out or os.path.join(os.path.dirname(cfg["images"]), "detections.json")

    images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not images:
        ap.error(f"Không tìm thấy ảnh .png trong {images_dir}")

    device = get_device()
    print(f"Device: {device} | {len(images)} ảnh | output: {out_path}")

    proc, model = load_deplot(device)
    ocr = None if args.no_ocr else load_ocr()

    detections = {}
    for i, path in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            detections[stem] = detect_image(proc, model, device, ocr, path)
            print(f"[{i}/{len(images)}] {stem} ✓")
        except Exception as e:
            detections[stem] = ""
            print(f"[{i}/{len(images)}] {stem} ✗ {e}")

    with open(out_path, "w") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)
    print(f"Saved cache: {out_path}")


if __name__ == "__main__":
    main()
