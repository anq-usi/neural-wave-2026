import argparse, json
from pathlib import Path
from PIL import Image, ImageOps
from rembg import remove, new_session
import cv2, numpy as np


def row_id(row):
    for k in ("pair_id", "garment_id", "id"):
        if k in row:
            return row[k]
    raise KeyError(f"no id field in {list(row)}")


def resolve(rel, dataset, manifest):
    for base in (dataset, manifest.parent):
        p = base / rel
        if p.exists():
            return p
    raise FileNotFoundError(rel)


def make_packshot(img, session, max_side, margin, rotate=0.0):
    img = ImageOps.exif_transpose(img).convert("RGB")
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor=(255, 255, 255))
    img.thumbnail((max_side, max_side))
    mask = remove(img, session=session, only_mask=True,
                  post_process_mask=False).convert("L")
        # 摆正：用 mask 的最小外接矩形估计倾斜角
    m = np.array(mask.point(lambda p: 255 if p > 32 else 0))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        (_, _), (rw, rh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        if rw < rh:
            ang -= 90
        ang = ((ang + 45) % 90) - 45  # 归一到 [-45, 45]
        if abs(ang) > 2:
            img = img.rotate(ang, expand=True, fillcolor=(255, 255, 255))
            mask = mask.rotate(ang, expand=True, fillcolor=0)
    bbox = mask.point(lambda p: 255 if p > 32 else 0).getbbox()
    if bbox is None:  # 抠图失败时退回原图
        return img
    img, mask = img.crop(bbox), mask.crop(bbox)
    w, h = img.size
    pad = int(max(w, h) * margin)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (255, 255, 255))
    canvas.paste(img, (pad, pad), mask)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--dataset", default="../sartiq_neural_wave_2026")
    ap.add_argument("--model", default="birefnet-general")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--rotate", type=float, default=0.0)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()

    manifest, dataset = Path(a.manifest), Path(a.dataset)
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    rows = rows[a.offset:]
    if a.limit:
        rows = rows[: a.limit]

    session = new_session(a.model)  # 只创建一次
    for i, row in enumerate(rows, 1):
        gid = row_id(row)
        dst = Path(a.out) / a.team / gid / "front.png"
        if a.skip_existing and dst.exists():
            continue
        src = resolve(row["inputs"][0], dataset, manifest)
        out = make_packshot(Image.open(src), session, a.max_side, a.margin, a.rotate)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst)
        print(f"[{i}/{len(rows)}] {gid}", flush=True)


if __name__ == "__main__":
    main()