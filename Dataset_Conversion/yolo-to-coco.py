import os, json, yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_subset(
    images_dir: Path,
    labels_dir: Path,
    class_names: dict,
    dest_dir: Path,
    start_img_id=1,
    start_ann_id=1,
):
    images, annotations = [], []
    cats = [
        {"id": cid, "name": cname, "supercategory": "none"}
        for cid, cname in class_names.items()
    ]
    img_id, ann_id = start_img_id, start_ann_id

    img_paths = sorted(
        [
            p
            for p in images_dir.glob("*.*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
    )
    for img_path in tqdm(img_paths, desc=f"Converting {dest_dir.name}"):
        fname = img_path.name
        w, h = Image.open(img_path).size
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        lbl = labels_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if not parts or len(parts) < 5:
                    continue
                cls_idx = int(parts[0])
                if cls_idx not in class_names:
                    raise ValueError(f"Invalid YOLO class {cls_idx} in label {lbl}")
                xc, yc, w_n, h_n = map(float, parts[1:])
                cid = cls_idx  # 🟢 Zero-based ID
                x = (xc - w_n / 2) * w
                y = (yc - h_n / 2) * h
                bw = w_n * w
                bh = h_n * h
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cid,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        img_id += 1

    coco = {"images": images, "annotations": annotations, "categories": cats}
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "_annotations.coco.json").write_text(json.dumps(coco, indent=2))

    # Copy images
    for img_path in img_paths:
        (dest_dir / img_path.name).write_bytes(img_path.read_bytes())

    return img_id, ann_id


def validate_coco(split_dir: Path):
    fn = split_dir / "_annotations.coco.json"
    if not fn.exists():
        raise FileNotFoundError(f"Missing COCO JSON in {split_dir}")
    data = json.loads(fn.read_text())

    img_ids = {img["id"] for img in data["images"]}
    cat_ids = {cat["id"] for cat in data["categories"]}
    modified = False

    for cat in data["categories"]:
        if "supercategory" not in cat:
            cat["supercategory"] = "none"
            modified = True

    valid_anns = []
    for ann in data["annotations"]:
        if ann.get("image_id") not in img_ids or ann.get("category_id") not in cat_ids:
            continue
        valid_anns.append(ann)
    if len(valid_anns) != len(data["annotations"]):
        print(
            f"⚠️ {split_dir.name}: Removed {len(data['annotations']) - len(valid_anns)} invalid annotations"
        )
        data["annotations"] = valid_anns
        modified = True

    if modified:
        fn.write_text(json.dumps(data, indent=2))


def convert_and_validate(yolo_root: str, dest_root: str, yaml_file: str = "data.yaml"):
    root = Path(yolo_root)
    dest = Path(dest_root)
    info = yaml.safe_load((root / yaml_file).read_text())
    class_names = {int(k): v for k, v in info["names"].items()}
    mapping = {"train": info["train"], "valid": info["val"], "test": info["test"]}

    img_id = ann_id = 1
    for split, rel in mapping.items():
        print(f"\n>>> Processing {split}")
        img_dir = root / rel
        lbl_dir = Path(str(img_dir).replace("images", "labels"))
        img_id, ann_id = convert_subset(
            img_dir, lbl_dir, class_names, dest / split, img_id, ann_id
        )
        print(f"Validating {split}")
        validate_coco(dest / split)
        print(f"{split} ✅ OK")


if __name__ == "__main__":
    convert_and_validate(
        yolo_root="/home/robomy/Desktop/THALHATH/Inspection/Dataset_Conversion/03 Dataset V8-2 YOLO (4 class) - no frm - sw",
        dest_root="/home/robomy/Desktop/THALHATH/Inspection/Dataset_Conversion/COCO_V8-2",
        yaml_file="/home/robomy/Desktop/THALHATH/Inspection/Dataset_Conversion/03 Dataset V8-2 YOLO (4 class) - no frm - sw/data.yaml",
    )
