import os
import json
import copy
from PIL import Image
import argparse
import subprocess
from tqdm import tqdm
import shutil
from glob import glob

from utils.coco import *
from utils.transforms import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True,
        help="")
    parser.add_argument("--font_dir", type=str, required=True,
        help="")
    parser.add_argument("--trdg_repo", type=str, required=True,
        help="")
    parser.add_argument("--font_sizes", type=str, required=True,
        help="")
    parser.add_argument("--count", type=str, required=True,
        help="")
    parser.add_argument("--threads", type=str, default="4",
        help="")
    parser.add_argument("--max_words", type=str, default="8",
        help="")
    args = parser.parse_args()

    os.makedirs(args.save_dir)
    font_sizes = args.font_sizes.split(",")
    trdg_dirs = []

    for size in font_sizes:
        trdg_dir = os.path.join(args.save_dir, f"trdg_s{size}")
        os.makedirs(trdg_dir)
        trdg_dirs.append(trdg_dir)
        subprocess.run([
            "python", os.path.join(args.trdg_repo, "run.py"), 
            "--output_dir", trdg_dir,  
            "-l", "en", 
            "-c", args.count, 
            "--length", args.max_words, 
            "--random",
            "-f", size, 
            "--output_bboxes", "2", 
            "-na", "2", 
            "-fd", args.font_dir, 
            "-t", args.threads,
        ])

    image_id = 0
    anno_id = 0
    coco = copy.deepcopy(COCO_JSON_SKELETON)
    record_keeping_tuples = []

    for trdg_dir in tqdm(trdg_dirs):
        bbox_files = sorted([x for x in os.listdir(trdg_dir) if x.endswith(".box")])
        img_files = sorted([x for x in os.listdir(trdg_dir) if x.endswith(".jpg")])

        for imgf, bboxf in zip(img_files, bbox_files):
            img = Image.open(os.path.join(trdg_dir, imgf))
            simgf = f"{os.path.basename(trdg_dir)}_{imgf}"
            W, H = img.size
            coco["images"].append(create_coco_image_entry(simgf, H, W, image_id))
            with open(os.path.join(trdg_dir, bboxf)) as f:
                bboxes = [[int(c) for c in x.split()[1:-1]] for x in f.read().split("\n") if len(x.split()) == 6]
                for x0, y0, x1, y1 in bboxes:
                    coco["annotations"].append(create_coco_anno_entry(x0, H-y1, x1-x0, y1-y0, anno_id, image_id))
                    anno_id += 1
            image_id += 1
            record_keeping_tuples.append((trdg_dir, imgf, ))

    with open(os.path.join(args.save_dir, "coco.json"), "w") as f:
        json.dump(coco, f, indent=2)

    images_dir = os.path.join(args.save_dir, "images")
    os.makedirs(images_dir)

    trdg_transform = TRANSFORM_DICT["trdgcolor"]

    for trdg_dir in tqdm(trdg_dirs):
        for path in glob(f"{trdg_dir}/*.jpg"):
            img_tr = trdg_transform(Image.open(path))
            img_tr.save(os.path.join(images_dir, f"{os.path.basename(trdg_dir)}_{os.path.basename(path)}"))
