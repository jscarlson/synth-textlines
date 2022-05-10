import numpy as np
import json
import argparse
import os
import copy


def split_coco_json(coco_json, tvt_split, seed=99):
    images = coco_json["images"]
    train_pct, val_pct, test_pct = tvt_split
    assert 0.9999 < train_pct + val_pct + test_pct <= 1
    train_N = int(train_pct * len(images))
    val_N = int(val_pct * len(images))
    np.random.seed(seed)
    np.random.shuffle(images)
    train_images = images[:train_N]
    val_images = images[train_N:train_N+val_N]
    test_images = images[train_N+val_N:]
    train_image_ids = [im["id"] for im in train_images]
    val_image_ids = [im["id"] for im in val_images]
    test_image_ids = [im["id"] for im in test_images]
    annotations = coco_json["annotations"]
    train_annotations = [a for a in annotations if a["image_id"] in train_image_ids]
    val_annotations = [a for a in annotations if a["image_id"] in val_image_ids]
    test_annotations = [a for a in annotations if a["image_id"] in test_image_ids]
    assert len(train_images) + len(val_images) + len(test_images) == len(images)
    return train_images, val_images, test_images, train_annotations, val_annotations, test_annotations
    

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", type=str, required=True,
        help="")
    parser.add_argument("--tvt_split", type=str, required=True,
        help="")
    args = parser.parse_args()

    with open(args.coco_json) as f:
        coco_json = json.load(f)
    train_coco_json = copy.deepcopy(coco_json)
    val_coco_json = copy.deepcopy(coco_json)
    test_coco_json = copy.deepcopy(coco_json)

    tvt_split = [float(x) for x in args.tvt_split.split(",")]
    train_images, val_images, test_images, \
        train_annotations, val_annotations, test_annotations = \
            split_coco_json(coco_json, tvt_split)

    train_coco_json["images"] = train_images
    train_coco_json["annotations"] = train_annotations

    val_coco_json["images"] = val_images
    val_coco_json["annotations"] = val_annotations

    test_coco_json["images"] = test_images
    test_coco_json["annotations"] = test_annotations

    input_basename, _ = os.path.splitext(os.path.basename(args.coco_json))
    src_dir = os.path.dirname(args.coco_json)
    pct_train, pct_val, pct_test = [int(100*x) for x in tvt_split]

    with open(os.path.join(src_dir, input_basename.replace("all", f"train{pct_train}") + ".json"), "w") as f:
        json.dump(train_coco_json, f, indent=2)
    with open(os.path.join(src_dir, input_basename.replace("all", f"val{pct_val}") + ".json"), "w") as f:
        json.dump(val_coco_json, f, indent=2)
    with open(os.path.join(src_dir, input_basename.replace("all", f"test{pct_test}") + ".json"), "w") as f:
        json.dump(test_coco_json, f, indent=2)
