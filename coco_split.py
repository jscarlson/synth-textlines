import numpy as np
import json
import argparse
import os
import copy


def split_coco_json(coco_json, train_pct, seed=99):
    images = coco_json["images"]
    train_N = int(train_pct * len(images))
    np.random.seed(seed)
    np.random.shuffle(images)
    train_images = images[:train_N]
    test_images = images[train_N:]
    train_image_ids = [im["id"] for im in train_images]
    test_image_ids = [im["id"] for im in test_images]
    annotations = coco_json["annotations"]
    train_annotations = [a for a in annotations if a["image_id"] in train_image_ids]
    test_annotations = [a for a in annotations if a["image_id"] in test_image_ids]
    return train_images, test_images, train_annotations, test_annotations
    

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_coco_json", type=str, required=True,
        help="")
    parser.add_argument("--train_pct", type=float, required=True,
        help="")
    args = parser.parse_args()

    with open(args.input_coco_json) as f:
        coco_json = json.load(f)
    train_coco_json = copy.deepcopy(coco_json)
    test_coco_json = copy.deepcopy(coco_json)

    train_images, test_images, train_annotations, test_annotations = split_coco_json(coco_json, args.train_pct)

    train_coco_json["images"] = train_images
    train_coco_json["annotations"] = train_annotations

    test_coco_json["images"] = test_images
    test_coco_json["annotations"] = test_annotations

    with open(os.path.join(os.path.dirname(args.input_coco_json), f"train{int(100*args.train_pct)}.json"), "w") as f:
        json.dump(train_coco_json, f, indent=2)
    with open(os.path.join(os.path.dirname(args.input_coco_json), f"test{100-int(100*args.train_pct)}.json"), "w") as f:
        json.dump(test_coco_json, f, indent=2)
