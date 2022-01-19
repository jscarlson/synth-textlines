import os
from tqdm import tqdm
import json
import argparse

import torchvision.transforms as T

from utils.colors import color_shift_from_targets, color_shift
from utils.fonts import load_chars, get_unicode_coverage_from_ttf
from core.core import generate_textline


if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True, default=1_000,
        help="")
    parser.add_argument("--font_folder", type=str, required=True, default="./fonts/en",
        help="")
    parser.add_argument("--char_folder", type=str, required=True, default="./chars/en",
        help="")
    parser.add_argument("--char_sets", type=str, required=True, default="latin,numerals,punc_basic",
        help="")
    parser.add_argument("--char_set_props", type=str, required=True, default="1.0,0.0,0.0",
        help="")
    parser.add_argument("--train_test_val_props", type=str, default="0.8,0.2,0.0",
        help="")
    parser.add_argument("--output_folder", type=str, default="./output",
        help="")
    parser.add_argument("--textline_max_numbers", type=int, default=2,
        help="")
    parser.add_argument("--textline_numbers_geom_p", type=float, default=0.005,
        help="")
    parser.add_argument("--textline_size", type=int, default=64,
        help="")
    parser.add_argument("--textline_max_length", type=int, default=20,
        help="")
    parser.add_argument("--textline_max_spaces", type=int, default=5,
        help="")
    parser.add_argument('--transforms',
        choices=['default', 'pr'], type=str, default="default",
        help="")
    args = parser.parse_args()

    # create transforms
    if args.transforms == "default":
        synth_transform = T.Compose([
            T.ToTensor(),
            T.RandomApply([color_shift], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
            T.RandomApply([T.GaussianBlur(11)], p=0.35),
            T.RandomInvert(p=0.2),
            T.RandomGrayscale(p=0.2),
            T.ToPILImage()
        ])
    elif args.transforms == "pr":    
        synth_transform = T.Compose([
            T.ToTensor(),
            lambda x: color_shift_from_targets(x, targets=[[234,234,212], [225, 207, 171]]),
            T.RandomApply([T.GaussianBlur(11)], p=0.35),
            T.ToPILImage()
        ])

    # get font paths
    font_paths = [os.path.join(args.font_folder, x) for x in os.listdir(args.font_folder)]

    # make coverage dict
    coverage_dict = {}
    for font_path in font_paths:
        _, covered_chars = get_unicode_coverage_from_ttf(font_path)
        coverage_dict[font_path] = covered_chars

    # get char paths
    char_paths = [os.path.join(args.char_folder, x) for x in os.listdir(args.char_folder)]
    char_sets = args.char_sets.split(",")
    chosen_char_paths = [x for x in char_paths if any(c in x for c in char_sets)]
    print(f"Chosen character sets: {chosen_char_paths}")
    char_set_props = [float(x) for x in args.char_set_props.split(",")]
    assert sum(char_set_props) == 1, "Character set proportions do not sum to 1!"
    char_set_lists = [load_chars(x) for x in chosen_char_paths]
    char_sets_and_props = zip(char_set_lists, char_set_props)
    
    # create output folder
    outdir = args.output_folder
    os.makedirs(outdir, exist_ok=True)

    # train test val split
    train_test_val_split = [float(x) for x in args.train_test_val_props.split(",")]
    train_test_val_counts = [args.count * x for x in train_test_val_split]
    
    # set dicts
    SETNAMES = ("train", "test", "val",)
    anns_dict = {SETNAMES[0]: [], SETNAMES[1]: [], SETNAMES[2]: []}
    images_dict = {SETNAMES[0]: [], SETNAMES[1]: [], SETNAMES[2]: []}
    anno_id = 0

    # save for images
    images_path = os.path.join(outdir, "images")
    os.makedirs(images_path, exist_ok=True)

    # create segs
    for setname, count in zip(SETNAMES, train_test_val_counts):
        for image_id in tqdm(range(count)):

            bboxes, image_name, synth_image = generate_textline(
                font_paths, char_sets_and_props, images_path,
                image_id, synth_transform, coverage_dict,
                max_length=args.textline_max_length,
                size=args.textline_size, 
                max_spaces=args.textline_max_spaces,
                num_geom_p=args.textline_numbers_geom_p, 
                max_numbers=args.textline_max_numbers
            )

            imgw, imgh = synth_image.shape
            image = {"width": imgw, "height": imgh, "id": image_id, "file_name": image_name}
            images_dict[setname].append(image)

            for bbox in bboxes:
                x, y, width, height = bbox
                assert (x >= 0) and (y >= 0)
                if (x + width > imgw): width = imgw - x - 1
                if (y + height > imgh): height = imgh - y - 1
                annotation = {
                    "id": anno_id, 
                    "image_id": image_id, 
                    "category_id": 0,
                    "area": int(width*height), 
                    "bbox": [int(x), int(y), int(width), int(height)],
                    "segmentation": [[int(x), int(y), int(x)+int(width), int(y), 
                        int(x)+int(width), int(y)+int(height), int(x), int(y)+int(height)]],
                    "iscrowd": 0,
                    "ignore": 0
                }
                anns_dict[setname].append(annotation)
                anno_id += 1

    # save output
    coco_json_skeletion = {
        "images": [],
        "annotations": [],
        "info": {"year": 2022, "version": "1.0", "contributor": "synth-textlines"},
        "categories": [{"id": 0, "name": "character"}],
        "licenses": ""
    }

    for setname in SETNAMES:
        coco_json = coco_json_skeletion.copy()
        coco_json["images"] = images_dict[setname]
        coco_json["annotations"] = anns_dict[setname]
        with open(os.path.join(outdir, f"synth_coco_{setname}.json"), 'w') as f:
            json.dump(coco_json, f, indent=2)
