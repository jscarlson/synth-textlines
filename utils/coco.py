

COCO_JSON_SKELETON = {
        "images": [],
        "annotations": [],
        "info": {"year": 2022, "version": "1.0", "contributor": "synth-textlines"},
        "categories": [{"id": 0, "name": "character"}],
        "licenses": ""
    }


def create_coco_annotation_field(anno_id, image_id, width, height, x, y):
    return {
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