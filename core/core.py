from numpy.lib.function_base import kaiser
import numpy as np
from PIL import ImageOps, Image, ImageFont, ImageDraw
import os


def generate_textline(
        font_paths, char_sets_and_props, save_path, 
        image_id, transform, coverage_dict,
        max_length, size, max_spaces, num_geom_p, max_numbers
    ):

    # get font
    font_path = np.random.choice(font_paths)
    digital_font = ImageFont.truetype(font_path, size=size)
    covered_chars = set(coverage_dict[font_path])

    # create synthetic sequence
    seq_chars = []
    num_chars = np.random.choice(range(1, max_length))
    for char_set, prop in char_sets_and_props:
        char_set_count = int(prop * num_chars)
        available_chars = covered_chars.intersection(set(char_set))
        seq_chars.extend(np.random.choice(list(available_chars), char_set_count))
    
    num_spaces = np.random.choice(range(0, max_spaces))
    seq_spaces = num_spaces * ["_"]
    
    num_numbers = np.random.choice(range(0, max_numbers))
    seq_numbers = np.random.geometric(p=num_geom_p, size=num_numbers)

    synth_seq = seq_numbers + seq_spaces + seq_chars
    np.random.shuffle(synth_seq)
    synth_text = "".join(synth_seq)

    # create bboxes, image
    coco_bboxes, canvas = generate_image_and_bboxes(synth_text, digital_font, font_size=size, num_symbols=len(synth_text))

    # output
    out_image = transform(canvas)
    image_name = f"synth_{image_id}.png"
    out_image.save(os.path.join(save_path, image_name))

    return coco_bboxes, image_name, out_image


def generate_image_and_bboxes(text, font, font_size, num_symbols, scale_w=(1.0, 1.4), scale_h=(1.1, 1.2)):
    
    # init
    padding = font_size // 25
    char_renders = []

    # create character renders
    for c in text:
        img = Image.new('RGB', (font_size, font_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((font_size,font_size), c, (255, 255, 255), font=font, anchor='mm')
        bbox = img.getbbox()
        if bbox is None:
            num_symbols -= 1
            continue
        x0, y0, x1, y1 = bbox
        pbbox = (x0-padding, y0-padding, x1+padding, y1+padding)
        char_render = ImageOps.invert(img.crop(pbbox))
        char_renders.append(char_render)
    
    # create canvas
    rand_scale_w = np.random.uniform(scale_w)
    rand_scale_h = np.random.uniform(scale_h)
    canvas_w = int(font_size * num_symbols * rand_scale_w)
    canvas_h = int(font_size * rand_scale_h)
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    
    # more init
    coco_bboxes = []
    x = np.random.uniform(0, font_size // 4)

    # pasting
    for i in range(num_symbols):

        # special cases
        if text[i] == "_":
            x += font_size
            continue
        if text[i] == ",":
            y = canvas_h - font_size // 3
        else:
            y = font_size // 15

        # resize render randomly
        curr_render = char_renders[i]
        w, h = curr_render.size
        render_rescale = np.random.normal(1, 0.1)
        curr_render = curr_render.resize((int(w*render_rescale), int(h*render_rescale)))
        w, h = curr_render.size

        # random x and y scaling
        rand_x_scale = np.random.uniform(0, 0.25)
        rand_y_scale = np.random.uniform(0.97, 1.03)
        
        x = int(w * rand_x_scale) + int(x)
        y = int(y * rand_y_scale)

        # stopping protocols
        if x + w > canvas_w:
            break
        if y + h > canvas_h:
            continue

        # paste!
        canvas.paste(curr_render, (x, y, x + w, y + h))
        coco_bboxes.append((x, y, w, h))

        # move x position along
        x += w
    
    return coco_bboxes, canvas
