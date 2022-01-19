from numpy.lib.function_base import kaiser
import numpy as np
from PIL import ImageOps, Image, ImageFont, ImageDraw
import os

from utils.misc import to_string_list


def generate_textline(
        setname, font_paths, char_sets_and_props, save_path, 
        image_id, synth_transform, coverage_dict,
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
        char_set_count = round(prop * num_chars)
        available_chars = covered_chars.intersection(set(char_set))
        seq_chars.extend(np.random.choice(list(available_chars), char_set_count))
    
    num_spaces = np.random.choice(range(0, max_spaces))
    seq_spaces = num_spaces * ["_"]
    
    num_numbers = np.random.choice(range(0, max_numbers))
    seq_numbers = np.random.geometric(p=num_geom_p, size=num_numbers)

    synth_seq = to_string_list(seq_numbers) + to_string_list(seq_spaces) + to_string_list(seq_chars)
    np.random.shuffle(synth_seq)
    synth_text = "".join(synth_seq)
    print(f"Synth text: {synth_text}")

    # create bboxes, image
    bboxes, canvas = generate_image_and_bboxes(synth_text, digital_font, font_size=size, num_symbols=len(synth_text))

    # output
    out_image = synth_transform(canvas)
    image_name = f"{setname}_{image_id}.png"
    out_image.save(os.path.join(save_path, image_name))

    return bboxes, image_name, out_image


def generate_image_and_bboxes(
        text, font, font_size, num_symbols,
        char_dist=2, low_chars=",."
    ):
    
    # create character renders
    char_renders = []
    for c in text:
        img = Image.new('RGB', (font_size*4, font_size*4), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((font_size,font_size), c, (255, 255, 255), font=font, anchor='mm')
        bbox = img.getbbox()
        if bbox is None:
            num_symbols -= 1
            continue
        x0, y0, x1, y1 = bbox
        pbbox = (x0, y0, x1, y1)
        char_render = ImageOps.invert(img.crop(pbbox))
        char_renders.append(char_render)

    # create canvas
    total_width = sum(cr.width for cr in char_renders)
    canvas_w = int((char_dist * (num_symbols + 1)) + total_width)
    canvas_h = int(font_size) 
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    
    # more init
    bboxes = []
    x = char_dist

    # pasting
    for i in range(num_symbols):

        # get render
        curr_text = text[i]
        curr_render = char_renders[i]
        w, h = curr_render.size
        
        # account for spaces
        if curr_text == "_":
            x += w
            continue

        # create y offset
        height_diff = canvas_h - h
        if height_diff < 0: height_diff = 0
        y = height_diff // 2
        if curr_text in low_chars:
            y = canvas_h - h - char_dist

        # pasting!
        canvas.paste(curr_render, (x, y))
        bboxes.append((x, y, w, h))

        # move x position along
        x += w + char_dist
    
    return bboxes, canvas
