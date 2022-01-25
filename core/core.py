from numpy.lib.function_base import kaiser
import numpy as np
from PIL import ImageOps, Image, ImageFont, ImageDraw
import os

from utils.misc import to_string_list


class TextlineGenerator:

    def __init__(
            self, setname, font_paths, char_sets_and_props, save_path, 
            synth_transform, coverage_dict,
            max_length, size, max_spaces, num_geom_p, max_numbers,
            language
        ):

        self.setname = setname
        self.font_paths = font_paths
        self.char_sets_and_props = char_sets_and_props
        self.save_path = save_path
        self.synth_transform = synth_transform
        self.coverage_dict = coverage_dict
        self.max_length = max_length
        self.size = size
        self.font_size = size
        self.max_spaces = max_spaces
        self.num_geom_p = num_geom_p
        self.max_numbers = max_numbers
        self.low_chars = ",.ygjqp"
        self.language = language

    def select_font(self):

        font_path = np.random.choice(self.font_paths)
        self.digital_font = ImageFont.truetype(font_path, size=self.size)
        self.covered_chars = set(self.coverage_dict[font_path])

    def generate_synthetic_textline_text(self):

        seq_chars = []
        num_chars = np.random.choice(range(1, self.max_length))
        for char_set, prop in self.char_sets_and_props:
            char_set_count = round(prop * num_chars)
            available_chars = self.covered_chars.intersection(set(char_set))
            seq_chars.extend(np.random.choice(list(available_chars), char_set_count))
        
        num_spaces = np.random.choice(range(0, self.max_spaces))
        seq_spaces = num_spaces * ["_"]
        
        num_numbers = np.random.choice(range(0, self.max_numbers))
        seq_numbers = np.random.geometric(p=self.num_geom_p, size=num_numbers)

        synth_seq = to_string_list(seq_numbers) + to_string_list(seq_spaces) + to_string_list(seq_chars)
        np.random.shuffle(synth_seq)
        synth_text = "".join(synth_seq)
        self.num_symbols = len(synth_text)

        return synth_text

    def generate_synthetic_textline_image_latin_based(self, text, char_dist, char_dist_std=3):

        W = sum(self.digital_font.getsize(c)[0] + char_dist for c in text)
        H = self.font_size
        image = Image.new("RGB", (W, H), (255,255,255))
        draw = ImageDraw.Draw(image)
        x_pos, y_pos = 0, 0
        bboxes = []
        
        for i, c in enumerate(text):

            w, h = self.digital_font.getmask(c).size
            bottom_1 = self.digital_font.getsize(c)[1]
            bottom_2 = self.digital_font.getsize(text[:i+1])[1]
            bottom = bottom_1 if bottom_1 < bottom_2 else bottom_2

            if c == "_":
                x_pos += w
                continue

            bbox = (x_pos, max(bottom - h, 0), w, h)
            bboxes.append(bbox)

            draw.text((x_pos, y_pos), c, font=self.digital_font, fill=1)
            x_jiggle = min(char_dist, abs(np.random.normal(0, char_dist_std)))
            x_pos += w + int(char_dist - x_jiggle)
            
        return bboxes, image
    
    def generate_synthetic_textline_image_character_based(self, text, char_dist, char_dist_std=3):

        # create character renders
        char_renders = []
        for c in text:
            img = Image.new('RGB', (self.font_size*4, self.font_size*4), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((self.font_size, self.font_size), c, (255, 255, 255), 
                font=self.digital_font, anchor='mm')
            bbox = img.getbbox()
            if bbox is None:
                self.num_symbols -= 1
                continue
            x0, y0, x1, y1 = bbox
            pbbox = (x0, y0, x1, y1)
            char_render = ImageOps.invert(img.crop(pbbox))
            char_renders.append(char_render)

        # create canvas
        total_width = sum(cr.width for cr in char_renders)
        canvas_w = int((char_dist * (self.num_symbols + 1)) + total_width)
        canvas_h = int(self.font_size) 
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        
        # pasting
        bboxes = []
        x = char_dist
        for i in range(self.num_symbols):

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
            if curr_text in self.low_chars:
                y = canvas_h - h - char_dist

            # pasting!
            canvas.paste(curr_render, (x, y))
            bboxes.append((x, y, w, h))

            # move x position along
            x_jiggle = min(char_dist, abs(np.random.normal(0, char_dist_std)))
            x += w + int(char_dist - x_jiggle)
        
        return bboxes, canvas

    def generate_synthetic_textline(self, char_dist, image_id):

        self.select_font()
        textline_text = self.generate_synthetic_textline_text()
        if self.language == "jp":
            bboxes, canvas = self.generate_synthetic_textline_image_character_based(textline_text, char_dist)
        elif self.language == "en":
            bboxes, canvas = self.generate_synthetic_textline_image_latin_based(textline_text, char_dist)
        out_image = self.synth_transform(canvas)
        image_name = f"{self.setname}_{image_id}.png"
        out_image.save(os.path.join(self.save_path, image_name))

        return bboxes, image_name, out_image, textline_text
