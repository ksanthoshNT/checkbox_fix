import json
import os.path

import cv2
from pytesseract import image_to_data, Output
import numpy as np

from client_code.trade_finance_structure_document.src.main.checkbox_detection.utils.const import SAVE_OCR_INTERMEDIATE


class ImagePostprocessor:
    @staticmethod
    def whiten_chars(src):
        if SAVE_OCR_INTERMEDIATE:
            if os.path.exists('intermediate_ocr.json'):
                with open('intermediate_ocr.json',"r") as f:
                    data = json.load(f)
            else:
                data = image_to_data(src, output_type=Output.DICT)
                with open("intermediate_ocr.json","w") as f:
                    json.dump(data, f)
        else:
            data = image_to_data(src, output_type=Output.DICT)
        result = ImagePostprocessor._get_coords(data)
        previous_idx = 0
        for idx, b in enumerate(result):
            w: str = b['word']
            if ImagePostprocessor._should_whiten(w, b, previous_idx, idx):
                previous_idx = idx
                cv2.rectangle(src, (b['x1']+2, b['y1']), (b['x2']-2, b['y2']), (255, 255, 255), cv2.FILLED)
        return src, result

    @staticmethod
    def _get_coords(data):
        result = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text:
                result.append({
                    'word': text,
                    'word_coordinates': [data['left'][i], data['top'][i], data['left'][i] + data['width'][i],
                                         data['top'][i] + data['height'][i]],
                    'x1': data['left'][i],
                    'y1': data['top'][i],
                    'x2': data['left'][i] + data['width'][i],
                    'y2': data['top'][i] + data['height'][i]
                })
        return result

    @staticmethod
    def _should_whiten(word, bbox, previous_idx, current_idx):
        return (len(word) > 4 and all(c.isalnum() for c in word[:4]) and len(word) < 15 and
                '[' not in word and ']' not in word and
                (np.abs(bbox['x2'] - bbox['x1']) < 60 or
                 (ImagePostprocessor._allowed_character_combinations(word) and np.abs(bbox['x2'] - bbox['x1']) > 100)) and
                previous_idx - current_idx > 1)

    @staticmethod
    def _allowed_character_combinations(word):
        COMBINATIONS = [
            'EE', 'EF', 'EH', 'EL', 'EN', 'EP', 'ER', 'ET',
            'FE', 'FF', 'FH', 'FI', 'FL', 'FP', 'FR', 'FT',
            'HE', 'HF', 'HH', 'HI', 'HL', 'HN', 'HP', 'HR',
            'IE', 'IF', 'IH', 'II', 'IL', 'IN', 'IP', 'IR', 'IT',
            'LE', 'LF', 'LH', 'LI', 'LL', 'LN', 'LP', 'LR', 'LT',
            'NE', 'NF', 'NH', 'NI', 'NL', 'NN', 'NP', 'NR',
            'PE', 'PF', 'PH', 'PI', 'PL', 'PN', 'PP', 'PR',
            'TE', 'TF', 'TH', 'TI', 'TL', 'TN', 'TP', 'TT'
        ]
        return any(comb.lower() in word.lower() for comb in COMBINATIONS)