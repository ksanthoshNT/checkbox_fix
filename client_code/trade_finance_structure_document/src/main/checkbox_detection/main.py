import argparse
import json
import time
import math 
import cv2
import os
import heapq
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust this path if necessary

from google.cloud import vision
from client_code.trade_finance_structure_document.src.main.checkbox_detection.image_processing.preprocess import ImagePreprocessor
from client_code.trade_finance_structure_document.src.main.checkbox_detection.image_processing.postprocess import ImagePostprocessor
from client_code.trade_finance_structure_document.src.main.checkbox_detection.detection.checkbox_finder import CheckboxFinder
from client_code.trade_finance_structure_document.src.main.checkbox_detection import Config
from collections import defaultdict
from PIL import Image, ImageDraw


def shift_coordinates(boxes_result, roi):
    result = {}
    for box_id,val in boxes_result.items():
        coords = val.get('coordinates',None)
        if coords:
            x1,y1,x2,y2 = coords
            x1 += roi[0]
            y1 += roi[1]
            x2 += roi[0]
            y2 += roi[1]
            result[box_id] = {'ischecked':val.get("ischecked",None),'coordinates':[x1,y1,x2,y2]}
    return result

def find_closest_left_right_bbox(bbox_coordinates, ocr_coords_lst):
    x_min, y_min, x_max, y_max = bbox_coordinates    
    closest_left_bbox_coords = []
    heapq.heapify(closest_left_bbox_coords)
    closest_right_bbox_coords = []
    heapq.heapify(closest_right_bbox_coords)
    for key_bbox_coordinates in ocr_coords_lst:
        if key_bbox_coordinates == bbox_coordinates:
            continue
        x1, y1, x2, y2 = key_bbox_coordinates
        
        if (abs(y1 - y_min)/100 <= 0.2 and abs(y2 - y_max)/100 <= 0.2):
            if x2  <= x_min + 12:
                current_left_dist = x_min - x2
                
                if current_left_dist > 0:
                    heapq.heappush(closest_left_bbox_coords, (-current_left_dist, key_bbox_coordinates))
                    if len(closest_left_bbox_coords) > 2:
                        heapq.heappop(closest_left_bbox_coords)
            
            if x1 >= x_max - 12:
                current_right_dist = (x1 - x_max)
                current_right_dist = 0 if current_right_dist < 0 else current_right_dist
                heapq.heappush(closest_right_bbox_coords, (-current_right_dist, key_bbox_coordinates))
                if len(closest_right_bbox_coords) > 2:
                    heapq.heappop(closest_right_bbox_coords)
    
    left_bbox_coords = [(-dist, coords) for dist, coords in (closest_left_bbox_coords)]
    right_bbox_coords = [(-dist, coords) for dist, coords in (closest_right_bbox_coords)]
    left_bbox_coords.sort(key=lambda x:x[0])
    right_bbox_coords.sort(key=lambda x:x[0])
    return ((left_bbox_coords), (right_bbox_coords))

def same_distance(bbox_lst,side):
    if len(bbox_lst) < 2:
        return False    
    distance1, bbox_coords_1 = bbox_lst[0]
    _, bbox_coords_2 = bbox_lst[1]    
    if side == 'left':
        diff = bbox_coords_1[2] - bbox_coords_2[0]
    elif side == 'right':
        diff = bbox_coords_2[0] - bbox_coords_1[2]
    tolerance = 2
    return abs(distance1 - abs(diff)) <= tolerance

def left_right_check(left_bbox_lst, right_bbox_lst):
    if not left_bbox_lst or not right_bbox_lst or len(left_bbox_lst) < 2 or len(right_bbox_lst) < 2:
        return False

    left_distance_1, bbox_coords_1 = left_bbox_lst[0]
    _, bbox1_coords_2 = left_bbox_lst[1]
    left_distance_2 = abs(bbox_coords_1[0] - bbox1_coords_2[2])

    right_distance_1, bbox_coords_1 = right_bbox_lst[0]
    _, bbox1_coords_2 = right_bbox_lst[1]
    right_distance_2 = abs(bbox1_coords_2[0] - bbox_coords_1[2])  # Distance between the left edge of bbox2 and right edge of bbox1

    if abs(left_distance_1 - right_distance_1) <= 1:
        return True 
    elif abs(left_distance_1 - right_distance_2) <= 1:
        return True
    elif abs(right_distance_1 - left_distance_2) <= 1:
        return True 
    return False

def post_process(box_results,ocr_coords_lst):
    sorted_boxes = sorted((box['coordinates'] + [box['ischecked']] for box in box_results.values()), key=lambda x: (x[0], x[1]))

    def check_overlap(box1, box2):
        x1, y1, x2, y2, _ = box1
        x3, y3, x4, y4, _ = box2
        overlap_x = max(x1, x3) < min(x2, x4)
        overlap_y = max(y1, y3) < min(y2, y4)
        return overlap_x and overlap_y

    merged_boxes = []
    current_box = sorted_boxes[0]
    
    for box in sorted_boxes[1:]:
        if check_overlap(current_box, box):
            current_box[0] = min(current_box[0], box[0])  # x1
            current_box[1] = min(current_box[1], box[1])  # y1
            current_box[2] = max(current_box[2], box[2])  # x2
            current_box[3] = max(current_box[3], box[3])  # y2
            current_box[4] = current_box[4] or box[4]  # ischecked
        else:
            merged_boxes.append(current_box)
            current_box = box
    
    merged_boxes.append(current_box)
    merged_boxes_dict = {
        f'box_{i}': {'coordinates': box[:4], 'ischecked': box[4]} for i, box in enumerate(merged_boxes)
    }
    
    duplicate_keys = set() 
    keys_list = list(merged_boxes_dict.keys())
    n = len(keys_list)

    for i in range(n):
        key1 = keys_list[i]
        if key1 in duplicate_keys:
            continue
        for j in range(i + 1, n):
            key2 = keys_list[j]
            if key2 in duplicate_keys:
                continue
            box1, box2 = merged_boxes_dict[key1]['coordinates'], merged_boxes_dict[key2]['coordinates']
            matching_count = sum(1 for coord1, coord2 in zip(box1, box2) if coord1 == coord2)
            if matching_count >= 3:
                duplicate_keys.add(key2)
    for key in duplicate_keys:
        del merged_boxes_dict[key]
        
    keys_to_remove = []
    for key in merged_boxes_dict.keys():
        for bbox in [merged_boxes_dict[key]['coordinates']]:
            left_bbox_lst, right_bbox_lst = find_closest_left_right_bbox(bbox, ocr_coords_lst)
            if (left_bbox_lst and len(left_bbox_lst) > 1 and same_distance(left_bbox_lst,'left')):
                keys_to_remove.append(key)
                continue
            if (right_bbox_lst and len(right_bbox_lst) > 1 and same_distance(right_bbox_lst,'right')):
                keys_to_remove.append(key)
                continue 
            if (left_right_check(left_bbox_lst, right_bbox_lst)):
                keys_to_remove.append(key)
    for key in keys_to_remove:
        del merged_boxes_dict[key]
        
    return merged_boxes_dict

def extract_characters(image_path):
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    characters = defaultdict(list)
    for i in range(len(data['text'])):
        if data['text'][i].strip():
            text_length = len(data['text'][i])
            width_per_char = data['width'][i] // text_length
            for char_index, char in enumerate(data['text'][i]):
                x1 = data['left'][i] + char_index * width_per_char
                y1 = data['top'][i]
                x2 = x1 + width_per_char
                y2 = y1 + data['height'][i]
                characters[char].append([x1, y1, x2, y2])
    return characters

def print_time_till_now(curr_time,id=0):
    elapsed_time = time.perf_counter() - curr_time
    print(f"Time elapsed{id}: {elapsed_time:.4f}s")

def detect_boxes(img =None , image_path =None,roi = None):
    start_time = time.perf_counter()

    preprocessor = ImagePreprocessor(Config.EROSION_KERNEL, Config.DILATION_KERNEL, Config.BINARY_THRESHOLD)
    binary_image = preprocessor.preprocess(img = img,image_path=image_path,roi = roi)
    binary_image,_ = ImagePostprocessor.whiten_chars(binary_image)
    finder = CheckboxFinder(Config.BOX_MAX_WIDTH, Config.BOX_MAX_HEIGHT)
    print_time_till_now(start_time,1)

    s = time.perf_counter()
    boxes_result = finder.find_checkboxes(binary_image)
    print_time_till_now(s,2)

    if roi:
        boxes_result = shift_coordinates(boxes_result,roi)

    total_time = time.perf_counter() - start_time
    print(f"Total time: {total_time:.4f}s")
    
    character_coords = extract_characters(image_path)
    
    ocr_coords_lst = []
    for key in character_coords.keys():
        if key != '':
            ocr_coords_lst += character_coords[key]
        
    return post_process(boxes_result,ocr_coords_lst)

if __name__ == '__main__':
    image_path = '/home/data_science/project_files/santhosh/checkbox_fix/client_code/trade_finance_structure_document/src/main/checkbox_detection/InputImages/scblcapplication_1.png'
    response = detect_boxes(image_path=image_path)
    with open('result.json', 'w') as f:
        json.dump(response, f, indent=2)

        # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw boxes
    for box in response.values():
        coordinates = box['coordinates']
        color = 'green' if box['ischecked'] else 'red'
        draw.rectangle(coordinates, outline=color, width=2)

    # Save the result
    image.save('output.png')
    print(f"Image with boxes saved")
