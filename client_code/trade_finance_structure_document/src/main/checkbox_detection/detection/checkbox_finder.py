import time

import cv2
import numpy as np
from client_code.trade_finance_structure_document.src.main.checkbox_detection.utils.matrix_operations import MatrixOperations

class CheckboxFinder:
    def __init__(self, box_max_width, box_max_height):
        self.box_max_width = box_max_width
        self.box_max_height = box_max_height

    def find_checkboxes(self, binary_image):
        rows, cols = binary_image.shape
        right, down = MatrixOperations.create_dp_tables(binary_image) # avg 5s
        count_idx = 0
        result={}
        for y in range(rows):
            for x in range(cols):
                if binary_image[y, x] == 0 and (
                    binary_image[y - 1, x] != 0 or binary_image[y - 2, x] != 0 if y >= 2 else True) and (
                    binary_image[y, x - 1] != 0 or binary_image[y, x - 2] != 0 if x >= 2 else True):
                    checkbox = self._find_checkbox(binary_image, (y, x), right, down)
                    if checkbox:
                        result.update({'box_'+str(count_idx): checkbox})
                        count_idx += 1

        return result

    def _find_checkbox(self, matrix, source, right, down):
        from .rectangle_detector import RectangleDetector

        x, y = source
        if not self._is_valid_start_point(matrix, x, y):
            return None

        detector = RectangleDetector(matrix, right, down, self.box_max_width, self.box_max_height)
        rectangles = detector.find_rectangles(x, y)

        if not rectangles:
            return None

        point = detector.post_process_rectangles(rectangles)
        if not point:
            return None

        is_checked = self._is_checked(matrix, point)
        return {
            'coordinates': list(point),
            'ischecked': is_checked
        }

    def _is_valid_start_point(self, matrix, x, y):
        rows, cols = matrix.shape
        return 0 <= x < rows and 0 <= y < cols and matrix[x][y] == 0

    def _is_checked(self, matrix, point):
        if point:
            x1, y1, x2, y2 = point
            sub_matrix = matrix[y1:y2 + 1, x1:x2 + 1]
        else:
            sub_matrix = matrix

        rows, cols = sub_matrix.shape

        if not (sub_matrix.ndim and sub_matrix.size):
            return False
        three_splits = np.linspace(0, cols, 26, dtype=np.int8)[1:-1]

        filtered_array = sub_matrix[:, three_splits]

        filtered_array = np.int8(filtered_array != 0)

        if not (filtered_array.ndim and filtered_array.size):
            return False

        if np.all(filtered_array == 0):
            return True

        non_zero_rows = np.any(filtered_array, axis=1)
        non_zero_cols = np.any(filtered_array, axis=0)
        filtered_array = filtered_array[np.ix_(non_zero_rows, non_zero_cols)]

        ones = np.count_nonzero(filtered_array != 0)
        zeroes = np.count_nonzero(filtered_array == 0)
        otz = ones / (zeroes + ones) if zeroes + ones != 0 else float('inf')
        zto = zeroes / (ones + zeroes) if ones + zeroes != 0 else float('inf')

        if otz == float('inf') or zto == float('inf'):
            return False
        elif otz > 0.88:
            return False
        elif zto > 0.95:
            return True

        diff_arr = np.diff(filtered_array, axis=0)
        one_zeroes = np.sum(diff_arr == -1, axis=0)

        first_five_rows_util = filtered_array[:6, :]
        first_five_rows = np.all(first_five_rows_util == 0, axis=0)

        base_ones = np.count_nonzero(filtered_array != 0, axis=0)
        base_ones = np.where(base_ones == 0, 1, base_ones)
        percentage_of_zeroes = np.count_nonzero(filtered_array == 0, axis=0) / base_ones
        percentage_of_zeroes = np.where(percentage_of_zeroes == float('inf'), 0, percentage_of_zeroes)
        total_percentage = np.round(percentage_of_zeroes.mean(), decimals=2)

        intial_zeroes = np.int8(filtered_array[0] == 0)

        poz_merged = first_five_rows * intial_zeroes
        percentage_of_zeroes_merged = (poz_merged >= 0.5).astype(int)

        if total_percentage > 0.2:
            count_zeroes_group = np.sum((one_zeroes, percentage_of_zeroes_merged), axis=0)
        else:
            count_zeroes_group = one_zeroes

        max_val_count = np.max(count_zeroes_group, initial=-1)

        if max_val_count == 2:
            try:
                two_by_ones = np.count_nonzero(count_zeroes_group == 2)
                _ones = np.count_nonzero(count_zeroes_group == 1)
                two_by_ones = two_by_ones / _ones
                if two_by_ones <= 0.2 and _ones > 10:
                    max_val_count = 1
                else:
                    max_val_count = 2
            except Exception:
                max_val_count = 2

        if max_val_count == 1 or max_val_count > 2:
            return True
        else:
            return False

