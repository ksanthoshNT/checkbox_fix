import numpy as np

from client_code.trade_finance_structure_document.src.main.checkbox_detection.config import Config


class MatrixOperations:
    @staticmethod
    def create_dp_tables(binary_image):
        rows, cols = binary_image.shape
        right = np.zeros((rows, cols), dtype=int)
        down = np.zeros((rows, cols), dtype=int)

        for i in range(rows - 1, -1, -1):
            for j in range(cols - 1, -1, -1):
                if binary_image[i][j] == 0:
                    right[i][j] = 1 + (right[i][j + 1] if j + 1 < cols else 0)
                    down[i][j] = 1 + (down[i + 1][j] if i + 1 < rows else 0)

        return right, down

    @staticmethod
    def valid_base_checkbox(y1, x1, y2, x2, matrix):
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix, np.uint8)

        length = x2 - x1
        width = y2 - y1
        perimeter = 2 * (length + width)
        if length == 0 or width == 0: return False
        ratio1 = length / width
        ratio2 = width / length

        sub_matrix = matrix[y1:y2, x1:x2]

        n_dark = np.count_nonzero(sub_matrix == 0)
        n_bright = np.count_nonzero(sub_matrix != 0)

        dark_to_bright_ratio = n_dark / n_bright if n_bright else 100

        if np.abs(y2 - y1) < Config.MIN_DIMENSION or np.abs(x2 - x1) < Config.MIN_DIMENSION:
            return False

        if (
                ratio1 <= Config.MAX_LENGTH_TO_WIDTH_RATIO
                and ratio2 < Config.MAX_WIDTH_TO_LENGTH_RATIO
                and perimeter > Config.MIN_PERIMETER
                and dark_to_bright_ratio < Config.MAX_DARK_TO_BRIGHT_RATIO
                and n_bright >= Config.MIN_NUM_BRIGHT_PIXELS
        ):
            return True
        return False

    @staticmethod
    def get_center_matrix(array, size=(2, 2)):
        rows, cols = array.shape
        center_row = rows // 2
        center_col = cols // 2
        row_start = center_row - size[0] // 2
        col_start = center_col - size[1] // 2

        return array[row_start:row_start + size[0], col_start:col_start + size[1]]

    @staticmethod
    def is_not_shape_of_x(matrix):
        center_matrix = MatrixOperations.get_center_matrix(array=matrix)
        return np.any(center_matrix == 0)

    @staticmethod
    def is_borders_all_dark(binary_image):
        border_value = 0
        if (np.all(binary_image[0, :] == border_value) and
                np.all(binary_image[-1, :] == border_value) and
                np.all(binary_image[:, 0] == border_value) and
                np.all(binary_image[:, -1] == border_value)):
            return True
        return False

    @staticmethod
    def is_bright_by_dark_ratio(threshold, matrix):
        n_dark = np.count_nonzero(matrix == 0)
        n_bright = np.count_nonzero(matrix != 0)
        dark_to_bright = n_dark / n_bright if n_bright else 100
        return dark_to_bright < threshold

    @staticmethod
    def is_symmetric(matrix):
        r, c = matrix.shape
        if r < 3 or c < 3:
            return True
        top_left = matrix[:r // 2, :c // 2]
        top_right = matrix[:r // 2, c // 2:]

        bottom_left = matrix[r // 2:, :c // 2]
        bottom_right = matrix[r // 2:, c // 2:]

        if not (
                np.all(np.count_nonzero(top_left == 0)) and
                np.all(np.count_nonzero(top_right == 0)) and
                np.all(np.count_nonzero(bottom_right == 0)) and
                np.all(np.count_nonzero(bottom_left == 0))
        ):
            return False

        status = None

        quarter_wise_status = []
        for array in [top_left, bottom_left]:
            sums = np.sum(array == 0, axis=0)
            diff_arr = np.diff(sums)

            non_zeroes_in_diff = np.count_nonzero(diff_arr)
            zeroes_in_diff = np.count_nonzero(diff_arr == 0)

            if diff_arr.size == 0 or (zeroes_in_diff and zeroes_in_diff > non_zeroes_in_diff + 5):
                s = False
            else:
                s = np.all(diff_arr <= 0)

            status = (status and s) if status is not None else s
            quarter_wise_status.append(s)

        for array in [top_right, bottom_right]:
            sums = np.sum(array == 0, axis=0)
            diff_arr = np.diff(sums)

            non_zeroes_in_diff = np.count_nonzero(diff_arr)
            zeroes_in_diff = np.count_nonzero(diff_arr == 0)

            if diff_arr.size == 0 or (zeroes_in_diff and zeroes_in_diff > non_zeroes_in_diff + 5):
                s = False
            else:
                s = np.all(diff_arr >= 0)

            status = (status and s) if status is not None else s
            quarter_wise_status.append(s)

        if quarter_wise_status:
            quarter_wise_status = [  # TL , BL , TR, BR
                quarter_wise_status[0],
                quarter_wise_status[2],
                quarter_wise_status[3],
                quarter_wise_status[0]  # TL, BL, TR, BR
            ]
        for cycle in [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]:
            if quarter_wise_status[cycle[0]] and quarter_wise_status[cycle[1]] and quarter_wise_status[cycle[2]]:
                return True

        return status

    @staticmethod
    def process_character_o_or_checkbox(x, y, bottom, right_edge, matrix):
        if x and y and bottom and right_edge:
            matrix = matrix[x:bottom + 1, y:right_edge + 1]
        # stage1 = MatrixOperations.is_borders_all_dark(matrix)
        # if not stage1: return False

        stage2 = MatrixOperations.is_not_shape_of_x(matrix)
        if stage2: return True

        stage3 = MatrixOperations.is_symmetric(matrix)
        return not stage3  # not symmetric => not character O => checkbox

    @staticmethod
    def remove_inner_rectangles(rectangles):
        filtered_rectangles = []

        rectangles = sorted(rectangles)
        n = len(rectangles)

        for idx, rect in enumerate(rectangles, 1):
            if (
                    idx < n
                    and rect[0] == rectangles[idx][0]
                    and rect[1] == rectangles[idx][1]
                    and rect[2] < rectangles[idx][2]
                    and rect[3] < rectangles[idx][3]
            ):
                continue
            filtered_rectangles.append(rect)

        return filtered_rectangles
