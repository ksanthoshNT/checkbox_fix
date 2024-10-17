from client_code.trade_finance_structure_document.src.main.checkbox_detection.utils.matrix_operations import MatrixOperations

class RectangleDetector:
    def __init__(self, matrix, right, down, box_max_width, box_max_height):
        self.matrix = matrix
        self.right = right
        self.down = down
        self.box_max_width = box_max_width
        self.box_max_height = box_max_height

    def find_rectangles(self, x, y):
        rectangles = []
        rows, cols = self.matrix.shape
        for bottom in range(x,rows):
            if bottom - x > self.box_max_width: break
            for right_edge in range(y, cols):
                if right_edge - y > self.box_max_height: break
                if self._is_valid_rectangle(x, y, bottom, right_edge):
                    rectangles.append((x, y, bottom, right_edge))
        return rectangles

    def _is_valid_rectangle(self, x, y, bottom, right_edge):
        return (self.right[x][y] >= right_edge - y + 1 and
                self.down[x][y] >= bottom - x + 1 and
                self.right[bottom][y] >= right_edge - y + 1 and
                self.down[x][right_edge] >= bottom - x + 1 and
                MatrixOperations.valid_base_checkbox(x, y, bottom, right_edge, self.matrix) and
                MatrixOperations.process_character_o_or_checkbox(x, y, bottom, right_edge, self.matrix))

    def post_process_rectangles(self, rectangles):
        res = []
        for point in rectangles:
            y1, x1, y2, x2 = point
            if not (x1 == x2 or y1 == y2):  # or is_checkbox_not_character_o(matrix[y1:y2+1, x1:x2+1]):
                res.append((x1, y1, x2, y2))

        filtered = MatrixOperations.remove_inner_rectangles(res)
        mx1, my1, mx2, my2 = float('-inf'), float('-inf'), float('-inf'), float('-inf')
        for point in filtered:
            x1, y1, x2, y2 = point
            if x2 > mx2:
                mx1, my1, mx2, my2 = x1, y1, x2, y2
                mx2 = x2
            elif mx2 == x2:
                if y1 > my2:
                    mx1, my1, mx2, my2 = x1, y1, x2, y2
                    my2 = y2

        if mx2 == float('-inf') or my2 == float('-inf'):
            return []

        return mx1, my1, mx2, my2

