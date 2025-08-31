from typing import List, Iterable


# https://leetcode.com/problems/sudoku-solver
def solve_sudoku(board: List[List[int]]) -> None:
    """
    >>> board = [["5","3",".",".","7",".",".",".","."],
    ... ["6",".",".","1","9","5",".",".","."],
    ... [".","9","8",".",".",".",".","6","."],
    ... ["8",".",".",".","6",".",".",".","3"],
    ... ["4",".",".","8",".","3",".",".","1"],
    ... ["7",".",".",".","2",".",".",".","6"],
    ... [".","6",".",".",".",".","2","8","."],
    ... [".",".",".","4","1","9",".",".","5"],
    ... [".",".",".",".","8",".",".","7","9"]]
    >>> solve_sudoku(board)
    >>> board
    [['5', '3', '4', '6', '7', '8', '9', '1', '2'], ['6', '7', '2', '1', '9', '5', '3', '4', '8'], ['1', '9', '8', '3', '4', '2', '5', '6', '7'], ['8', '5', '9', '7', '6', '1', '4', '2', '3'], ['4', '2', '6', '8', '5', '3', '7', '9', '1'], ['7', '1', '3', '9', '2', '4', '8', '5', '6'], ['9', '6', '1', '5', '3', '7', '2', '8', '4'], ['2', '8', '7', '4', '1', '9', '6', '3', '5'], ['3', '4', '5', '2', '8', '6', '1', '7', '9']]
    >>> board = [[".",".","9","7","4","8",".",".","."],
    ... ["7",".",".",".",".",".",".",".","."],
    ... [".","2",".","1",".","9",".",".","."],
    ... [".",".","7",".",".",".","2","4","."],
    ... [".","6","4",".","1",".","5","9","."],
    ... [".","9","8",".",".",".","3",".","."],
    ... [".",".",".","8",".","3",".","2","."],
    ... [".",".",".",".",".",".",".",".","6"],
    ... [".",".",".","2","7","5","9",".","."]]
    >>> solve_sudoku(board)
    >>> board
    [['5', '1', '9', '7', '4', '8', '6', '3', '2'], ['7', '8', '3', '6', '5', '2', '4', '1', '9'], ['4', '2', '6', '1', '3', '9', '8', '7', '5'], ['3', '5', '7', '9', '8', '6', '2', '4', '1'], ['2', '6', '4', '3', '1', '7', '5', '9', '8'], ['1', '9', '8', '5', '2', '4', '3', '6', '7'], ['9', '7', '5', '8', '6', '3', '1', '2', '4'], ['8', '3', '2', '4', '9', '1', '7', '5', '6'], ['6', '4', '1', '2', '7', '5', '9', '8', '3']]
    >>> board = [[".",".",".",".",".","7",".",".","9"],
    ... [".","4",".",".","8","1","2",".","."],
    ... [".",".",".","9",".",".",".","1","."],
    ... [".",".","5","3",".",".",".","7","2"],
    ... ["2","9","3",".",".",".",".","5","."],
    ... [".",".",".",".",".","5","3",".","."],
    ... ["8",".",".",".","2","3",".",".","."],
    ... ["7",".",".",".","5",".",".","4","."],
    ... ["5","3","1",".","7",".",".",".","."]]
    >>> solve_sudoku(board)
    >>> board
    [['3', '1', '2', '5', '4', '7', '8', '6', '9'], ['9', '4', '7', '6', '8', '1', '2', '3', '5'], ['6', '5', '8', '9', '3', '2', '7', '1', '4'], ['1', '8', '5', '3', '6', '4', '9', '7', '2'], ['2', '9', '3', '7', '1', '8', '4', '5', '6'], ['4', '7', '6', '2', '9', '5', '3', '8', '1'], ['8', '6', '4', '1', '2', '3', '5', '9', '7'], ['7', '2', '9', '8', '5', '6', '1', '4', '3'], ['5', '3', '1', '4', '7', '9', '6', '2', '8']]
    """

    def can_place(d, r, c):
        return not (d in rows[r] or d in cols[c] or d in boxes[box_index(r, c)])

    def place_digit(d, r, c):
        rows[r].add(d)
        cols[c].add(d)
        boxes[box_index(r, c)].add(d)
        board[r][c] = str(d)

    def remove_digit(d, r, c):
        rows[r].remove(d)
        cols[c].remove(d)
        boxes[box_index(r, c)].remove(d)
        board[r][c] = "."

    def place_next_digits(r, c):
        if c == N - 1 and r == N - 1:
            return True
        else:
            if c == N - 1:
                return backtrack(r + 1, 0)
            else:
                return backtrack(r, c + 1)

    def backtrack(r, c):
        if board[r][c] == ".":
            for d in range(1, 10):
                if can_place(d, r, c):
                    place_digit(d, r, c)
                    solved = place_next_digits(r, c)
                    if solved:
                        return True
                    remove_digit(d, r, c)
        else:
            return place_next_digits(r, c)
        return False

    n = 3
    N = n * n
    box_index = lambda r, c: (r // n) * n + c // n

    rows = [set() for _ in range(N)]
    cols = [set() for _ in range(N)]
    boxes = [set() for _ in range(N)]

    for r in range(N):
        for c in range(N):
            if board[r][c] != ".":
                d = int(board[r][c])
                place_digit(d, r, c)

    if not backtrack(0, 0):
        raise Exception("no solution found")


# https://leetcode.com/problems/valid-sudoku
def is_valid_sudoku(board: List[List[str]]) -> bool:
    """
    >>> is_valid_sudoku([["5", "3", ".", ".", "7", ".", ".", ".", "."]
    ...     , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
    ...     , [".", "9", "8", ".", ".", ".", ".", "6", "."]
    ...     , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
    ...     , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
    ...     , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
    ...     , [".", "6", ".", ".", ".", ".", "2", "8", "."]
    ...     , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
    ...     , [".", ".", ".", ".", "8", ".", ".", "7", "9"]])
    True
    >>> is_valid_sudoku([["8", "3", ".", ".", "7", ".", ".", ".", "."]
    ... , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
    ... , [".", "9", "8", ".", ".", ".", ".", "6", "."]
    ... , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
    ... , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
    ... , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
    ... , [".", "6", ".", ".", ".", ".", "2", "8", "."]
    ... , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
    ... , [".", ".", ".", ".", "8", ".", ".", "7", "9"]])
    False
    """

    def has_repeat(range: Iterable[tuple[int, int]]) -> bool:
        encountered = set()
        for x, y in range:
            digit = board[x][y]
            if digit == ".":
                continue
            if digit in encountered:
                return True
            encountered.add(digit)
        return False

    rows = cols = 9
    for r in range(rows):
        row = [(r, c) for c in range(cols)]
        if has_repeat(row):
            return False

    for c in range(cols):
        col = [(r, c) for r in range(rows)]
        if has_repeat(col):
            return False

    for r in range(0, rows, 3):
        for c in range(0, cols, 3):
            square = [(x, y) for x in range(r, r + 3) for y in range(c, c + 3)]
            if has_repeat(square):
                return False

    return True


# https://leetcode.com/problems/alice-and-bob-playing-flower-game
def flower_game(n: int, m: int) -> int:
    """
    The person to start the game will be able to pick the last flower
    if there are total odd number of flowers. So, with m and n, we
    need to find how many x and y we can find so that x + y = odd,
    and this can only happen if x is odd and y is even, and vice
    versa.
    >>> flower_game(3, 2)
    3
    >>> flower_game(1, 1)
    0
    """

    def odd_even(x: int) -> tuple[int, int]:
        even = x // 2
        odd = x - even
        return (odd, even)

    (odd_n, even_n) = odd_even(n)
    (odd_m, even_m) = odd_even(m)

    return odd_n * even_m + even_n * odd_m


if __name__ == "__main__":
    import doctest

    doctest.testmod()
