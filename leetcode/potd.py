from typing import List, Iterable


# https://leetcode.com/problems/valid-sudoku
def is_valid_sudoku(board: List[List[int]]) -> bool:
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
