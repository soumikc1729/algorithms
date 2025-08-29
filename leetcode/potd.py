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
