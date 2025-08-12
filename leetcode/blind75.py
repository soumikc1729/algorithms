# https://leetcode.com/problems/set-matrix-zeroes
def set_zeroes(matrix):
    """
    >>> matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    >>> set_zeroes(matrix)
    >>> matrix
    [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    >>> matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    >>> set_zeroes(matrix)
    >>> matrix
    [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]
    """
    rows, cols = len(matrix), len(matrix[0])
    row0_has_zero = any(matrix[0][c] == 0 for c in range(cols))
    col0_has_zero = any(matrix[r][0] == 0 for r in range(rows))
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][c] == 0:
                matrix[r][0] = matrix[0][c] = 0
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0
    if row0_has_zero:
        for c in range(cols):
            matrix[0][c] = 0
    if col0_has_zero:
        for r in range(rows):
            matrix[r][0] = 0


# https://leetcode.com/problems/palindromic-substrings
def count_substrings(string):
    """
    >>> count_substrings("abc")
    3
    >>> count_substrings("aaa")
    6
    """
    total_palindromes = 0

    def expand_from_center(left, right):
        palindromes_found = 0
        while left >= 0 and right < len(string) and string[left] == string[right]:
            left -= 1
            right += 1
            palindromes_found += 1
        return palindromes_found

    for i in range(len(string)):
        total_palindromes += expand_from_center(i, i)
        total_palindromes += expand_from_center(i, i + 1)

    return total_palindromes


# https://leetcode.com/problems/valid-anagram
def is_anagram(first, second):
    """
    >>> is_anagram("anagram", "nagaram")
    True
    >>> is_anagram("rat", "car")
    False
    """
    freq = [0] * 26
    for c in first:
        freq[ord(c) - ord("a")] += 1
    for c in second:
        freq[ord(c) - ord("a")] -= 1
    return all(f == 0 for f in freq)


# https://leetcode.com/problems/invert-binary-tree
def invert_tree(root):
    """
    >>> from datastructures.binary_tree import deserialize_binary_tree, serialize_binary_tree
    >>> serialize_binary_tree(invert_tree(deserialize_binary_tree([4, 2, 7, 1, 3, 6, 9])))
    [4, 7, 2, 9, 6, 3, 1]
    >>> serialize_binary_tree(invert_tree(deserialize_binary_tree([2, 1, 3])))
    [2, 3, 1]
    >>> serialize_binary_tree(invert_tree(deserialize_binary_tree([])))
    []
    """
    if not root:
        return root

    left = invert_tree(root.left)
    right = invert_tree(root.right)

    root.left, root.right = right, left

    return root


# https://leetcode.com/problems/contains-duplicate
def contains_duplicate(nums):
    """
    >>> contains_duplicate([1, 2, 3, 1])
    True
    >>> contains_duplicate([1, 2, 3, 4])
    False
    >>> contains_duplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2])
    True
    """
    num_set = set()
    for num in nums:
        if num in num_set:
            return True
        num_set.add(num)
    return False


# https://leetcode.com/problems/jump-game
def can_jump(nums):
    """
    >>> can_jump([2, 3, 1, 1, 4])
    True
    >>> can_jump([3, 2, 1, 0, 4])
    False
    """
    can_jump_upto = 0
    for i, num in enumerate(nums):
        if i > can_jump_upto:
            return False
        can_jump_upto = max(can_jump_upto, i + num)
    return True


# https://leetcode.com/problems/maximum-subarray
def max_sub_array(nums):
    """
    >>> max_sub_array([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    6
    >>> max_sub_array([1])
    1
    >>> max_sub_array([5, 4, -1, 7, 8])
    23
    >>> max_sub_array([-1])
    -1
    """
    MIN_POSSIBLE_NUM = -(10**4) - 1
    max_sum_ending_here = max_sum_so_far = MIN_POSSIBLE_NUM
    for num in nums:
        max_sum_ending_here = max(max_sum_ending_here + num, num)
        max_sum_so_far = max(max_sum_so_far, max_sum_ending_here)
    return max_sum_so_far


if __name__ == "__main__":
    import doctest

    doctest.testmod()
