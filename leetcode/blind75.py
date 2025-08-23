# https://leetcode.com/problems/insert-interval
def insert(intervals, new_interval):
    """
    >>> insert([[1, 3], [6, 9]], [2, 5])
    [[1, 5], [6, 9]]
    >>> insert([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8])
    [[1, 2], [3, 10], [12, 16]]
    >>> insert([], [5, 7])
    [[5, 7]]
    >>> insert([[2, 5], [6, 7], [8, 9]], [0, 1])
    [[0, 1], [2, 5], [6, 7], [8, 9]]
    """

    def within(interval, point):
        return point >= interval[0] and point <= interval[1]

    def overlaps(interval1, interval2):
        return (
            within(interval1, interval2[0])
            or within(interval1, interval2[1])
            or within(interval2, interval1[0])
            or within(interval2, interval1[1])
        )

    inserted = False
    updated = []
    for interval in intervals:
        if overlaps(interval, new_interval):
            new_interval = [
                min(new_interval[0], interval[0]),
                max(new_interval[1], interval[1]),
            ]
        else:
            if not inserted and new_interval[1] < interval[0]:
                updated.append(new_interval)
                inserted = True
            updated.append(interval)

    if not inserted:
        updated.append(new_interval)

    return updated


# https://leetcode.com/problems/merge-intervals
def merge(intervals):
    """
    >>> merge([[1, 3], [2, 6], [8, 10], [15, 18]])
    [[1, 6], [8, 10], [15, 18]]
    >>> merge([[1, 4], [4, 5]])
    [[1, 5]]
    >>> merge([[1, 4], [2, 3]])
    [[1, 4]]
    """
    intervals.sort()
    curr = intervals[0]
    merged = []
    for i in range(1, len(intervals)):
        curr_start, curr_end = curr
        new_start, new_end = intervals[i]
        if curr_end >= new_start:
            curr = [curr_start, max(curr_end, new_end)]
        else:
            merged.append(curr)
            curr = [new_start, new_end]
    merged.append(curr)
    return merged


# https://leetcode.com/problems/maximum-product-subarray
def max_product(nums):
    """
    >>> max_product([2, 3, -2, 4])
    6
    >>> max_product([-2, 0, -1])
    0
    >>> max_product([-2, 3, -4])
    24
    """
    max_here = min_here = max_overall = nums[0]

    for i in range(1, len(nums)):
        tmp_min_here = min(min_here * nums[i], max_here * nums[i], nums[i])
        tmp_max_here = max(min_here * nums[i], max_here * nums[i], nums[i])
        min_here, max_here = tmp_min_here, tmp_max_here

        max_overall = max(max_overall, max_here)

    return max_overall


# https://leetcode.com/problems/valid-palindrome
def is_palindrome(string):
    """
    >>> is_palindrome("A man, a plan, a canal: Panama")
    True
    >>> is_palindrome("race a car")
    False
    >>> is_palindrome(" ")
    True
    """
    chars = list(filter(lambda char: char.isalnum(), string.lower()))
    n = len(chars)
    for i in range(n // 2):
        if chars[i] != chars[n - 1 - i]:
            return False

    return True


# https://leetcode.com/problems/course-schedule
def can_finish(num_courses, prerequisites):
    """
    >>> can_finish(2, [[1, 0]])
    True
    >>> can_finish(2, [[1, 0], [0, 1]])
    False
    """
    from data_structures.graph import DirectedGraph

    g = DirectedGraph(num_courses, prerequisites)

    visited = set()
    for node in range(num_courses):
        if node not in visited and g.dfs(node, visited, stack=set()):
            return False

    return True


# https://leetcode.com/problems/valid-parentheses
def is_valid(string):
    """
    >>> is_valid("()")
    True
    >>> is_valid("()[]{}")
    True
    >>> is_valid("(]")
    False
    >>> is_valid("([])")
    True
    >>> is_valid("([)]")
    False
    """
    from collections import deque

    bracket_map = {"(": ")", "{": "}", "[": "]"}

    stack = deque()

    for ch in string:
        if ch in bracket_map:
            stack.append(ch)
        elif stack and bracket_map[stack[-1]] == ch:
            stack.pop()
        else:
            return False

    return len(stack) == 0


# https://leetcode.com/problems/missing-number
def missing_number(nums):
    """
    >>> missing_number([3, 0, 1])
    2
    >>> missing_number([0, 1])
    2
    >>> missing_number([9, 6, 4, 2, 3, 5, 7, 0, 1])
    8
    """
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


# https://leetcode.com/problems/non-overlapping-intervals
def erase_overlap_intervals(intervals):
    """
    >>> erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]])
    1
    >>> erase_overlap_intervals([[1, 2], [1, 2], [1, 2]])
    2
    >>> erase_overlap_intervals([[1, 2], [2, 3]])
    0
    """
    intervals.sort()

    intervals_to_be_removed = 0

    curr_end = intervals[0][1]
    for i in range(1, len(intervals)):
        [start, end] = intervals[i]

        if start < curr_end:
            intervals_to_be_removed += 1
            if end < curr_end:
                curr_end = end
        else:
            curr_end = end

    return intervals_to_be_removed


# https://leetcode.com/problems/product-of-array-except-self
def product_except_self(nums):
    """
    >>> product_except_self([1, 2, 3, 4])
    [24, 12, 8, 6]
    >>> product_except_self([-1, 1, 0, -3, 3])
    [0, 0, 9, 0, 0]
    """
    prefix_products = [1] * len(nums)
    for i in range(1, len(nums)):
        prefix_products[i] = prefix_products[i - 1] * nums[i - 1]

    suffix_products = [1] * len(nums)
    for i in range(len(nums) - 2, -1, -1):
        suffix_products[i] = suffix_products[i + 1] * nums[i + 1]

    return [
        prefix_product * suffix_product
        for prefix_product, suffix_product in zip(prefix_products, suffix_products)
    ]


# https://leetcode.com/problems/pacific-atlantic-water-flow
def pacific_atlantic(heights):
    """
    >>> pacific_atlantic([[1, 2, 2, 3, 5], [3, 2, 3, 4, 4], [2, 4, 5, 3, 1], [6, 7, 1, 4, 5], [5, 1, 1, 2, 4]])
    [[4, 0], [0, 4], [3, 1], [1, 4], [3, 0], [2, 2], [1, 3]]
    >>> pacific_atlantic([[1]])
    [[0, 0]]
    """
    rows, cols = len(heights), len(heights[0])

    def height(node):
        (r, c) = node
        return heights[r][c]

    def neighbors(node):
        (r, c) = node
        if r > 0:
            yield (r - 1, c)
        if r < rows - 1:
            yield (r + 1, c)
        if c > 0:
            yield (r, c - 1)
        if c < cols - 1:
            yield (r, c + 1)

    def walk_coast(coast):
        from data_structures.graph import bfs

        visited = set()
        for node in coast:
            if node not in visited:
                can_access = lambda node, neighbor: height(node) <= height(neighbor)
                bfs(node, visited, neighbors, can_access)

        return visited

    pacific_coast = [(r, 0) for r in range(rows)] + [(0, c) for c in range(cols)]
    pacific_access = walk_coast(pacific_coast)

    atlantic_coast = [(r, cols - 1) for r in range(rows)] + [
        (rows - 1, c) for c in range(cols)
    ]
    atlantic_access = walk_coast(atlantic_coast)

    return [list(node) for node in (pacific_access & atlantic_access)]


# https://leetcode.com/problems/graph-valid-tree
def valid_tree(n, edges):
    """
    >>> valid_tree(5, [[0, 1], [0, 2], [0, 3], [1, 4]])
    True
    >>> valid_tree(5, [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]])
    False
    """
    from data_structures.graph import UndirecterdGraph

    g = UndirecterdGraph(n, edges)
    visited = set()
    has_cycle = g.dfs(0, visited, -1)
    connected = len(visited) == n
    return connected and not has_cycle


# https://leetcode.com/problems/house-robber-ii
def rob2(stashes):
    """
    >>> rob2([2, 3, 2])
    3
    >>> rob2([1, 2, 3, 1])
    4
    >>> rob2([1, 2, 3])
    3
    """
    if len(stashes) == 1:
        return stashes[0]

    dp_with_0 = [0] * len(stashes)
    dp_with_0[0] = stashes[0]
    dp_with_0[1] = max(dp_with_0[0], stashes[1])

    dp_without_0 = [0] * len(stashes)
    dp_without_0[1] = stashes[1]

    for i in range(2, len(stashes)):
        dp_with_0[i] = max(dp_with_0[i - 1], dp_with_0[i - 2] + stashes[i])
        dp_without_0[i] = max(dp_without_0[i - 1], dp_without_0[i - 2] + stashes[i])

    return max(dp_with_0[-2], dp_without_0[-1])


# https://leetcode.com/problems/number-of-islands
def num_islands(grid):
    """
    >>> num_islands([["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]])
    1
    >>> num_islands([["1", "1", "0", "0", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "1", "0", "0"], ["0", "0", "0", "1", "1"]])
    3
    >>> num_islands([["0"]])
    0
    """
    from data_structures.union_find import UnionFind

    rows, cols = len(grid), len(grid[0])

    location_ones = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                location_ones.append((r, c))

    if not location_ones:
        return 0

    uf = UnionFind(location_ones)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                if r > 0 and grid[r - 1][c] == "1":
                    uf.union((r, c), (r - 1, c))
                if c > 0 and grid[r][c - 1] == "1":
                    uf.union((r, c), (r, c - 1))

    return uf.num_components


# https://leetcode.com/problems/remove-nth-node-from-end-of-list
def remove_nth_from_end(head, n):
    """
    >>> from data_structures.linked_list import deserialize_linked_list, serialize_linked_list
    >>> serialize_linked_list(remove_nth_from_end(deserialize_linked_list([1, 2, 3, 4, 5]), 2))
    [1, 2, 3, 5]
    >>> serialize_linked_list(remove_nth_from_end(deserialize_linked_list([1]), 1))
    []
    >>> serialize_linked_list(remove_nth_from_end(deserialize_linked_list([1, 2]), 1))
    [1]
    """

    def remove_nth_from_end_util(ptr):
        if not ptr:
            return (None, 0)

        next, next_len = remove_nth_from_end_util(ptr.next)
        curr_len = next_len + 1
        if curr_len == n:
            return (next, curr_len)
        else:
            ptr.next = next
            return (ptr, curr_len)

    return remove_nth_from_end_util(head)[0]


# https://leetcode.com/problems/3sum
def three_sum(nums):
    """
    >>> three_sum([-1, 0, 1, 2, -1, -4])
    [[-1, -1, 2], [-1, 0, 1]]
    >>> three_sum([0, 1, 1])
    []
    >>> three_sum([0, 0, 0])
    [[0, 0, 0]]
    """
    nums.sort()

    def two_sum(target, left):
        ans2 = []
        right = len(nums) - 1
        while left < right:
            sum = nums[left] + nums[right]
            if sum < target:
                left += 1
            elif sum > target:
                right -= 1
            else:
                ans2.append([nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                right -= 1

        return ans2

    ans3 = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        ans2 = two_sum(-nums[i], i + 1)
        if ans2:
            ans3 += [[nums[i]] + a2 for a2 in ans2]

    return ans3


# https://leetcode.com/problems/binary-tree-level-order-traversal
def level_order(root):
    """
    >>> from data_structures.binary_tree import deserialize_binary_tree
    >>> level_order(deserialize_binary_tree([3, 9, 20, None, None, 15, 7]))
    [[3], [9, 20], [15, 7]]
    >>> level_order(deserialize_binary_tree([1]))
    [[1]]
    >>> level_order(deserialize_binary_tree([]))
    []
    """
    from collections import deque

    if not root:
        return []

    level_order_traversal = []
    current_level = deque()
    current_level.append(root)

    while current_level:
        level_order_traversal.append([node.val for node in current_level])
        next_level = deque()
        while current_level:
            node = current_level.popleft()
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        current_level = next_level

    return level_order_traversal


# https://leetcode.com/problems/coin-change
def coin_change(coins, amount):
    """
    >>> coin_change([1, 2, 5], 11)
    3
    >>> coin_change([2], 3)
    -1
    >>> coin_change([1], 0)
    0
    >>> coin_change([186, 419, 83, 408], 6249)
    20
    """
    dp = [-1] * (amount + 1)
    dp[0] = 0
    for amt in range(1, amount + 1):
        for coin in coins:
            if amt >= coin and dp[amt - coin] != -1:
                dp[amt] = (
                    dp[amt - coin] + 1
                    if dp[amt] == -1
                    else min(dp[amt], dp[amt - coin] + 1)
                )
    return dp[amount]


# https://leetcode.com/problems/house-robber
def rob(stashes):
    """
    >>> rob([1, 2, 3, 1])
    4
    >>> rob([2, 7, 9, 3, 1])
    12
    """
    if len(stashes) == 1:
        return stashes[0]

    dp = [0] * len(stashes)
    dp[0] = stashes[0]
    dp[1] = max(stashes[0], stashes[1])

    for i in range(2, len(stashes)):
        dp[i] = max(dp[i - 1], dp[i - 2] + stashes[i])

    return dp[-1]


# https://leetcode.com/problems/same-tree
def is_same_tree(first, second):
    """
    >>> from data_structures.binary_tree import deserialize_binary_tree
    >>> is_same_tree(deserialize_binary_tree([1, 2, 3]), deserialize_binary_tree([1, 2, 3]))
    True
    >>> is_same_tree(deserialize_binary_tree([1, 2]), deserialize_binary_tree([1, None, 2]))
    False
    >>> is_same_tree(deserialize_binary_tree([1, 2, 1]), deserialize_binary_tree([1, 1, 2]))
    False
    """
    if not first and not second:
        return True

    if first and second and first.val == second.val:
        left_subtree_same = is_same_tree(first.left, second.left)
        right_subtree_same = is_same_tree(first.right, second.right)
        return left_subtree_same and right_subtree_same

    return False


# https://leetcode.com/problems/two-sum
def two_sum(nums, target):
    """
    >>> two_sum([2, 7, 11, 15], 9)
    [0, 1]
    >>> two_sum([3, 2, 4], 6)
    [1, 2]
    >>> two_sum([3, 3], 6)
    [0, 1]
    """
    indices = dict()
    for i, num in enumerate(nums):
        needed = target - num
        if needed in indices:
            return [indices[needed], i]
        indices[num] = i
    raise Exception("no solution found")


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
    >>> from data_structures.binary_tree import deserialize_binary_tree, serialize_binary_tree
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
