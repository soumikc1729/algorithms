from collections import deque


class UndirecterdGraph:
    def __init__(self, n, edges):
        self.adj = [[] for _ in range(n)]
        for edge in edges:
            self.add_edge(*edge)

    def add_edge(self, n1, n2):
        self.adj[n1].append(n2)
        self.adj[n2].append(n1)

    def dfs(self, node, visited, parent):
        visited.add(node)

        for neighbor in self.adj[node]:
            if neighbor not in visited:
                if self.dfs(neighbor, visited, node):
                    return True
            elif neighbor != parent:
                return True

        return False


def bfs(src, visited, neighbors, can_access):
    q = deque()

    q.append(src)
    visited.add(src)

    while q:
        node = q.popleft()

        for neighbor in neighbors(node):
            if neighbor not in visited and can_access(node, neighbor):
                q.append(neighbor)
                visited.add(neighbor)
