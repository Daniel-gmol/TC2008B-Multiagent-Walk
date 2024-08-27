import networkx as nx
import math

def euclidean_distance(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def create_grid_graph(width: int, height: int, impassable_cells: list[tuple]) -> nx.Graph:
    # Create a grid graph (Starts at 0-index)
    G = nx.grid_2d_graph(width + 1, height + 1)

    # Adding diagonal connections
    for (x, y) in G.nodes():
        if (x + 1, y + 1) in G:
            G.add_edge((x, y), (x + 1, y + 1))
        if (x + 1, y - 1) in G:
            G.add_edge((x, y), (x + 1, y - 1))
        if (x - 1, y + 1) in G:
            G.add_edge((x, y), (x - 1, y + 1))
        if (x - 1, y - 1) in G:
            G.add_edge((x, y), (x - 1, y - 1))

    # Remove edges associated with impassable cells
    for cell in impassable_cells:
        if cell in G:
            G.remove_node(cell)

    return G


def a_star_shortestpath(G: nx.Graph, start: tuple, goal: tuple) -> list[tuple]:
    # Use A* algorithm to find the shortest path, avoiding impassable cells
    try:
        path = nx.astar_path(G, start, goal, heuristic=euclidean_distance)
    except nx.NetworkXNoPath:
        raise ValueError("No path found between start and goal")

    return path


# Draw the graph
def print_grid_graph(G: nx.Graph) -> None:
    # Extract node coordinates
    nodes = G.nodes()
    min_x = min(x for x, y in nodes)
    max_x = max(x for x, y in nodes)
    min_y = min(y for x, y in nodes)
    max_y = max(y for x, y in nodes)

    # Create a grid with placeholders
    grid_width = max_x - min_x + 1
    grid_height = max_y - min_y + 1
    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

    # Place nodes on the grid
    for (x, y) in nodes:
        grid[y - min_y][x - min_x] = '.'

    # Print grid with equal spacing
    print(" " + " ".join(str(i) for i in range(min_x, max_x + 1)))
    for y in range(grid_height):
        print(f"{max_y - y} " + " ".join(grid[y]))
