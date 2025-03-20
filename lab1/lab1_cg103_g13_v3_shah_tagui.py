from queue import PriorityQueue
from typing import List, Tuple, Dict, Set, Callable

# Define type aliases for clarity
Maze = List[List[int]]
MazeViz = List[List[str]]
Position = Tuple[int, int]

# Heuristic 1 - Manhattan Distance
def manhattan_distance(current: Position, finish: Position) -> int:
    return abs(current[0] - finish[0]) + abs(current[1] - finish[1])

# Heuristic 2 - 

def astar(maze: Maze, start: Position, finish: Position) -> Tuple[int, List[MazeViz]]:
    """
    A* search

    Parameters:
    - maze: The 2D matrix that represents the maze with 0 represents emptry space and 1 represents a wall
    - start: A tuple with the coordinates of starting position
    - finish: A tuple with the coordinates of finishing position

    Returns:
    - Number of steps from start to finish, equals -1 if the path is not found
    - Viz - everything required for step-by-step vizualization
    - Path - list of positions from start to finish
    
    """

    # Define valid position function
    is_valid_pos = lambda pos: 0 <= pos[0] < len(maze) and 0 <= pos[1] < len(maze[0]) and maze[pos[0]][pos[1]] == 0

    if not is_valid_pos(start) or not is_valid_pos(finish):
        return -1, []

    num_steps: int = -1
    path: List[Position] = []

    heuristic_func: Callable[[Position, Position], int] = manhattan_distance

    f_cost = lambda pos, heuristic: pos[1] + heuristic(pos[0], finish)

    # Define directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Initialise frontier and explored set
    frontier: PriorityQueue = PriorityQueue()
    explored: Set[Position] = set()

    # Initialise visualisation variables
    parent: Dict[Position, Position] = {}
    visualisation: List[MazeViz] = []

    # Add start position to frontier
    frontier.put((0, (start, 0)))

    # Loop until frontier is empty
    while frontier.qsize() > 0:
        _, (curr_pos, curr_cost) = frontier.get()
        
        # Check if current position is the finish
        if curr_pos == finish:
            num_steps = curr_cost
            
            # Reconstruct the path
            path = []
            current = curr_pos
            while current != start:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()  # Path is from start to finish
            
            break

        # Add current position to explored set
        explored.add(curr_pos)

        # Generate successors
        for d_x, d_y in directions:
            new_pos = (curr_pos[0] + d_x, curr_pos[1] + d_y)
            new_cost = curr_cost + 1

            # If the new position is valid and not explored, add it to the frontier
            if is_valid_pos(new_pos) and new_pos not in explored:
                frontier.put((f_cost((new_pos, new_cost), heuristic_func), (new_pos, new_cost)))
                parent[new_pos] = curr_pos  # Record the parent of the new position
            
            # If the new position is in the frontier, update the cost if it is lower
            elif is_valid_pos(new_pos) and new_pos in [pos for _, (pos, _) in frontier.queue]:
                for i, (_, (pos, cost)) in enumerate(frontier.queue):
                    if pos == new_pos and new_cost < cost:
                        frontier.queue[i] = (f_cost((new_pos, new_cost), heuristic_func), (new_pos, new_cost))
                        parent[new_pos] = curr_pos  # Update parent if cost is improved
                        break

        # Visualisation
        viz: MazeViz = [[' ' if cell == 0 else '#' for cell in row] for row in maze]
        for pos in explored:
            viz[pos[0]][pos[1]] = '.'
        for pos in [pos for _, pos in frontier.queue]:
            pos = pos[0]
            viz[pos[0]][pos[1]] = 'i'
        viz[start[0]][start[1]] = 'S'
        viz[finish[0]][finish[1]] = 'F'
        visualisation.append(viz)

    # Final visualisation of path (if path found)
    if num_steps != -1:
        viz: MazeViz = [[' ' if cell == 0 else '#' for cell in row] for row in maze]
        for pos in explored:
            viz[pos[0]][pos[1]] = '.'
        for i, pos in enumerate(path):
            viz[pos[0]][pos[1]] = str(i)
        viz[start[0]][start[1]] = 'S'
        viz[finish[0]][finish[1]] = 'F'
        visualisation.append(viz)

    return (num_steps, visualisation)

def vizualize(viz: List[MazeViz]) -> None:
    """
    Vizualization function. Shows step by step the work of the search algorithm

    Parameters:
    - viz: everything required for step-by-step vizualization
    """

    if len(viz) != 0:
        print(". - explored")
        print("i - frontier")
        print("S - start")
        print("F - finish")
        print("# - wall")
        print("Number - step number")
        print()
    
    for i, v in enumerate(viz):
        if i != len(viz) - 1:
            for row in v:
                print(' '.join(row))
            print()
        else:
            print("Final path:")
            for row in v:
                print(' '.join(row))
            print()

# Example usage:
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]
]

start_position = (0, 0)
finish_position = (4, 0)

num_steps, viz = astar(maze, start_position, finish_position)

# Print number of steps in path
if num_steps != -1:
    print(f"Path from {start_position} to {finish_position} using A* is {num_steps} steps.")
else:
    print(f"No path from {start_position} to {finish_position} exists.")

# Vizualize algorithm step-by-step even if the path was not found
vizualize(viz)