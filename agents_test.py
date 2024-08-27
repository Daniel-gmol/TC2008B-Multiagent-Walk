# Model design
import agentpy as ap
import numpy as np

# Random
import random

# Visualization
import matplotlib.pyplot as plt
import IPython

import json
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


def print_grid_graph(G: nx.Graph, current_pos:tuple = None) -> None:
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

    # Place current position on the grid
    if current_pos is not None:
        grid[current_pos[1] - min_y][current_pos[0] - min_x] = 'X'
    

    # Print grid with equal spacing
    print(" " + " ".join(str(i) for i in range(min_x, max_x + 1)))
    for y in range(grid_height):
        print(f"{y} " + " ".join(grid[y]))



class Pedestrian(ap.Agent):
    def setup(self):
        # Agent attributes
        self.priority = 0             # migth be used for agent neogotiation
        self.velocity = [1, 0, 0]     # Initial velocity for movement
        self.look_ahead_distance = 5  # Distance to start checking the semaphore
        self.id = None                # Agent ID
        self.vectors = []

        # Agent perceptions
        self.semaphore_x = None     # aquí empieza el paso peatonal
        self.sempahore_state = None  # semaphore state (0 - red, 1 - green)

    def setup_pos(self, space, grid):
        self.space = space               # space where the agent is
        self.full_pos = space.positions[self]  # agent position in space
        self.neighbors = space.neighbors  # might be used for collision avoidance
        self.G = grid                    # graph for A* path calculation
        self.goal = (self.model.p.space_size, round(self.full_pos[1]))  # Goal position
        
        self.update_known_world()
        self.update_goal()
        self.path = a_star_shortestpath(self.G, (round(self.full_pos[0]), round(self.full_pos[1])), self.current_goal)

    def update_known_world(self):
        self.known_world = self.model.G.copy()
        for node in list(self.known_world.nodes):
            if euclidean_distance((round(self.full_pos[0]), round(self.full_pos[1])), node) > self.look_ahead_distance:
                # Remove nodes outside of perception radius
                self.known_world.remove_node(node)
        
    def update_goal(self):
        if euclidean_distance((round(self.full_pos[0]), round(self.full_pos[1])), self.goal) <= self.look_ahead_distance:
            self.current_goal = self.goal  # Goal is within the field of view
        else:
            # Find the nearest point on the boundary of the perception radius
            closest_node = None
            closest_distance = float('inf')
            for node in self.known_world.nodes:
                dist_to_goal = euclidean_distance(node, self.goal)
                if dist_to_goal < closest_distance:
                    closest_node = node
                    closest_distance = dist_to_goal
            self.current_goal = closest_node

    def percept_semaphore(self, semaphore_x, semaphore_state):
        self.semaphore_x = semaphore_x         # x-coordinate of the semaphore
        self.semaphore_state = semaphore_state  # semaphore state

    def check_semaphore_state(self):
        if self.semaphore_x is not None:
            distance_to_semaphore = self.semaphore_x - self.full_pos[0]
            if 0 <= distance_to_semaphore <= self.look_ahead_distance:
                semaphore_state = self.model.semaphore[0].state
                return semaphore_state, distance_to_semaphore
        return None, None

    def get_neighbor_paths(self):
        neighbor_paths = []
        for neighbor in self.neighbors(self, self.look_ahead_distance):
            if hasattr(neighbor, 'path'):
                neighbor_paths.append((neighbor, neighbor.path))
        return neighbor_paths

    def check_for_collisions_and_recalculate_path(self):
        neighbor_paths = self.get_neighbor_paths()
        next_pos = (self.path[1][0], self.path[1][1]) if self.path and len(self.path) > 1 else None
        
        for neighbor, n_path in neighbor_paths:
            neighbor_next_pos = (n_path[1][0], n_path[1][1]) if n_path and len(n_path) > 1 else None
        
            if n_path and (n_path[0] == next_pos or neighbor_next_pos == next_pos):
                # print(f"Collision detected with neighbor {neighbor} at {next_pos}")
         
                if neighbor.priority > self.priority:
                    # Neighbor has higher priority, so recalculate path for this agent
                    # neighbor_node = (round(neighbor.full_pos[0]), round(neighbor.full_pos[1]))
                    if neighbor_next_pos in self.known_world:
                        self.known_world.remove_node(neighbor_next_pos)
                        # print(f"Removed neighbor's node {neighbor_node} from known world.")
                    
                    # Verify that start and goal nodes exist in the graph
                    current_pos = (round(self.full_pos[0]), round(self.full_pos[1]))
                    if current_pos not in self.known_world.nodes or self.current_goal not in self.known_world.nodes:
                        print(f"Cannot find path, as either current position {current_pos} or goal {self.current_goal} is not in known world.")
                        return True, neighbor

                    # Recalculate the path
                    self.path = a_star_shortestpath(
                    self.known_world, (round(self.full_pos[0]), round(self.full_pos[1])), self.current_goal)
                
                    # After recalculation, check again if the new path still collides
                    return True, neighbor
                elif neighbor.priority < self.priority:
                    # Neighbor should yield
                    continue
                else:
                    # Equal priority, choose to stop or proceed based on another rule
                    print(f"Equal priority with neighbor {neighbor}. Deciding to stop.")
                    return True, neighbor 
        
        #No collision detected
        return False, None

    def update_position(self):
        # Update the known world and current goal
        self.update_known_world()
        self.update_goal()
        semaphore_state, distance_to_semaphore = self.check_semaphore_state()
        
        # Recalculate the path if needed
        if not self.path or self.path[0] != (round(self.full_pos[0]), round(self.full_pos[1])) or len(self.path) == 1 and self.goal != (round(self.full_pos[0]), round(self.full_pos[1])):
            self.path = a_star_shortestpath(
                self.known_world, (round(self.full_pos[0]), round(self.full_pos[1])), self.current_goal)

        # Check for collisions and recalculate path if necessary
        collision, neighbor = self.check_for_collisions_and_recalculate_path()

        pos = (round(self.full_pos[0]), round(self.full_pos[1]))
        self.vectors.append(pos)
        # Move along the path 
        if not collision and self.path and (round(self.full_pos[0]), round(self.full_pos[1])) != self.goal:
            if  (semaphore_state is None or semaphore_state == 1 and distance_to_semaphore <= self.look_ahead_distance):
                current_pos = self.path.pop(0)
                next_pos = (self.path[0][0], self.path[0][1], 0)
                self.space.move_to(self, next_pos)
            elif collision:
                return
                # print(f"Stopping to avoid collision with neighbor")
        else:
            return
            # print("Goal reached or collision detected. Stopping.")



class Semaphore(ap.Agent):
    def setup(self):
        self.state = 0  # 0 - red, 1 - green

    def setup_pos(self, space):
        self.space = space
        self.pos = space.positions[self]

    def toggle_state(self):
        self.state = 1 if self.state == 0 else 0

    def set_state(self, number):
        if number in [0, 1]:
            self.state = number
        else:
            raise ValueError("State must be 0 or 1")

    def communicate_state(self, pedestrian):
        if isinstance(pedestrian, Pedestrian):
            pedestrian.percept_semaphore(self.pos[0], self.state)


class Vehicle(ap.Agent):
    def setup(self):
        self.velocity = [1, 0, 0]
        self.look_ahead_distance = 5

    def setup_pos(self, space):
        self.space = space
        self.pos = space.positions[self]
        self.neighbors = space.neighbors


class Model(ap.Model):
    def setup(self):
        # Create space
        self.space = ap.Space(
            self, shape=[self.p.space_size]*self.p.ndim, torus=False)

        # Define non-walkable space road | building | etc

        # from sidewalk width to reamining space
        impassable_cells = []
        for y in range(self.p.sidewalk_size+1, self.p.space_size+1):
            for x in range(self.p.space_size+1):
                impassable_cells.append((x, y))

        # Define random obstacles in the space
        for i in range(15):
            impassable_cells.append(
                (random.randint(20, self.p.space_size), random.randint(0, 10)))

        # Discretize the space for A* algorithm to find pedestrian routes
        self.G = create_grid_graph(
            self.p.space_size, self.p.space_size, impassable_cells)

        # Create Pedestrian and sempahore agents
        self.pedestrianAgents = ap.AgentList(self, self.p.n_people, Pedestrian)
        self.semaphore = ap.AgentList(self, 1, Semaphore)

        # Create random positons according to sidewalk size and agent inbetween space
        available_x_pos = np.arange(
            0, self.p.sidewalk_size, self.p.ag_x_distance)
        available_y_pos = np.arange(
            0, self.p.sidewalk_size, self.p.ag_y_distance)
        available_positions = [(x, y, 0)
                               for x in available_x_pos for y in available_y_pos]

        # Randomly select positions for agents
        ag_pos = random.sample(available_positions, self.p.n_people)
    
        # Add agents to space (Pedestrians and semaphores)
        self.space.add_agents(self.pedestrianAgents, ag_pos)
        self.space.add_agents(self.semaphore, [(30, 5, 0)])

        # Set up semaphore and pdestrian own position
        self.semaphore.setup_pos(self.space)
        # self.pedestrianAgents.setup_pos(self.space)
        self.pedestrianAgents.setup_pos(self.space, self.G)

        # set random prorities
        for agent in self.pedestrianAgents:
            agent.priority = random.randint(1, 100)

        # Add IDs
        for i, agent in enumerate(self.pedestrianAgents):
            agent.id = i


    def step(self):
        # Toggle the semaphore state every n steps
        if self.t % 45 == 0:
            self.semaphore.toggle_state()

        # Communicate the semaphore state to all pedestrians
        for pedestrian in self.pedestrianAgents:
            self.semaphore[0].communicate_state(pedestrian)

        # Pedestrians move based on their updated velocities
        self.pedestrianAgents.update_position()


def animation_plot_single(m, ax):
    ndim = m.p.ndim
    ax.set_title(f"Model {ndim}D t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # Transform
    semaphore_state = m.semaphore[0].state
    color_list = ['black', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray']
    if semaphore_state == 0:
        colors = [color_list[i] for i in range(m.p.n_people)] + ['red']
    else: # 
        colors = [color_list[i] for i in range(m.p.n_people)] + ['green']
    ax.scatter(*pos, s=20, c=colors)
    ax.set_xlim(0, m.p.space_size)
    ax.set_ylim(0, m.p.space_size)
    if ndim == 3:
        ax.set_zlim(0, m.p.space_size)
    ax.grid(True)        


def animation_plot(m, p):
    projection = '3d' if p['ndim'] == 3 else None
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=projection)
    animation = ap.animate(m(p), fig, ax, animation_plot_single)
    # print last postion of agents

    return IPython.display.HTML(animation.to_jshtml(fps=20))


# Start
print("Starting ...")

# Set up parameters
parameters = {
    'steps': 30,       # Number of simulation steps

    'space_size': 50,  # 1 metro por unidad
    'ndim': 3,         # 3D space

    'n_people': 2,       # Number of pedestrian agents
    'sidewalk_size': 10,  # 1 metro por unidad

    # Spacing between pedestriang agents
    'ag_x_distance': 0.7,
    'ag_y_distance': 1,
}

# animation_plot(Model, parameters)

# animation_plot(Model, parameters)

# Run the model
# model = Model(parameters)
# run = model.run()

# info = {}
# for agent in model.pedestrianAgents:
#     info[agent.id] = agent.vectors
   

# # Send parametrs, send result stored in postion
# json_output = json.dumps(info, indent=4)


# json_parameters = {"parameters" : parameters }
# json_parameters = json.dumps(json_parameters, indent=4)

# data1 = json.loads(json_output)
# data2 = json.loads(json_parameters)

# merge = {**data2, **data1}
# merge_json = json.dumps(merge, indent=4)

# with open("positions.json", "w") as f:
#     print(merge_json, file=f)

