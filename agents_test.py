# Model design
import agentpy as ap
import numpy as np

# Random
import random

# Visualization
import matplotlib.pyplot as plt
import IPython


class Pedestrian(ap.Agent):
    def setup(self):
        # Agent attributes
        self.priority = 0             # migth be used for agent neogotiation
        self.velocity = [1, 0, 0]     # Initial velocity for movement
        self.look_ahead_distance = 5  # Distance to start checking the semaphore

        # Agent perceptions
        self.semaphore_x = None     # x-coord of the semaphore (intersection)
        self.sempahore_state = None # semaphore state (0 - red, 1 - green)

    def setup_pos(self, space):
        self.space = space               # space where the agent is
        self.pos = space.positions[self] # agent position in space
        self.neighbors = space.neighbors # might be used for collision avoidance

    def percept_semaphore(self, semaphore_x, semaphore_state):
        self.semaphore_x = semaphore_x         # x-coordinate of the semaphore
        self.semaphore_state = semaphore_state # semaphore state

    def check_semaphore_state(self):
        if self.semaphore_x is not None:
            distance_to_semaphore = self.semaphore_x - \
                self.pos[0]  # distance of agent to semaphore
            # if distance is within perception, check semaphore state
            if 0 <= distance_to_semaphore <= self.look_ahead_distance:
                semaphore_state = self.model.semaphore.get_state()
                return semaphore_state, distance_to_semaphore
        return None, None

    def update_velocity(self):
        # Check if a semaphore can be seen
        # (update sempahore_x)

        semaphore_state, distance_to_semaphore = self.check_semaphore_state()

        if semaphore_state is not None:
            if semaphore_state == 0 and distance_to_semaphore <= self.look_ahead_distance:
                # Red light: Start slowing down or stop
                self.velocity = [max(0, 1 - (self.look_ahead_distance - distance_to_semaphore)/self.look_ahead_distance), 0, 0]
            elif semaphore_state == 1:
                # Green light: Continue moving at normal speed
                self.velocity = [1, 0, 0]
        else:
            # No semaphore or not within look-ahead distance: Continue moving at normal speed
            self.velocity = [1, 0, 0]

    def update_position(self):
        self.space.move_by(self, self.velocity)


class Semaphore(ap.Agent):
    def setup(self):
        self.state = 0  # 0 - red, 1 - green

    def setup_pos(self, space):
        self.space = space
        self.pos = space.positions[self]
        self.neighbors = space.neighbors

    def toggle_state(self):
        self.state = 1 if self.state == 0 else 0

    def set_state(self, number):
        # 0 - red, 1 - green
        self.state = number

    def communicate_state(self, pedestrian):
        # Get all agents within 2 units of the semaphore
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

        # Create Pedestrian agents
        self.pedestrianAgents = ap.AgentList(self, self.p.n_people, Pedestrian)

        # Create Semaphore agent
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
        self.space.add_agents(self.pedestrianAgents, ag_pos)

        # Add semaphore to space
        self.space.add_agents(self.semaphore, [(30, 5, 0)])

        # Set up semaphore position
        self.semaphore.setup_pos(self.space)

        # Set up agents positions
        self.pedestrianAgents.setup_pos(self.space)

    def step(self):
        # Toggle the semaphore state every 8 steps
        if self.t % 8 == 0:
            self.semaphore.toggle_state()

        # Pedestrians update their velocities based on the semaphore state
        self.pedestrianAgents.update_velocity()

        # Pedestrians move based on their updated velocities
        self.pedestrianAgents.update_position()


def animation_plot_single(m, ax):
    ndim = m.p.ndim
    ax.set_title(f"Model {ndim}D t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # Transform
    x = m.semaphore.get_state()
    if x[0] == 0:
        colors = ['black', 'black', 'black', 'black', 'black', 'red']
    else:
        colors = ['black', 'black', 'black', 'black', 'black', 'green']
    ax.scatter(*pos, s=20, c=colors)
    # print("Positions: ", *pos)
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
    return IPython.display.HTML(animation.to_jshtml(fps=20))


# Start
print("Starting ...")

# Set up parameters
parameters = {
    'steps': 50,  # Number of simulation steps

    'space_size': 50,  # 1 metro por unidad
    'ndim': 3,  # 3D space

    'n_people': 5,  # Number of people
    'sidewalk_size': 10,  # 1 metro por unidad
    # maybe genrate randomly for each agent likes but initally have a random distance between 0.5 and 1.5
    'ag_x_distance': 0.7,
    'ag_y_distance': 1,
}

animation_plot(Model, parameters)
