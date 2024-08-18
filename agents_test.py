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
        self.velocity = [1, 0, 0]
        self.semaphore_state = None

    def receive_semaphore_state(self, state):
        self.semaphore_state = state

    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):
        if self.semaphore_state == 1:  # Green light
            self.velocity = [0, 0, 0]  # Stop
        elif self.semaphore_state == 0  :  # Red light
            self.velocity = [1, 0, 0] 

    def update_position(self):
        self.space.move_by(self, self.velocity)


class Semaphore(ap.Agent):
    def setup(self):
        self.state = 0  # 0 - red, 1 - green

    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def set_state(self):
        self.state = 1 if self.state == 0 else 0

    def get_state(self):
        return self.state

    def communicate_state(self):
        # Get all agents within 2 units of the semaphore
        nearby_pedestrians = self.space.neighbors(self, distance=2)

        # Communicate the state to nearby pedestrians
        for pedestrian in nearby_pedestrians:
            if isinstance(pedestrian, Pedestrian):
                pedestrian.receive_semaphore_state(self.state)


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
        # Toggle the semaphore state every 5 steps
        if self.t % 8 == 0:
            self.semaphore.set_state()
        
        # Semaphore communicates its state to nearby pedestrians
        self.semaphore.communicate_state()

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
