import agentpy as ap
import random

class Pedestrian(ap.Agent):
    def setup(self):
        pass # No setup needed

    def move(self, grid, goal_position):
        # Get current position
        x, y = grid.positions[self]

        # Check if the car has reached the goal
        if (x, y) == goal_position:
            print("Reached the goal at position:", goal_position)
            return  # Stop moving

        # Calculate new position
        if x < grid.shape[0] - 1:  # Check if not at the rightmost edge
            new_position = (x + 1, y)
        else:
            return # Stop at the end of the grid
        
        # Remove agent from current position
        grid.remove_agents(self)
        
        # Add agent to new position
        grid.add_agents([self], positions=[new_position])



class Model(ap.Model):
    def setup(self):
        # Create space
        self.space = ap.Space(self, shape=[self.p.road_size]*self.p.ndim, torus=False) # torus true to return to the start
        
        # Create Pedestrian agents
        self.pedestrianAgents = ap.AgentList(self, self.p.n_people, Pedestrian)

        # Create random positons and put agents in space
        for agent in self.pedestrianAgents:
            x_position = random.uniform(0, 2)
            position = [x_position, 0, 0]  # z:0, y:0, x between 0 and 2
            self.space.add_agents(agent, position)

    
    
    def step(self):
        # Move each car in each step (there is only 1 in this example)
        for pedestrian in self.pedestrianAgents:
            # Position before step
            print("\nThis is my grid postion before: ", self.grid.positions[pedestrian])
            
            # Step: Move the car
            pedestrian.move(self.grid, self.goal_position)
            
            # Position after step
            print("This is my grid postion after: ", self.grid.positions[pedestrian])


# Start
print("Starting ...")

# Set up parameters
parameters = {
    'steps': 4,  # Number of simulation steps
    'seed' : 15, # Seed for random number generator

    'n_people': 15,  # Number of cars
    'road_size': 100, # 1 metro por unidad
    'ndim': 3, # 3D space
}

model = Model(parameters) # Create the model
results = model.run()   # Run the simulation

# Access the final positions of the cars (Only 1 car in this example)
for i, car in enumerate(model.pedestrianAgents):
    x, y = model.grid.positions[car]
    print(f'Car {i} final position: ({x}, {y})')

