import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib.colors import ListedColormap
import random

# CONSTANTS
TREE, FIRE, ASH = 0, 1, 2
COLORS = ['forestgreen', 'orange', 'black']
CUSTOM_MAP = ListedColormap(COLORS) 
GRID_SIZE = 150

# MODEL
grid = np.full((GRID_SIZE, GRID_SIZE), TREE, dtype=int)

rng = np.random.default_rng(42) #seed yung 42
tree_texture = rng.uniform(0.75, 1.15, size=(GRID_SIZE, GRID_SIZE)) 

def colorTrees(state_grid):
    palette = np.array([
        [34, 139, 34],   
        [255, 140, 0],  
        [25, 25, 25],      
    ], dtype=float) / 255.0

    rgb = palette[state_grid]
    tree_mask = (state_grid == TREE)
    rgb[tree_mask] = np.clip(rgb[tree_mask] * tree_texture[tree_mask, None], 0, 1) #turnup brightness, clamp to 1

    return rgb


# def generate_terrain(height, width, water_prob=0.05):
#     new_grid = np.full((height, width), TREE, dtype=int)

#     for hei in range(height):
#         for wid in range(width):
#             water_coef = 10 if hei > 0 and wid > 0 and (new_grid[hei-1, wid] == WATER or new_grid[hei, wid-1] == WATER) else 1
#             if water_prob * water_coef > random.random():
#                 new_grid[hei, wid] = WATER
#                 continue
    
#     return new_grid

# grid = generate_terrain(GRID_SIZE, GRID_SIZE)

# INDEPENDENT VARIABLE
# 1. Probability of the fire spreading
# 2. Wind direction (can be random with random.choice(['N', 'E', 'W', 'S', 'NE', 'NW', 'SE', 'SW']))
# 3. Where in the model the fire starts
spread_probability = 0.5
wind_direction = 'N'
for i in range(0, 20):
    r, c = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    if grid[r, c] == TREE:
        grid[r, c] = FIRE

# DEPENDENT VARIABLE
burnt_percentage = 0

# Main function that follows the concept of CELLULAR AUTOMATON
# Each cell has a state and a neighbor
# The state of each cell changes over time
# The state of eacch cell in the nesxt step is determined by the current state of the cell and neighboring cells
# A set of rules defines how the state of a cell is updated

# If a cell is at the state of TREE, check the neighboring cells
# If the neighboring cells are on FIRE, check if their position is against the wind direction with respect to the cell
# A tree on fire will turn into ASH
def simulate(frame, img, grid):
    new_grid = grid.copy()
    for row in range(1, GRID_SIZE - 1):
        for col in range(1, GRID_SIZE - 1):
            
            # Turn trees that are burnt to ash
            if grid[row, col] == FIRE:
                new_grid[row, col] = ASH
            
            # Check if the neighboring trees
            elif grid[row, col] == TREE:
                tree_neighbors = {
                    'N':    grid[row-1, col],
                    'E':    grid[row, col+1],    
                    'W':    grid[row, col-1],    
                    'S':    grid[row+1, col],    
                    'NE':   grid[row-1, col+1],    
                    'NW':   grid[row-1, col-1],    
                    'SE':   grid[row+1, col+1],    
                    'SW':   grid[row+1, col-1]    
                }
                if FIRE in tree_neighbors.values():
                    current_p = 0.05
                    wind_force = False

                    if wind_direction == 'N' and FIRE in [tree_neighbors['S'], tree_neighbors['SW'], tree_neighbors['SE']]:   wind_force = True
                    elif wind_direction == 'S' and FIRE in [tree_neighbors['N'], tree_neighbors['NW'], tree_neighbors['NE']]: wind_force = True
                    elif wind_direction == 'W' and FIRE in [tree_neighbors['E'], tree_neighbors['NE'], tree_neighbors['SE']]: wind_force = True
                    elif wind_direction == 'E' and FIRE in [tree_neighbors['W'], tree_neighbors['NW'], tree_neighbors['SW']]: wind_force = True
                    elif wind_direction == 'NE' and FIRE in [tree_neighbors['SW'], tree_neighbors['S'], tree_neighbors['W']]: wind_force = True
                    elif wind_direction == 'NW' and FIRE in [tree_neighbors['SE'], tree_neighbors['S'], tree_neighbors['E']]: wind_force = True
                    elif wind_direction == 'SE' and FIRE in [tree_neighbors['NW'], tree_neighbors['N'], tree_neighbors['W']]: wind_force = True
                    elif wind_direction == 'SW' and FIRE in [tree_neighbors['NW'], tree_neighbors['N'], tree_neighbors['E']]: wind_force = True

                    if wind_force:
                        current_p += spread_probability
                    if np.random.random() < current_p:
                        new_grid[row, col] = FIRE
    
    burnt_percentage = ((np.sum(new_grid == ASH) / GRID_SIZE**2)) * 100
    ax.set_xlabel(f'{burnt_percentage:.2f}% of the forest burned down, {100 - burnt_percentage:.2f}% remained unaffected')
    img.set_data(colorTrees(new_grid))
    grid[:] = new_grid[:]
    return img,

fig, ax = plt.subplots()
img = ax.imshow(colorTrees(grid), interpolation='nearest')
ani = anime.FuncAnimation(fig, simulate, fargs=(img, grid), frames=200, interval=100)
burnt_percentage = (np.sum(grid == ASH) / (GRID_SIZE**2)) * 100

plt.title(f'Forest Fire Simulation\nWind Direction of {wind_direction}')
plt.show()