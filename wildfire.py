import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib.colors import ListedColormap
import random

# --- CONSTANTS ---
TREE, FIRE, ASH, WATER = 0, 1, 2, 3
COLORS = ['forestgreen', 'orange', 'black', 'royalblue'] 
CUSTOM_MAP = ListedColormap(COLORS) 
GRID_SIZE = 100 
WIND_DIRECTIONS = ['N', 'E', 'W', 'S', 'NE', 'NW', 'SE', 'SW']
AMBIENT_TEMP = 38.0  

# MODEL SETUP
rng = np.random.default_rng(42)
tree_texture = rng.uniform(0.75, 1.15, size=(GRID_SIZE, GRID_SIZE)) 

def generate_terrain():
    new_grid = np.full((GRID_SIZE, GRID_SIZE), TREE, dtype=int)
    
    center_col = GRID_SIZE // 2
    for row in range(GRID_SIZE):
        curve = int(12 * np.sin(row / 10.0))
        river_center = center_col + curve
        river_width = 4 
        new_grid[row, river_center - river_width : river_center + river_width] = WATER
    return new_grid

BASE_TERRAIN = generate_terrain()

def colorTrees(state_grid): 
    palette = np.array([
        [34, 139, 34],  
        [255, 140, 0], 
        [25, 25, 25],
        [65, 105, 225] 
    ], dtype=float) / 255.0
    
    rgb = palette[state_grid]
    tree_mask = (state_grid == TREE)
    rgb[tree_mask] = np.clip(rgb[tree_mask] * tree_texture[tree_mask, None], 0, 1)
    return rgb


def run_simulation_logic(spread_p, wind_dir):
    grid_inner = BASE_TERRAIN.copy()
    ignited = 0
    while ignited < 20:
        r, c = random.randint(1, GRID_SIZE-2), random.randint(1, GRID_SIZE-2)
        if grid_inner[r, c] == TREE:
            grid_inner[r, c] = FIRE
            ignited += 1

    while np.any(grid_inner == FIRE):
        new_grid = grid_inner.copy()
        
        for row in range(1, GRID_SIZE - 1):
            for col in range(1, GRID_SIZE - 1):
                if grid_inner[row, col] == FIRE:
                    new_grid[row, col] = ASH
                elif grid_inner[row, col] == TREE:
                    tree_neighbors = {
                        'N': grid_inner[row-1, col], 'E': grid_inner[row, col+1],    
                        'W': grid_inner[row, col-1], 'S': grid_inner[row+1, col],    
                        'NE': grid_inner[row-1, col+1], 'NW': grid_inner[row-1, col-1],    
                        'SE': grid_inner[row+1, col+1], 'SW': grid_inner[row+1, col-1]    
                    }
                    if FIRE in tree_neighbors.values():
                        temp_modifier = max(0.0, (AMBIENT_TEMP - 20.0) * 0.002)
                        current_p = 0.05 + temp_modifier
                        
                        wind_force = False
                        if wind_dir == 'N' and FIRE in [tree_neighbors['S'], tree_neighbors['SW'], tree_neighbors['SE']]:   wind_force = True
                        elif wind_dir == 'S' and FIRE in [tree_neighbors['N'], tree_neighbors['NW'], tree_neighbors['NE']]: wind_force = True
                        elif wind_dir == 'W' and FIRE in [tree_neighbors['E'], tree_neighbors['NE'], tree_neighbors['SE']]: wind_force = True
                        elif wind_dir == 'E' and FIRE in [tree_neighbors['W'], tree_neighbors['NW'], tree_neighbors['SW']]: wind_force = True
                        elif wind_dir == 'NE' and FIRE in [tree_neighbors['SW'], tree_neighbors['S'], tree_neighbors['W']]: wind_force = True
                        elif wind_dir == 'NW' and FIRE in [tree_neighbors['SE'], tree_neighbors['S'], tree_neighbors['E']]: wind_force = True
                        elif wind_dir == 'SE' and FIRE in [tree_neighbors['NW'], tree_neighbors['N'], tree_neighbors['W']]: wind_force = True
                        elif wind_dir == 'SW' and FIRE in [tree_neighbors['NW'], tree_neighbors['N'], tree_neighbors['E']]: wind_force = True

                        if wind_force: current_p += spread_p
                        if np.random.random() < current_p: new_grid[row, col] = FIRE

        grid_inner[:] = new_grid[:]
    
    total_trees = np.sum(BASE_TERRAIN == TREE)
    return (np.sum(grid_inner == ASH) / total_trees) * 100

# --- PSO ALGO ---
num_particles = 10
pso_iterations = 10

particles = np.random.uniform(0, 1, (num_particles, 2))
particles[:, 1] *= 7 
velocities = np.zeros((num_particles, 2))

p_best_pos = particles.copy()
p_best_scores = np.zeros(num_particles)
g_best_pos = particles[0].copy()
g_best_score = -1

print(f"Running PSO to find worst-case fire scenario at {AMBIENT_TEMP} Celsius...")
for i in range(pso_iterations):
    for j in range(num_particles):
        current_wind = WIND_DIRECTIONS[int(round(particles[j, 1])) % 8]
        score = run_simulation_logic(particles[j, 0], current_wind)
        
        if score > p_best_scores[j]:
            p_best_scores[j] = score
            p_best_pos[j] = particles[j].copy()
        
        if score > g_best_score:
            g_best_score = score
            g_best_pos = particles[j].copy()

    for j in range(num_particles):
        r1, r2 = random.random(), random.random()
        velocities[j] = (0.5 * velocities[j] + 
                         1.0 * r1 * (p_best_pos[j] - particles[j]) + 
                         1.0 * r2 * (g_best_pos - particles[j]))
        particles[j] += velocities[j]
        particles[j, 0] = np.clip(particles[j, 0], 0, 1)
        particles[j, 1] = np.clip(particles[j, 1], 0, 7)
    
    print(f"Iteration {i+1}: Current Max Burn Found: {g_best_score:.2f}%")

best_spread = g_best_pos[0]
best_wind = WIND_DIRECTIONS[int(round(g_best_pos[1])) % 8]
print(f"\nOptimization Done! Best params: Spread={best_spread:.2f}, Wind={best_wind}")

grid = BASE_TERRAIN.copy()
ignited = 0
while ignited < 20:
    r, c = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    if grid[r, c] == TREE:
        grid[r, c] = FIRE
        ignited += 1

def simulate_final(frame, img, grid):
    new_grid = grid.copy()
    for row in range(1, GRID_SIZE - 1):
        for col in range(1, GRID_SIZE - 1):
            if grid[row, col] == FIRE:
                new_grid[row, col] = ASH
            elif grid[row, col] == TREE:
                tree_neighbors = {
                    'N': grid[row-1, col], 'E': grid[row, col+1], 'W': grid[row, col-1], 'S': grid[row+1, col],    
                    'NE': grid[row-1, col+1], 'NW': grid[row-1, col-1], 'SE': grid[row+1, col+1], 'SW': grid[row+1, col-1]    
                }
                if FIRE in tree_neighbors.values():
                    temp_modifier = max(0.0, (AMBIENT_TEMP - 20.0) * 0.002)
                    current_p = 0.05 + temp_modifier
                    
                    wind_force = False
                    if best_wind == 'N' and FIRE in [tree_neighbors['S'], tree_neighbors['SW'], tree_neighbors['SE']]:   wind_force = True
                    elif best_wind == 'S' and FIRE in [tree_neighbors['N'], tree_neighbors['NW'], tree_neighbors['NE']]: wind_force = True
                    elif best_wind == 'W' and FIRE in [tree_neighbors['E'], tree_neighbors['NE'], tree_neighbors['SE']]: wind_force = True
                    elif best_wind == 'E' and FIRE in [tree_neighbors['W'], tree_neighbors['NW'], tree_neighbors['SW']]: wind_force = True
                    elif best_wind == 'NE' and FIRE in [tree_neighbors['SW'], tree_neighbors['S'], tree_neighbors['W']]: wind_force = True
                    elif best_wind == 'NW' and FIRE in [tree_neighbors['SE'], tree_neighbors['S'], tree_neighbors['E']]: wind_force = True
                    elif best_wind == 'SE' and FIRE in [tree_neighbors['NW'], tree_neighbors['N'], tree_neighbors['W']]: wind_force = True
                    elif best_wind == 'SW' and FIRE in [tree_neighbors['NW'], tree_neighbors['N'], tree_neighbors['E']]: wind_force = True

                    if wind_force: current_p += best_spread
                    if np.random.random() < current_p: new_grid[row, col] = FIRE
    
    total_trees = np.sum(BASE_TERRAIN == TREE)
    burnt_pct = (np.sum(new_grid == ASH) / total_trees) * 100
    ax.set_xlabel(f'{burnt_pct:.2f}% of trees burnt using Optimized Params', color='white', fontweight='bold')
    img.set_data(colorTrees(new_grid))
    grid[:] = new_grid[:]
    return img,

fig, ax = plt.subplots(facecolor='black')
ax.set_facecolor('black')

img = ax.imshow(colorTrees(grid), interpolation='nearest')
plt.title(f'Optimized Fire Simulation with River ({AMBIENT_TEMP}°C)\nWorst Case: {best_wind} Wind, {best_spread:.2f} Prob', color='yellow', fontweight='bold')
ax.tick_params(colors='white')
ani = anime.FuncAnimation(fig, simulate_final, fargs=(img, grid), frames=200, interval=50)
plt.show()