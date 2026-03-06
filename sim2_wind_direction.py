import numpy as np
import matplotlib.pyplot as plt
import random

# CONSTANTS
TREE, FIRE, ASH = 0, 1, 2
GRID_SIZE = 100
FRAMES = 150
TRIALS = 5

DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
FIXED_PROB = 0.5   # <-- change this if you want a different fixed probability

# ── Core simulation ────────────────────────────────────────────────────────────
def run_simulation(wind_direction, spread_probability, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    grid = np.full((GRID_SIZE, GRID_SIZE), TREE, dtype=int)
    for _ in range(20):
        r = random.randint(0, GRID_SIZE - 1)
        c = random.randint(0, GRID_SIZE - 1)
        if grid[r, c] == TREE:
            grid[r, c] = FIRE

    for _ in range(FRAMES):
        new_grid = grid.copy()
        for row in range(1, GRID_SIZE - 1):
            for col in range(1, GRID_SIZE - 1):
                if grid[row, col] == FIRE:
                    new_grid[row, col] = ASH
                elif grid[row, col] == TREE:
                    nb = {
                        'N':  grid[row-1, col],
                        'E':  grid[row, col+1],
                        'W':  grid[row, col-1],
                        'S':  grid[row+1, col],
                        'NE': grid[row-1, col+1],
                        'NW': grid[row-1, col-1],
                        'SE': grid[row+1, col+1],
                        'SW': grid[row+1, col-1],
                    }
                    if FIRE in nb.values():
                        p = 0.05
                        wind_force = False
                        if wind_direction == 'N'  and FIRE in [nb['S'],  nb['SW'], nb['SE']]: wind_force = True
                        elif wind_direction == 'S'  and FIRE in [nb['N'],  nb['NW'], nb['NE']]: wind_force = True
                        elif wind_direction == 'W'  and FIRE in [nb['E'],  nb['NE'], nb['SE']]: wind_force = True
                        elif wind_direction == 'E'  and FIRE in [nb['W'],  nb['NW'], nb['SW']]: wind_force = True
                        elif wind_direction == 'NE' and FIRE in [nb['SW'], nb['S'],  nb['W']]:  wind_force = True
                        elif wind_direction == 'NW' and FIRE in [nb['SE'], nb['S'],  nb['E']]:  wind_force = True
                        elif wind_direction == 'SE' and FIRE in [nb['NW'], nb['N'],  nb['W']]:  wind_force = True
                        elif wind_direction == 'SW' and FIRE in [nb['NW'], nb['N'],  nb['E']]:  wind_force = True
                        if wind_force:
                            p += spread_probability
                        if np.random.random() < p:
                            new_grid[row, col] = FIRE
        grid[:] = new_grid[:]

    return (np.sum(grid == ASH) / GRID_SIZE**2) * 100


# ── Run simulations ────────────────────────────────────────────────────────────
print(f"=== Simulation 2: Fixed Probability = {FIXED_PROB}, Varying Wind Direction ===")
means, stds = [], []

for direction in DIRECTIONS:
    trial_vals = []
    for t in range(TRIALS):
        print(f"  dir={direction}  trial {t+1}/{TRIALS}", end="\r")
        trial_vals.append(run_simulation(direction, FIXED_PROB, seed=t * 13))
    means.append(np.mean(trial_vals))
    stds.append(np.std(trial_vals))
    print(f"  dir={direction}  → {means[-1]:.1f}% ± {stds[-1]:.1f}%        ")


# ── Plot ───────────────────────────────────────────────────────────────────────
BG_DARK, BG_PANEL = '#1a1a2e', '#16213e'

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_PANEL)

x = np.arange(len(DIRECTIONS))
norm_colors = plt.cm.YlOrRd(np.array(means) / 100)

bars = ax.bar(x, means, color=norm_colors, edgecolor='#555', linewidth=0.8, zorder=3)
ax.errorbar(x, means, yerr=stds, fmt='none', color='white',
            capsize=5, elinewidth=1.4, capthick=1.4, zorder=4)

for bar, val, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1.2,
            f'{val:.1f}%', ha='center', va='bottom', color='white',
            fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(DIRECTIONS)
ax.set_ylim(0, 100)
ax.set_xlabel('Wind Direction', fontsize=12, color='white')
ax.set_ylabel('% of Forest Burned', fontsize=12, color='white')
ax.set_title(f'Effect of Wind Direction on Forest Burned\n(Spread Probability fixed = {FIXED_PROB})',
             color='white', fontsize=13, fontweight='bold', pad=12)
ax.tick_params(colors='white')
ax.yaxis.grid(True, color='#2a2a4a', linewidth=0.8, zorder=0)
for spine in ax.spines.values():
    spine.set_edgecolor('#333')

plt.tight_layout()
plt.savefig('sim2_wind_direction.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print("\nDone! Chart saved as 'sim2_wind_direction.png'")
plt.show()
