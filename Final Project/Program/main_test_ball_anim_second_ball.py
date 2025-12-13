# program based on Lecture 14 Discrete Elastic Plates by Prof. M. Khalid Jawed
import numpy as np
import matplotlib.pyplot as plt
# from utils_ball_v2 import *
from utils_ball_v3 import *

#ball 1 mass:
mb1 = 5
mb2 = 0.5

# MESH & MATERIAL CONFIGURATION
Lx = 1.0
Ly = 1.0
Nx = 25   
Ny = 25   

# Node coordinates
x_coords = np.linspace(0, Lx, Nx)
y_coords = np.linspace(0, Ly, Ny)

# Initial Shell Configuration (qOld)
qOld = np.zeros(3 * Nx * Ny)
node_index = 0
for j in range(Ny):
    for i in range(Nx):
        idx = 3 * node_index
        qOld[idx:idx+3] = [x_coords[i], y_coords[j], 0]
        node_index += 1

nv = Nx * Ny  
ndof_shell = 3 * nv 

# Edges
edges_list = []
for j in range(Ny):
    for i in range(Nx - 1): # Horizontal
        edges_list.append(((i + j * Nx), ((i + 1) + j * Nx)))
for i in range(Nx):
    for j in range(Ny - 1): # Vertical
        edges_list.append(((i + j * Nx), (i + (j + 1) * Nx)))
for j in range(Ny - 1):
    for i in range(Nx - 1): # Diagonal
        edges_list.append(((i + j * Nx), ((i + 1) + (j + 1) * Nx)))
edges = np.array(edges_list)

# Hinges
hinges_list = []
for j in range(Ny - 1):
    for i in range(Nx - 1):
        n1, n2 = i + j * Nx, (i + 1) + j * Nx
        n3, n4 = i + (j + 1) * Nx, (i + 1) + (j + 1) * Nx
        hinges_list.append((n1, n4, n2, n3))
        hinges_list.append((n2, n3, n1, n4))
hinges = np.array(hinges_list)

# Material Properties
Y = 0.01E9 
h = 0.001 
kb = 2.0 / np.sqrt(3.0) * Y * h**3.0 / 12 

# Stretching setup
refLen = np.zeros(edges.shape[0]) 
ks = np.zeros_like(refLen) 
for kEdge in range(edges.shape[0]):
    node0, node1 = edges[kEdge, 0], edges[kEdge, 1]
    x0 = qOld[3*node0:3*node0+3]
    x1 = qOld[3*node1:3*node1+3]
    refLen[kEdge] = np.linalg.norm(x1 - x0)
    ks[kEdge] = np.sqrt(3.0) / 2.0 * Y * h * (refLen[kEdge])**2

# Simulation Parameters
totalTime = 15 # Extended time to see second ball
dt = 0.001 
tol = kb / (0.01) * 1e-3 
visc = 0.1 # Added viscosity for stability

rho = 95 # Density
totalM = Lx*Ly*h*rho # total mass in kg
print("Total Mass =", totalM)

# Shell Mass & Gravity
massVector = np.zeros(ndof_shell)
dm = totalM / nv 
for c in range(nv):
    massVector[3*c:3*c+3] = dm
g = np.array([0, 0, -9.8])
Fg = np.zeros(ndof_shell)
for c in range(nv):
    Fg[3*c:3*c+3] = massVector[3*c:3*c+3] * g
thetaBar = 0 
uOld = np.zeros(ndof_shell) 

# Boundary Conditions (Fixed Edges)
fixedIndex_list = []
for j in range(Ny):
    fixedIndex_list.extend([3*(j*Nx)+k for k in range(3)]) # Left
    fixedIndex_list.extend([3*((Nx-1)+j*Nx)+k for k in range(3)]) # Right
for i in range(Nx):
    fixedIndex_list.extend([3*i+k for k in range(3)]) # Bottom
    fixedIndex_list.extend([3*(i+(Ny-1)*Nx)+k for k in range(3)]) # Top
fixedIndex = np.array(sorted(list(set(fixedIndex_list))))
freeIndex_shell = np.setdiff1d(np.arange(ndof_shell), fixedIndex)

# dynamic ball config
MAX_BALLS = 10 
ndof_balls = 3 * MAX_BALLS
ndof_total = ndof_shell + ndof_balls

# Global Arrays
massVector_total = np.zeros(ndof_total)
Fg_total = np.zeros(ndof_total)
qOld_total = np.zeros(ndof_total)
uOld_total = np.zeros(ndof_total)

# Initialize Shell Data
massVector_total[:ndof_shell] = massVector
Fg_total[:ndof_shell] = Fg
qOld_total[:ndof_shell] = qOld
uOld_total[:ndof_shell] = uOld

# Active Mask & Radii
ball_active = np.zeros(MAX_BALLS, dtype=bool) 
ball_radii = np.zeros(MAX_BALLS)

# Helper to Spawn Ball
def spawn_ball(idx, radius, mass, pos, vel=[0,0,0]):
    if idx >= MAX_BALLS: return
    
    start_dof = ndof_shell + (idx * 3)
    ind = [start_dof, start_dof+1, start_dof+2]
    
    massVector_total[ind] = mass
    Fg_total[ind] = mass * g
    qOld_total[ind] = pos
    uOld_total[ind] = vel
    
    ball_radii[idx] = radius
    ball_active[idx] = True
    print(f"--> Ball {idx} spawned at t={ctime:.3f}")

# Contact Parameters
# Area-weighted stiffness 
area_per_node = (Lx * Ly) / (Nx * Ny)
K_contact = 5e8 * area_per_node  /100
mu_contact = 0.1 

# INITIAL STATE

# Initialize time BEFORE spawning the first ball so the print statement works
ctime = 0.0

# Spawn first ball immediately
spawn_ball(0, 0.15, 10.0, [Lx/2.0, Ly/2.0, 0.1])

# Initialize Mass Matrix (Diagonal)
massMatrix_total = np.diag(massVector_total)

# Free Indices (Shell + ALL Ball DOFs)
freeIndex_balls = np.arange(ndof_shell, ndof_total)
freeIndex_total = np.concatenate([freeIndex_shell, freeIndex_balls])

# simulation loop
Nsteps = round(totalTime / dt)
# ctime = 0  <-- REMOVE THIS LINE (it is now initialized above)
q_history = []
frame_stride = 20

for timeStep in range(Nsteps):
    
    # spawn ball
    # At t=0.3s, spawn a second smaller ball, with some velocity
    if abs(ctime - 5) < dt/2:
        spawn_ball(1, 0.1, 0.5, [Lx/2.0 + 0.15, Ly/2.0 + 0.15, 0.025], vel = [-0.5, 0.5, 0.0])
        # Rebuild mass matrix because mass changed
        massMatrix_total = np.diag(massVector_total)

    qNew_total, uNew_total = objfun(qOld_total, uOld_total, freeIndex_total, dt, tol, 
                                    massVector_total, massMatrix_total, 
                                    ks, refLen, edges, 
                                    kb, thetaBar, hinges, 
                                    Fg_total, visc, 
                                    ball_radii, ball_active, # <--- CHANGED ARGS
                                    K_contact, mu_contact, ndof_shell)
    ctime += dt
    
    qOld_total = qNew_total.copy()
    uOld_total = uNew_total.copy()

    # store data
    if timeStep % frame_stride == 0:
        q_history.append(qNew_total.copy())
    if timeStep % 20 == 0:
        print(f"Time: {ctime:.4f}s")
    if timeStep % 500 == 0:
        plotShell(qOld_total, edges, ctime, ball_radii, ball_active, ndof_shell)

#ANIMATION
print("Generating animation...")
animate_shell(q_history, edges, dt*frame_stride, ball_radii, ball_active, ndof_shell, "multi_ball_orbit.gif")

q_history_arr = np.array(q_history)          # shape: (Nframes, ndof_total)
n_frames = q_history_arr.shape[0]

# Time stamps that correspond to each stored frame
t_frames = np.arange(n_frames) * dt * frame_stride

ball1_pos = q_history_arr[:, ndof_shell:ndof_shell+3]   # shape: (Nframes, 3)
ball1_x = ball1_pos[:, 0]
ball1_y = ball1_pos[:, 1]
ball1_z = ball1_pos[:, 2]

ball2_pos = q_history_arr[:, ndof_shell+3:ndof_shell+6]   # shape: (Nframes, 3)
ball2_x = ball2_pos[:, 0]
ball2_y = ball2_pos[:, 1]
ball2_z = ball2_pos[:, 2]

# 1) z-position vs time  ---------------------------------
plt.figure()
plt.plot(t_frames, ball1_z, '-o', markersize=3)
plt.plot(t_frames, ball2_z, '-o', markersize=3)
plt.xlabel('Time [s]')
plt.ylabel('Ball z-position [m]')
plt.title('Ball vertical position vs time')
plt.grid(True)
plt.tight_layout()
plt.savefig('ball_z_vs_time_two_balls.svg')
plt.close()

# 2) x–y path of the ball  -------------------------------
plt.figure()
plt.plot(ball1_x, ball1_y, '-ro', markersize=3)
plt.plot(ball2_x, ball2_y, '-bo', markersize=3)
plt.xlabel('x-position [m]')
plt.ylabel('y-position [m]')
plt.title('Ball trajectory in x–y plane')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.savefig('ball_xy_path_twoballs.svg')
plt.close()
