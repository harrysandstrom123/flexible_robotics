import numpy as np
import matplotlib.pyplot as plt
from utils import plotrod_simple, objfun, computeTangent, computeSpaceParallel, computeMaterialDirectors, getKappa

# Inputs
nv = 50 # number of nodes
ne = nv - 1
ndof = 3*nv + ne

k_D = []

dz_finals_list = []
# D_list = np.linspace(0.1, 0.5, 10)
D_list = [0.04]
# for D in np.linsapce(0.1, 0.5, 10):
for D in D_list:
  # Helix parameters
  r0 = 0.001 # cross-sectional radius of the rod # Given, d = 0.002 m
  # D = 0.04 # meter: helix diameter
  pitch = 2 * r0
  N = 5 # Number of turns
  # a and b are parameters used in standard (wikipedia) definition of helix
  a = D/2 # Helix radius
  b = pitch / (2.0 * np.pi)
  T = 2.0 * np.pi * N # Angle created by the helix (N turns in the center)
  L = T * np.sqrt( a**2 + b ** 2) # Arc length of the helix
  axial_l = N * pitch # Axial length

  print('Helix diameter = ', D)
  print('Pitch = ', pitch)
  print('N = ', N)
  print('Arc length = ', L)
  Estimated_Arc = np.pi * D * N
  print('Estimated arc length = ', Estimated_Arc)
  print('axial_l = ', axial_l)

  # Create our nodes matrix
  nodes = np.zeros((nv, 3))
  for c in range(nv):
    t = c * T / (nv - 1.0)
    nodes[c,0] = a * np.cos(t)
    nodes[c,1] = a * np.sin(t)
    nodes[c,2] = - b * t


  # Elastic Stiffness

  # Material Parameters
  Y = 10e6 # 10 MPa - Young's modulus
  nu = 0.5 # Poisson's ration
  G = Y / ( 2 * (1 + nu)) # Shear modulus

  # Stiffness variables
  EA = Y * np.pi * r0**2 # Stretching stiffness
  EI = Y * np.pi * r0**4 / 4.0 # Bending stiffness
  GJ = G * np.pi * r0**4 / 2.0 # Twisting stiffness

  # Time Parameters
  totalTime = 7000 # seconds - total time of the simulation
  if np.size(D_list) > 1:
    dt = 30
  else:
    dt = 30 # TIme step size -- may need to be adjusted

  # Tolerance
  tol = EI / L ** 2 * 1e-3

  # Mass Vector and Matrix
  rho = 1000 # kg/m^3 -- density
  totalM = L * np.pi * r0**2 * rho  # Total mass of the rod
  dm = totalM / ne

  massVector = np.zeros(ndof)
  for c in range(nv):
    ind = [4*c, 4*c+1, 4*c+2] # x, y, z coordinates of c-th node
    if c == 0 or c == nv - 1:
      massVector[ind] = dm / 2
    else:
      massVector[ind] = dm

  for c in range(ne):
    massVector[4*c+3] = 0.5 * dm * r0 ** 2 # Equation for a solid cylinder
    # Because r0 is really small, we may get away with just using 0 angular mass

  massMatrix = np.diag(massVector)

  # External Force: Point load on the last node (instead of gravity)
  F_char = EI / L ** 2
  scale_factor = np.geomspace(0.1, 10, 10)
  scale_factor = np.array([1])
  Fends = scale_factor * F_char
  dz_finals = []
  for X in scale_factor:
    F_end = X*F_char
    print("F_end = ",  F_end)
    vectorLoad = np.array([0, 0, -F_end]) # Point load vector

    Fg = np.zeros(ndof) # Eexternal force vector
    c = nv-1
    ind = [4*c, 4*c + 1, 4*c + 2] # last node
    Fg[ind] += vectorLoad

    # Initial DOF vector
    qOld = np.zeros(ndof)
    for c in range(nv):
      ind = [4*c, 4*c + 1, 4*c + 2] # c-th node
      qOld[ind] = nodes[c, :]

    uOld = np.zeros_like(qOld) # Velocity is zero initially

    # plotrod_simple(qOld, 0)

    # Compute the reference lengths
    # Reference length of each edge
    refLen = np.zeros(ne)
    for c in range(ne):
      refLen[c] = np.linalg.norm(nodes[c + 1, :] - nodes[c, :])

    voronoiRefLen = np.zeros(nv)
    for c in range(nv):
      if c == 0:
        voronoiRefLen[c] = 0.5 * refLen[c]
      elif c == nv - 1:
        voronoiRefLen[c] = 0.5 * refLen[c - 1]
      else:
        voronoiRefLen[c] = 0.5 * (refLen[c - 1] + refLen[c])

    # Compute the frames
    # Reference frame (At t=0, we initialize it with space parallel reference frame but not mandatory)
    tangent = computeTangent(qOld)

    t0 = tangent[0, :]
    arb_v = np.array([0, 0, -1])
    a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))
    if np.linalg.norm(np.cross(t0, arb_v)) < 1e-3: # Check if t0 and arb_v are parallel
      arb_v = np.array([0, 1, 0])
      a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))

    a1, a2 = computeSpaceParallel(a1_first, qOld)

    # Material frame
    theta = qOld[3::4] # Extract theta angles
    m1, m2 = computeMaterialDirectors(a1, a2, theta)

    # Natural Curvature and Twist
    # Reference twist
    refTwist = np.zeros(nv) # Or use the function we computed

    # Natural curvature
    kappaBar = getKappa(qOld, m1, m2)

    # Natural twist
    twistBar = np.zeros(nv)

    # Set up boundary conditions: First two nodes and first theta angle is fixed
    # Fixed and free DOFs
    fixedIndex = np.arange(0, 7)
    freeIndex = np.arange(7, ndof)
    # If we include the x and y coordinates of the last node as FIXED DOFs, we will get better agreement

    # Time Stepping loop
    Nsteps = round(totalTime / dt ) # number of steps
    ctime = 0 # Current time
    endZ_0 = qOld[-1] # End Z coordinate of the first node
    endZ = np.zeros(Nsteps)

    a1_old = a1
    a2_old = a2

    timeStep = 0
    n_window = round(totalTime*0.1/dt)  # Number of steps in the window to check for convergence
    dz_ratio = 1

    while timeStep < Nsteps and dz_ratio > 0.01:

      print('Current time: ', ctime)

      q_new, u_new, a1_new, a2_new, flag = objfun(qOld, uOld, a1_old, a2_old,
                                            freeIndex, dt, tol, refTwist,
                                            massVector, massMatrix,
                                            EA, refLen,
                                            EI, GJ, voronoiRefLen,
                                            kappaBar, twistBar,
                                            Fg)

      # Save endZ (z coordinate of the last node)
      endZ[timeStep] = q_new[-1] - endZ_0

      if timeStep > n_window:
            # End Z at the start of the window (100 steps ago)
            z_max_start = np.max(endZ)
            z_min_start = np.min(endZ)
            z_max_recent = np.max(endZ[timeStep - n_window:timeStep])
            z_min_recent = np.min(endZ[timeStep - n_window:timeStep])
            dz_start = z_max_start - z_min_start
            dz_recent = z_max_recent - z_min_recent

            dz_ratio = np.abs(dz_recent) / (np.abs(dz_start) + 1e-12)
            print('dz_start = ',  dz_start, 'dz_recent = ', dz_recent, 'dz_ratio = ', dz_ratio)

      # if timeStep % 25 == 0:
      #   plotrod_simple(q_new, ctime)

      ctime += dt # Current time
      timeStep += 1
      # Old parameters become new
      qOld = q_new.copy()
      uOld = u_new.copy()
      a1_old = a1_new.copy()
      a2_old = a2_new.copy()

    dz_finals.append(endZ[timeStep-1])

    # plt.figure(2)
    # time_array = np.arange(1, timeStep+1, 1) * dt
    # plt.plot(time_array, endZ[0:timeStep], 'ro-')
    # plt.xlabel('Time (s)')
    # plt.ylabel('End Z (m)')
    # plt.title("Displacement vs Time for F_end = {:.4e} N".format(F_end))
    # plt.show()
  print(dz_finals)

  dz_finals_arr = np.array(dz_finals)
  dz_finals_list.append(dz_finals_arr)
  # Step 1: Prepare the data matrices
  # In the equation F = k*delta, the 'x' matrix (A) is the delta_data
  # We need to reshape delta_data into a 2D column vector (n_samples, 1)
  A = dz_finals_arr.reshape(-1, 1)
  # Step 2: Solve the linear system for 'k' using least-squares
  # The function returns (x, residuals, rank, singular_values)
  solution, residuals, rank, s = np.linalg.lstsq(A, Fends, rcond=None)

  # The fit coefficient k is the first (and only) element of the solution array
  k_fit = solution[0]
  k_D.append(k_fit)

  print(f"F = k*delta Fit (Zero Intercept)")
  print(f"Stiffness k = {k_fit:.4e} N/m")
  print(f"Sum of Squared Residuals = {residuals[0]:.4e}")

  # To plot the best-fit line:
  F_fit = k_fit * dz_finals_arr # Calculate F values based on the best-fit k

  plt.figure(3)
  plt.plot(dz_finals_arr, Fends, 'bo', label='Data Points')
  plt.plot(dz_finals_arr, F_fit, 'r-', label='Best Fit Line')
  plt.title(f'Force vs Displacement Fit for D = {D} m')
  plt.xlabel('Displacement (m)')
  plt.ylabel('Force (N)')


print('k_D = ', k_D)
print('dz_finals_list = ', dz_finals_list) 
