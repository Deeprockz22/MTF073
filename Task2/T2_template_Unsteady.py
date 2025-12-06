# MTF073 Computational Fluid Dynamics
# Task 2: 2D convection-diffusion
# Case 7 Implementation

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
# Packages needed
import numpy as np
import copy
import matplotlib.pyplot as plt
# Close all plots when running entire code:
plt.close('all')
# All functions of the code (some to be implemented by you):
import T2_codeFunctions_template as cF

#===================== Inputs =====================

# Case number (same as case in description, 1-25)
caseID = 7

# Geometric and mesh inputs (mesh is read from file)
# L, H, mI, mJ, nI, nJ are set later, from the imported mesh
grid_type = 'coarse'  # Either 'coarse' or 'fine'

# Physical properties
rho     = 1      # Density
k       = 1      # Thermal conductivity 
Cp      = 500    # Specific heat
gamma   = k/Cp   # Calculated diffusion coefficient

unsteady = True # True or False

if unsteady:
    # For unsteady:
    deltaT = 1.0   
    endTime = 300.0 
    # Note that a frame is saved every "saveInterval" time step if
    # unsteady = True and createAnimatedPlots = True! Don't overload
    # your computer! Set createAnimatedPlots to False to save time.
    saveInterval = 10 # Save T at every "saveInterval" time step, for
                     # animated plot
    createAnimatedPlots = True # True or False
    # Set any number of probe positions, relative to L and H (0-1)
    probeX = np.array([0.1, 0.9, 0.1, 0.9])
    probeY = np.array([0.1, 0.1, 0.9,0.9])
    
    #probeX = np.array([0.2, 0.5, 0.8])
    #probeY = np.array([0.5, 0.5, 0.5])
else:
    # For steady-state:
    deltaT = 1e30  
    endTime = 1e30 

# Boundary condition value preparation (Case 7)
# Inlet A (West, Top part) -> T = 20 (set via T_in)
# West Wall (West, Bottom part) -> T = 5 (set via T_west)
# Others -> Adiabatic (Homogeneous Neumann)
T_init  = 0      # Initial guess for temperature
T_east  = T_init # Default (Neumann)
T_west  = 5.0    # Case 7 Specific: West Wall fixed at 5K
T_north = T_init # Default (Neumann)
T_south = T_init # Default (Neumann)
q_wall  = 0      # Adiabatic for standard walls
T_in    = 20.0   # Inlet A temperature

# Solver inputs
nExplCorrIter = 2000   # Maximum number of explicit correction iterations
nLinSolIter   = 20    # Number of linear solver iterations
resTol        = 0.001  # Convergence criterium for residuals
solver        = 'TDMA'   # Either 'GS' (Gauss-Seidel) or 'TDMA'

#====================== Code ======================

# Read grid and velocity data:
grid_numbers = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]
grid_number  = grid_numbers[caseID-1]
path = 'data/grid%d/%s_grid' % (grid_number,grid_type)
pointXvector = np.genfromtxt('%s/xc.dat' % (path)) # x node coordinates
pointYvector = np.genfromtxt('%s/yc.dat' % (path)) # y node coordinates
u_datavector = np.genfromtxt('%s/u.dat' % (path))  # u velocity at the nodes
v_datavector = np.genfromtxt('%s/v.dat' % (path))  # v veloctiy at the nodes

nan = float("nan")

# Allocate arrays 
mI     = len(pointXvector);          # Number of mesh points X direction
mJ     = len(pointYvector);          # Number of mesh points X direction
nI     = mI + 1;                     # Number of nodes in X direction, incl. boundaries
nJ     = mJ + 1;                     # Number of nodes in Y direction, incl. boundaries
pointX = np.zeros((mI,mJ))*nan       # X coords of the mesh points, in points
pointY = np.zeros((mI,mJ))*nan       # Y coords of the mesh points, in points
nodeX  = np.zeros((nI,nJ))*nan       # X coords of the nodes, in nodes
nodeY  = np.zeros((nI,nJ))*nan       # Y coords of the nodes, in nodes
dx_PE  = np.zeros((nI,nJ))*nan       # X distance to east node, in nodes
dx_WP  = np.zeros((nI,nJ))*nan       # X distance to west node, in nodes
dy_PN  = np.zeros((nI,nJ))*nan       # Y distance to north node, in nodes
dy_SP  = np.zeros((nI,nJ))*nan       # Y distance to south node, in nodes
dx_we  = np.zeros((nI,nJ))*nan       # X size of the control volume, in nodes
dy_sn  = np.zeros((nI,nJ))*nan       # Y size of the control volume, in nodes
fxe    = np.zeros((nI,nJ))*nan       # Interpolation factor, in nodes
fxw    = np.zeros((nI,nJ))*nan       # Interpolation factor, in nodes
fyn    = np.zeros((nI,nJ))*nan       # Interpolation factor, in nodes
fys    = np.zeros((nI,nJ))*nan       # Interpolation factor, in nodes
aE     = np.zeros((nI,nJ))*nan       # Array for east coefficient, in nodes
aW     = np.zeros((nI,nJ))*nan       # Array for west coefficient, in nodes
aN     = np.zeros((nI,nJ))*nan       # Array for north coefficient, in nodes
aS     = np.zeros((nI,nJ))*nan       # Array for south coefficient, in nodes
aP     = np.zeros((nI,nJ))*nan       # Array for central coefficient, in nodes
Su     = np.zeros((nI,nJ))*nan       # Array for source term for temperature, in nodes
Sp     = np.zeros((nI,nJ))*nan       # Array for source term for temperature, in nodes
T      = np.zeros((nI,nJ))*nan       # Array for temperature, in nodes
T_o    = np.zeros((nI,nJ))*nan       # Array for old temperature, in nodes
De     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for east face, in nodes
Dw     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for west face, in nodes
Dn     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for north face, in nodes
Ds     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for south face, in nodes
Fe     = np.zeros((nI,nJ))*nan       # Convective coefficients for east face, in nodes
Fw     = np.zeros((nI,nJ))*nan       # Convective coefficients for west face, in nodes
Fn     = np.zeros((nI,nJ))*nan       # Convective coefficients for north face, in nodes
Fs     = np.zeros((nI,nJ))*nan       # Convective coefficients for south face, in nodes
P      = np.zeros((nI,nJ))*nan       # Array for TDMA, in nodes
Q      = np.zeros((nI,nJ))*nan       # Array for TDMA, in nodes
u      = u_datavector.reshape(nI,nJ) # Values of x-velocity, in nodes
v      = v_datavector.reshape(nI,nJ) # Values of y-velocity, in nodes
res    = []                          # Array for appending residual each iteration
savedT = []                          # Array for saving T, for animated plot
probeValues = []                     # Array for saving probe values
# Set wall velocities to exactly zero:
u[abs(u) < 1e-9] = 0
v[abs(v) < 1e-9] = 0

# Create mesh - point coordinates
cF.createMesh(pointX, pointY,
              mI, mJ, pointXvector, pointYvector)

# Calculate length and height:
L = pointX[mI-1,0] - pointX[0,0]
H = pointY[0,mJ-1] - pointY[0,0]
# Scale probe locations with L and H
if unsteady:
    probeX*=L
    probeY*=H

# Calculate node positions
cF.calcNodePositions(nodeX, nodeY,
                     nI, nJ, pointX, pointY)

# Calculate distances once and keep
cF.calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                 nI, nJ, nodeX, nodeY, pointX, pointY)

# Calculate interpolation factors once and keep
cF.calcInterpolationFactors(fxe, fxw, fyn, fys,
                            nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                            nodeX, nodeY, pointX, pointY)

# Initialize dependent variable array
cF.initArray(T,
             T_init)

# Set Dirichlet boundary conditions
cF.setDirichletBCs(T,
                   nI, nJ, u, v, T_in, T_west, T_east, T_south, T_north)

# Calculate constant diffusive (D) coefficients
cF.calcD(De, Dw, Dn, Ds,
         gamma, nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn)

# Calculate constant convective (F) coefficients
cF.calcF(Fe, Fw, Fn, Fs,
         rho, nI, nJ, dx_we, dy_sn, fxe, fxw, fyn, fys, u, v)

# Add time loop
saveCounter = 0
for time in np.arange(deltaT, endTime + deltaT, deltaT):

    if unsteady:
        print('Time: ',time)
        saveCounter+=1
    
    # Set old T
    T_o = copy.deepcopy(T)
        
    # Calculate source terms
    cF.calcSourceTerms(Su, Sp,
                       nI, nJ, q_wall, Cp, u, v, dx_we, dy_sn, rho, deltaT, T_o, caseID)
              
    # Calculate coefficients for Hybrid scheme
    cF.calcHybridCoeffs(aE, aW, aN, aS, aP,
                        nI, nJ, De, Dw, Dn, Ds, Fe, Fw, Fn, Fs,
                        fxe, fxw, fyn, fys, dy_sn, Sp, u, v,
                        nodeX, nodeY, L, H, caseID)

    for explCorrIter in range(nExplCorrIter):

        # Solve for T using Gauss-Seidel
        if solver == 'GS':
            cF.solveGaussSeidel(T,
                                nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter)
        
        # Solve for T using TDMA
        if solver == 'TDMA':
            cF.solveTDMA(T, P, Q,
                          nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter)
    
        # Copy T to boundaries (and corners) where (non-)homegeneous Neumann is applied:
        cF.correctBoundaries(T,
                             nI, nJ, q_wall, k, dx_PE, dx_WP, dy_PN, dy_SP,
                             u, v, nodeX, nodeY, L, H, caseID)

        # Calculate normalized residuals
        cF.calcNormalizedResiduals(res,
                                    nI, nJ, explCorrIter, T,
                                    aP, aE, aW, aN, aS, Su, Sp)

        if res[-1]/res[0] < resTol:
            break

    if unsteady:
        probeValues.append(cF.probe(nodeX, nodeY, T,probeX, probeY))
    if unsteady and createAnimatedPlots and not saveCounter%saveInterval:
        savedT.append(T.copy())
    
#================ Plotting section ================
cF.createDefaultPlots(
                      nI, nJ, pointX, pointY, nodeX, nodeY,
                      dx_WP, dx_PE, dy_SP, dy_PN, Fe, Fw, Fn, Fs,
                      aE, aW, aN, aS, L, H, T, u, v, k,
                      explCorrIter, res, grid_type, caseID)
if unsteady:
    cF.createTimeEvolutionPlots(
                                probeX, probeY, probeValues, caseID, grid_type)
if unsteady and createAnimatedPlots:
    cF.createAnimatedPlots(
                          nodeX, nodeY, savedT)
cF.createAdditionalPlots()