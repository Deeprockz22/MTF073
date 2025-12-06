# Packages needed
import numpy as np
import matplotlib.pyplot as plt
# Set default font size in plots:
plt.rcParams.update({'font.size': 12})
import os # For saving plots

def createMesh(pointX, pointY,
               mI, mJ, pointXvector, pointYvector):
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    # Only changes arrays in first row of argument list!
    # Sets point coordinates for Task 2 cases.
    for i in range(0, mI):
        for j in range(0, mJ):
            pointX[i,j] = pointXvector[i]
            pointY[i,j] = pointYvector[j]
    
def calcNodePositions(nodeX, nodeY,
                      nI, nJ, pointX, pointY):
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    # Only changes arrays in first row of argument list!
    # Calculates node coordinates.
    # Internal nodes:
    for i in range(0, nI):
        for j in range(0, nJ):
            if i > 0 and i < nI-1:
                nodeX[i,j] = 0.5*(pointX[i,0] + pointX[i-1,0])
            if j > 0 and j < nJ-1:
                nodeY[i,j] = 0.5*(pointY[0,j] + pointY[0,j-1])
    # Boundary nodes:
    nodeX[0,:]  = pointX[0,0]  # Note: corner points needed for contour plot
    nodeY[:,0]  = pointY[0,0]  # Note: corner points needed for contour plot
    nodeX[-1,:] = pointX[-1,0] # Note: corner points needed for contour plot
    nodeY[:,-1] = pointY[0,-1] # Note: corner points needed for contour plot

def calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                  nI, nJ, nodeX, nodeY, pointX, pointY):
    # Calculate distances in first line of argument list.
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            dx_PE[i,j] = nodeX[i+1, j] - nodeX[i, j]
            dx_WP[i,j] = nodeX[i, j] - nodeX[i-1, j]
            dy_PN[i,j] = nodeY[i, j+1] - nodeY[i, j]
            dy_SP[i,j] = nodeY[i, j] - nodeY[i, j-1]
            # Control volume dimensions (face to face)
            dx_we[i,j] = pointX[i, j] - pointX[i-1, j]
            dy_sn[i,j] = pointY[i, j] - pointY[i, j-1]

def calcInterpolationFactors(fxe, fxw, fyn, fys,
                             nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                             nodeX, nodeY, pointX, pointY):
    # Calculate interpolation factors in first row of argument list.
    # geometric interpolation factor: f_e = (x_e - x_P) / (x_E - x_P)
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            fxe[i,j] = (pointX[i,j] - nodeX[i,j]) / dx_PE[i,j]
            fxw[i,j] = (nodeX[i,j] - pointX[i-1,j]) / dx_WP[i,j]
            fyn[i,j] = (pointY[i,j] - nodeY[i,j]) / dy_PN[i,j]
            fys[i,j] = (nodeY[i,j] - pointY[0,j-1]) / dy_SP[i,j] 

def initArray(T,
              T_init):
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    # Sets initial default temperature
    T[:,:] = T_init

def setDirichletBCs(T,
                    nI, nJ, u, v, T_in, T_west, T_east, T_south, T_north):
    # Set Dirichlet boundary conditions
    # South and North Boundaries
    for i in range(nI):
        # North
        j = nJ-1
        if v[i,j] < 0:      # Velocity points down (Inlet)
            T[i,j] = T_in
        # South
        j = 0
        if v[i,j] > 0:      # Velocity points up (Inlet)
            T[i,j] = T_in
            
    # West and East Boundaries
    for j in range(nJ):
        # East
        i = nI-1
        if u[i,j] < 0:      # Velocity points left (Inlet)
            T[i,j] = T_in
        # West
        i = 0
        if u[i,j] > 0:      # Velocity points right (Inlet)
            T[i,j] = T_in
        elif u[i,j] == 0:   # Wall
            # Check Case 6 & 7 specifically: West Wall is Dirichlet 
            T[i,j] = T_west

def calcSourceTerms(Su, Sp,
                    nI, nJ, q_wall, Cp, u, v, dx_we, dy_sn, rho, deltaT, T_o, caseID):
    # Calculate constant source terms
    
    # 1. Reset terms
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            Su[i,j] = 0
            Sp[i,j] = 0
  
    # 2. Heat rate walls (q_wall is positive INTO domain):
    for i in range(1,nI-1):
        # North Wall (Boundary at j=nJ-1)
        j = nJ-2
        if u[i,nJ-1] == 0 and v[i,nJ-1] == 0:
            Su[i,j] += q_wall * dx_we[i,j] / Cp

        # South Wall (Boundary at j=0)
        j = 1
        if u[i,0] == 0 and v[i,0] == 0:
            Su[i,j] += q_wall * dx_we[i,j] / Cp

    for j in range(1,nJ-1):
        # East Wall (Boundary at i=nI-1)
        i = nI-2
        if u[nI-1,j] == 0 and v[nI-1,j] == 0:
            Su[i,j] += q_wall * dy_sn[i,j] / Cp

        # West Wall (Boundary at i=0)
        i = 1
        # Only add flux if it's NOT Case 6 or 7 (Dirichlet). 
        if caseID != 6 and caseID != 7:
            if u[0,j] == 0 and v[0,j] == 0:
                Su[i,j] += q_wall * dy_sn[i,j] / Cp

    # 3. Unsteady Term (Implicit Euler)
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            term = (rho * dx_we[i,j] * dy_sn[i,j]) / deltaT
            Su[i,j] += term * T_o[i,j]
            Sp[i,j] -= term

def calcD(De, Dw, Dn, Ds,
          gamma, nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn):
    # Calculate diffusions conductances
    for i in range (1,nI-1):
        for j in range(1,nJ-1):
            De[i,j] = gamma * dy_sn[i,j] / dx_PE[i,j]
            Dw[i,j] = gamma * dy_sn[i,j] / dx_WP[i,j]
            Dn[i,j] = gamma * dx_we[i,j] / dy_PN[i,j]
            Ds[i,j] = gamma * dx_we[i,j] / dy_SP[i,j]

def calcF(Fe, Fw, Fn, Fs,
          rho, nI, nJ, dx_we, dy_sn, fxe, fxw, fyn, fys, u, v):
    # Calculate constant convective (F) coefficients by linear interpolation
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            # Velocity interpolation
            ue = fxe[i,j] * u[i+1,j] + (1 - fxe[i,j]) * u[i,j]
            uw = fxw[i,j] * u[i-1,j] + (1 - fxw[i,j]) * u[i,j]
            vn = fyn[i,j] * v[i,j+1] + (1 - fyn[i,j]) * v[i,j]
            vs = fys[i,j] * v[i,j-1] + (1 - fys[i,j]) * v[i,j]
            
            # Mass flow rates
            Fe[i,j] = rho * ue * dy_sn[i,j]
            Fw[i,j] = rho * uw * dy_sn[i,j]
            Fn[i,j] = rho * vn * dx_we[i,j]
            Fs[i,j] = rho * vs * dx_we[i,j]

def calcHybridCoeffs(aE, aW, aN, aS, aP,
                     nI, nJ, De, Dw, Dn, Ds, Fe, Fw, Fn, Fs,
                     fxe, fxw, fyn, fys, dy_sn, Sp, u, v,
                     nodeX, nodeY, L, H, caseID):
    # 1. Calculate Standard Hybrid Coefficients for ALL interfaces
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            aE[i,j] = max(-Fe[i,j], (De[i,j] - Fe[i,j] * fxe[i,j]), 0.0)
            aW[i,j] = max(Fw[i,j],  (Dw[i,j] + Fw[i,j] * fxw[i,j]), 0.0)
            aN[i,j] = max(-Fn[i,j], (Dn[i,j] - Fn[i,j] * fyn[i,j]), 0.0)
            aS[i,j] = max(Fs[i,j],  (Ds[i,j] + Fs[i,j] * fys[i,j]), 0.0)

    # 2. Handle OUTLETS (Neumann Gradient=0) -> a_link = 0
    # East Outlet (u > 0)
    for j in range(1,nJ-1):
        i = nI-2
        if u[nI-1,j] > 0: aE[i,j] = 0.0
            
    # West Outlet (u < 0)
    for j in range(1,nJ-1):
        i = 1
        if u[0,j] < 0: aW[i,j] = 0.0

    # North Outlet (v > 0)
    for i in range(1,nI-1):
        j = nJ-2
        if v[i,nJ-1] > 0: aN[i,j] = 0.0

    # South Outlet (v < 0)
    for i in range(1,nI-1):
        j = 1
        if v[i,0] < 0: aS[i,j] = 0.0

    # 3. Handle ADIABATIC WALLS (Neumann Flux=0)
    # Zero out the coefficient connecting to the wall.
    
    # East Wall
    for j in range(1,nJ-1):
        i = nI-2
        if u[nI-1,j] == 0 and v[nI-1,j] == 0: aE[i,j] = 0.0
            
    # West Wall
    if caseID != 6 and caseID != 7: # Cases 6 & 7 West is Dirichlet. Do NOT zero coefficient.
        for j in range(1,nJ-1):
            i = 1
            if u[0,j] == 0 and v[0,j] == 0: aW[i,j] = 0.0
            
    # North Wall
    for i in range(1,nI-1):
        j = nJ-2
        if u[i,nJ-1] == 0 and v[i,nJ-1] == 0: aN[i,j] = 0.0
            
    # South Wall
    for i in range(1,nI-1):
        j = 1
        if u[i,0] == 0 and v[i,0] == 0: aS[i,j] = 0.0
    
    # 4. Calculate aP
    for i in range(1,nI-1):
        for j in range(1,nJ-1):       
            aP[i,j] = aE[i,j] + aW[i,j] + aN[i,j] + aS[i,j] - Sp[i,j]

def solveGaussSeidel(phi,
                     nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter):
    # Implement the Gauss-Seidel solver
    for linSolIter in range(nLinSolIter):   
        for i in range(1,nI-1):
            for j in range(1,nJ-1):
                numerator = aE[i,j]*phi[i+1,j] + aW[i,j]*phi[i-1,j] + \
                            aN[i,j]*phi[i,j+1] + aS[i,j]*phi[i,j-1] + Su[i,j]
                phi[i,j] = numerator / aP[i,j]

def solveTDMA(phi, P, Q,
              nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter):
    # Implement TDMA Solver
    for linSolIter in range(0,nLinSolIter):
        
        # --- SWEEP 1: West-East lines (varying i), Marching South-North (loop j) ---
        for j in range(1,nJ-1):
            # Forward Elimination
            term_others = aN[1,j]*phi[1,j+1] + aS[1,j]*phi[1,j-1] + Su[1,j]
            denom = aP[1,j] 
            
            P[1,j] = aE[1,j] / denom
            Q[1,j] = (term_others + aW[1,j]*phi[0,j]) / denom

            # Middle
            for i in range(2,nI-1):
                term_others = aN[i,j]*phi[i,j+1] + aS[i,j]*phi[i,j-1] + Su[i,j]
                denom = aP[i,j] - aW[i,j] * P[i-1,j]
                P[i,j] = aE[i,j] / denom
                Q[i,j] = (term_others + aW[i,j]*Q[i-1,j]) / denom
            
            # Backward Substitution
            for i in reversed(range(1,nI-1)):
                phi[i,j] = P[i,j] * phi[i+1,j] + Q[i,j]
""" 
        # --- SWEEP 2: South-North lines (varying j), Marching West-East (loop i) --- 
        for i in range(1,nI-1):
            # Forward Elimination (Start j=1)
            term_others = aE[i,1]*phi[i+1,1] + aW[i,1]*phi[i-1,1] + Su[i,1]
            denom = aP[i,1]
            
            P[i,1] = aN[i,1] / denom
            Q[i,1] = (term_others + aS[i,1]*phi[i,0]) / denom
            
            # Middle
            for j in range(2,nJ-1):
                term_others = aE[i,j]*phi[i+1,j] + aW[i,j]*phi[i-1,j] + Su[i,j]
                denom = aP[i,j] - aS[i,j] * P[i,j-1]
                P[i,j] = aN[i,j] / denom
                Q[i,j] = (term_others + aS[i,j]*Q[i,j-1]) / denom
            
            # Backward Substitution
            for j in reversed(range(1,nJ-1)):
                phi[i,j] = P[i,j] * phi[i,j+1] + Q[i,j]
"""
def correctBoundaries(T,
                      nI, nJ, q_wall, k, dx_PE, dx_WP, dy_PN, dy_SP,
                      u, v, nodeX, nodeY, L, H, caseID):
    # Updates Neumann boundaries (Gradient = 0 or Gradient = q/k)
    # Dirichlet boundaries are NOT updated here (they are fixed)

   # North Wall (j = nJ-1) - Adiabatic
    for i in range(1, nI-1):
        j = nJ-1
        if u[i,j] == 0 and v[i,j] == 0:
            T[i,j] = T[i,j-1] # Copy from internal

    # South Wall (j = 0) - Adiabatic
    for i in range(1, nI-1):
        j = 0
        if u[i,j] == 0 and v[i,j] == 0:
            T[i,j] = T[i,j+1] # Copy from internal

    # East Wall (i = nI-1) - Adiabatic
    for j in range(1, nJ-1):
        i = nI-1
        if u[i,j] == 0 and v[i,j] == 0:
            T[i,j] = T[i-1,j] # Copy from internal

    # West Wall (i = 0)
    # Case 6 & 7 West is Dirichlet. Do NOT update/overwrite it.
    if caseID != 6 and caseID != 7: 
        for j in range(1, nJ-1):
            i = 0
            if u[i,j] == 0 and v[i,j] == 0:
                T[i,j] = T[i+1,j]
    
    # Outlets (Homogeneous Neumann)
    # East Outlet
    for j in range(1, nJ-1):
        i = nI-1
        if u[i,j] > 0: T[i,j] = T[i-1,j]
            
    # West Outlet
    for j in range(1, nJ-1):
        i = 0
        if u[i,j] < 0: T[i,j] = T[i+1,j]

    # North Outlet
    for i in range(1, nI-1):
        j = nJ-1
        if v[i,j] > 0: T[i,j] = T[i,j-1]

    # South Outlet
    for i in range(1, nI-1):
        j = 0
        if v[i,j] < 0: T[i,j] = T[i,j+1]

    # Set cornerpoint values to average of neighbouring boundary points
    T[0,0]   = 0.5*(T[1,0]+T[0,1])     # DO NOT CHANGE
    T[-1,0]  = 0.5*(T[-2,0]+T[-1,1])   # DO NOT CHANGE
    T[0,-1]  = 0.5*(T[1,-1]+T[0,-2])   # DO NOT CHANGE
    T[-1,-1] = 0.5*(T[-2,-1]+T[-1,-2]) # DO NOT CHANGE

def calcNormalizedResiduals(res,
                            nI, nJ, explCorrIter, T,
                            aP, aE, aW, aN, aS, Su, Sp):
    # Compute and print residuals (taking into account normalization):
    # Non-normalized residual:
    r0 = 0.0 # Initialize sum
    
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            # Calculate the Right Hand Side (RHS) terms
            RHS = aE[i,j]*T[i+1,j] + aW[i,j]*T[i-1,j] + \
                  aN[i,j]*T[i,j+1] + aS[i,j]*T[i,j-1] + Su[i,j]
            
            # Add the absolute difference (imbalance) to the total sum
            r0 += abs(aP[i,j]*T[i,j] - RHS)

    # Append residual at present iteration to list of all residuals, for plotting:
    res.append(r0)
    
    print('iteration: %5d, res = %.5e' % (explCorrIter, res[-1]/res[0]))
def probe(nodeX, nodeY, T, probeX, probeY, method='linear'):
    # method (str): interpolation method ('linear', 'nearest', 'cubic')
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    from scipy.interpolate import griddata
    points = np.column_stack((nodeX.ravel(), nodeY.ravel()))
    values = T.ravel()
    probes = np.column_stack((probeX, probeY))
    probe = griddata(points, values, probes, method=method)
    return probe

def createDefaultPlots(
                       nI, nJ, pointX, pointY, nodeX, nodeY,
                       dx_WP, dx_PE, dy_SP, dy_PN, Fe, Fw, Fn, Fs,
                       aE, aW, aN, aS, L, H, T, u, v, k,
                       explCorrIter, res, grid_type, caseID):
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    if not os.path.isdir('Figures'):
        os.makedirs('Figures')

    nan = float("nan")
    
    # Plot mesh
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Computational mesh \n (Corner nodes only needed for visualization)')
    plt.axis('equal')
    plt.vlines(pointX[:,0],pointY[0,0],pointY[0,-1],colors = 'k',linestyles = 'dashed')
    plt.hlines(pointY[0,:],pointX[0,0],pointX[-1,0],colors = 'k',linestyles = 'dashed')
    plt.plot(nodeX, nodeY, 'ro')
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_mesh.png')
    
    # Plot velocity vectors
    plt.figure()
    plt.quiver(nodeX.T, nodeY.T, u.T, v.T)
    plt.title('Velocity vectors')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_velocityVectors.png')
    
    # Plot temperature contour
    plt.figure()
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature $[K]$')
    plt.title('Temperature $[K]$')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_temperatureDistribution.png')
    
    # Plot heat flux vectors
    qX = np.zeros((nI,nJ))*nan 
    qY = np.zeros((nI,nJ))*nan 
    for j in range(1,nJ-1):
        i = 0
        if u[i,j] == 0 and v[i,j] == 0:
            dist = nodeX[i+1,j] - nodeX[i,j]
            qX[i,j] = -k * (T[i+1,j] - T[i,j]) / dist 
            qY[i,j] = 0
        i = nI-1
        if u[i,j] == 0 and v[i,j] == 0:
            dist = nodeX[i,j] - nodeX[i-1,j]
            qX[i,j] = -k * (T[i,j] - T[i-1,j]) / dist 
            qY[i,j] = 0
    for i in range(1,nI-1):
        j = 0
        if u[i,j] == 0 and v[i,j] == 0:
            dist = nodeY[i,j+1] - nodeY[i,j]
            qX[i,j] = 0
            qY[i,j] = -k * (T[i,j+1] - T[i,j]) / dist 
        j = nJ-1
        if u[i,j] == 0 and v[i,j] == 0:
            qX[i,j] = 0
            qY[i,j] = -k * (T[i,j] - T[i,j-1]) / dist 
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Wall-normal heat flux vectors')
    plt.gca().set_aspect('equal', adjustable='box')
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature $[K]$')
    plt.quiver(nodeX, nodeY, qX, qY, color="black")
    plt.xlim(-0.5*L, 3/2*L)
    plt.ylim(-0.5*H, 3/2*H)
    plt.tight_layout()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_wallHeatFlux.png')
    
    # Plot residual convergence
    plt.figure()
    plt.title('Residual convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Residual [-]')
    resLength = np.arange(0,len(res),1)
    normalized = [x / res[0] for x in res]
    plt.plot(resLength, normalized)
    plt.grid()
    plt.yscale('log')
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_residualConvergence.png')  

def createTimeEvolutionPlots(probeX, probeY, probeValues, caseID, grid_type):
    data = np.vstack(probeValues) 
    n_steps, n_probes = data.shape
    plt.figure()
    for i in range(n_probes):
        plt.plot(range(1, n_steps+1), data[:, i], label=f'Probe {i+1} ({probeX[i]:.2f}, {probeY[i]:.2f})')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature [K]')
    plt.title('Evolution of Probe Values Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_timeEvolution.png')

def createAnimatedPlots(nodeX, nodeY, savedT):
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots()
    vmin = min(arr.min() for arr in savedT)
    vmax = max(arr.max() for arr in savedT)
    tempmap = ax.contourf(nodeX.T, nodeY.T, savedT[0].T,
                          cmap='coolwarm', levels=30, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(tempmap, ax=ax)
    cbar.set_label('Temperature [K]')
    ax.set_title('Temperature [K]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    fig.tight_layout()
    def update(frame):
        ax.clear() 
        ax.contourf(nodeX.T, nodeY.T, savedT[frame].T,
                    cmap='coolwarm', levels=30, vmin=vmin, vmax=vmax)
        ax.set_title(f'Temperature [K] - Frame {frame}')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal')
        return []
    ani = FuncAnimation(fig, update, frames=len(savedT), interval=100,
                        blit=False, repeat=True)
    ani.save('animated_contour.gif', writer='pillow')

def createAdditionalPlots():
    pass