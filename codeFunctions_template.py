# This file should not be executed by itself. It only contains the
# functions needed for the main code. Some of the functions are
# pre-coded (marked "DO NOT CHANGE ANYTHING HERE!"), and some of
# the functions you need to implement yourself (marked "ADD CODE HERE").
# You can easily find those strings using Ctrl-f.
#
# MAKE SURE THAT YOU ONLY CHANGE ARRAYS IN THE FIRST ROW OF THE ARGUMENT LISTS!
# DO NOT CHANGE THE ARGUMENT LISTS OR FUNCTION NAMES!
# ... with the exception of function createAdditionalPlots, which is prepared
# for your additional plots and post-processing.
#
# Special notes for functions:
# * Functions generally only have access to the variables supplied as arguments
#   or local variables created in the function.
# * Arrays are "mutable", meaning that if they are supplied as arguments to a
#   function, any change to the array in the function also happens to the
#   original array used when calling the function. This is not the case for
#   scalars, which are "non-mutable". One way to do similar changes of global
#   scalars inside functions is to define them as "global" in the function.
#   This should be avoided if possible, since it is not good coding and can
#   potentially lead to problems.
# * Although an array is supplied as argument to a function, the following
#   creates a NEW local array rather than changing the supplied array:
#       aP = aE + aW + aN + aS - Sp
#   The correct way to change the array in a function is either by looping
#   over all the components, or using:
#       aP[:,:] = aE[:,:] + aW[:,:] + aN[:,:] + aS[:,:] - Sp[:,:]
# * Although an array is supplied as argument to a function, the following
#   creates a NEW local array rather than changing the supplied array:
#       p = p + pp*alphaP
#   The correct way to change the array in a function is either by looping
#   over all the components, or using:
#       p += pp*alphaP
#   or, to be more clear that some of the variables are arrays:
#       p[:,:] += pp[:,:]*alphaP
#   or, also working:
#       p[:,:] = p[:,:] + pp[:,:]*alphaP

# Packages needed
import numpy as np
import matplotlib.pyplot as plt
# Set default font size in plots:
plt.rcParams.update({'font.size': 12})
import math # Only used for mesh example 1
import os # For saving plots

def createEquidistantMesh(pointX, pointY,
                          mI, mJ, L, H):
    # =========================================================================
    # DO NOT CHANGE ANYTHING HERE!
    # =========================================================================
    # Functionality:
    #   Creates an equidistant mesh by calculating the coordinates of the mesh 
    #   points.
    # Logic flow:
    #   The function iterates through each mesh point (i, j) and calculates its
    #   x and y coordinates based on the domain size (L, H) and the number of 
    #   mesh points (mI, mJ).
    # Variable purpose:
    #   - pointX: 2D numpy array to store the x-coordinates of the mesh points.
    #   - pointY: 2D numpy array to store the y-coordinates of the mesh points.
    #   - mI: Number of mesh points in the X direction.
    #   - mJ: Number of mesh points in the Y direction.
    #   - L: Length of the domain in the X direction.
    #   - H: Height of the domain in the Y direction.
    # Input:
    #   - pointX: An empty 2D numpy array of size (mI, mJ).
    #   - pointY: An empty 2D numpy array of size (mI, mJ).
    #   - mI, mJ, L, H: Scalar values defining the mesh and domain size.
    # Output:
    #   - pointX: A 2D numpy array filled with the x-coordinates of the mesh 
    #     points.
    #   - pointY: A 2D numpy array filled with the y-coordinates of the mesh 
    #     points.
    # Error handling:
    #   - No explicit error handling.
    # Calculate mesh point coordinates:
    # Equation for line: yy = kk*xx + mm
    # Use it for yy as position in x or y direction and xx a i or j
    # We here always start at x=y=0, so (for x-direction):
    # x = kk*i
    # Determine kk from end points, as kk = (L-0)/(mI-1)
    # Same for y-direction
    for i in range(0, mI):
        for j in range(0, mJ):
            pointX[i,j] = i*L/(mI - 1)
            pointY[i,j] = j*H/(mJ - 1)

def createNonEquidistantMesh(pointX, pointY,
                             mI, mJ, L, H):
    # =========================================================================
    # ADD CODE HERE - ADAPT FOR YOUR CASE!
    # =========================================================================
    # Functionality:
    #   Creates a non-equidistant mesh. The examples provided show different
    #   methods for clustering mesh points in certain regions.
    # Logic flow:
    #   The user is expected to implement a method to generate a non-equidistant 
    #   mesh. Several examples are provided and commented out.
    # Variable purpose:
    #   - pointX: 2D numpy array to store the x-coordinates of the mesh points.
    #   - pointY: 2D numpy array to store the y-coordinates of the mesh points.
    #   - mI: Number of mesh points in the X direction.
    #   - mJ: Number of mesh points in the Y direction.
    #   - L: Length of the domain in the X direction.
    #   - H: Height of the domain in the Y direction.
    # Input:
    #   - pointX: An empty 2D numpy array of size (mI, mJ).
    #   - pointY: An empty 2D numpy array of size (mI, mJ).
    #   - mI, mJ, L, H: Scalar values defining the mesh and domain size.
    # Output:
    #   - pointX: A 2D numpy array filled with the x-coordinates of the mesh 
    #     points.
    #   - pointY: A 2D numpy array filled with the y-coordinates of the mesh 
    #     points.
    # Error handling:
    #   - No explicit error handling.
    # Below you find some examples of how to implement non-equidistant
    # meshes. None of them might be ideal, and they all have pros and cons.
    # Play with them and modify as you like (and need).
    # Toggle commenting of a set of lines by marking them and pressing Ctrl-1.
    ###########
    # Example 1
    ###########
    # Use a non-linear function that starts at 0 and ends at L or H
    # The shape of the function determines the distribution of points
    # We here use the cos function for two-sided clustering
    # for i in range(0, mI):
    #     for j in range(0, mJ):
    #         pointX[i,j] = -L*(np.cos(math.pi*(i/(mI-1)))-1)/2
    #         pointY[i,j] = -H*(np.cos(math.pi*(j/(mJ-1)))-1)/2
    ###############
    # Example 2
    ###############
    growing_rate = 1.2 # growing rate for non-equidistant mesh
    tangens_growing_rate=np.tanh(growing_rate)
    for i in range(0,mI):
        s=(i)/(mI-1)
        pointX[i,:]=(np.tanh(growing_rate*s)/tangens_growing_rate)*L
    for j in range(0,mJ):
        s=(2*(j+1)-mJ-1)/(mJ-1)
        pointY[:,j]=(1+np.tanh(growing_rate*s)/tangens_growing_rate)*0.5*H
    ###############
    # Example 3
    ###############
    # r = 0.85
    # dx = L*(1-r)/(1-r**(mI-1))
    # dy = H*(1-r)/(1-r**(mJ-1))
    # pointX[0,:] = 0
    # pointY[:,0] = 0
    # for i in range(1, mI):
    #     for j in range(mJ):
    #         pointX[i,j] = pointX[i-1, j] + (r**(i-1)) * dx
    # for j in range(1, mJ):
    #     for i in range(mI):
    #         pointY[i,j] = pointY[i, j-1] + (r**(j-1)) * dy
    ###############
    # Example 4
    ###############
    # procent_increase = 1.15
    # inc = 1 / procent_increase
    # dx_mid = 1 / mI
    # dx = np.zeros((mI, 1))
    # for i in range(0, mI):
    #     if i < (mI - 1) / 2 or i == (mI - 1) / 2:
    #         dx[i] = dx_mid * inc ** (mI - i)
    #         dx[-1 - i] = dx_mid * inc ** (mI - i)
    #     for j in range(0, mJ):
    #         pointY[i, j] = j * H / (mJ - 1)
    # for i, d_x in enumerate(dx):
    #     pointX[i, :] = np.sum(dx[0:i])
    # pointX = (pointX / pointX[-1, 0]) * L
    ###############
    # Example 5
    ###############
    # First and second value in linspace must add to 2 (any combination works)
    # dx = np.linspace(1.7, 0.3, mI + 1) * L / (mI - 1)
    # dy = np.linspace(0.3, 1.7, mJ + 1) * H / (mJ - 1)
    # pointX = np.zeros((mI, mJ))
    # pointY = np.zeros((mI, mJ))
    # for i in range(mI):
    #     for j in range(mJ):
    #         # For the mesh points
    #         if i > 0: pointX[i, j] = pointX[i - 1, j] + dx[i]
    #         if j > 0: pointY[i, j] = pointY[i, j - 1] + dy[j]    

def calcNodePositions(nodeX, nodeY,
                      nI, nJ, pointX, pointY):
    # =========================================================================
    # DO NOT CHANGE ANYTHING HERE!
    # =========================================================================
    # Functionality:
    #   Calculates the coordinates of the nodes based on the mesh point
    #   coordinates.
    # Logic flow:
    #   The function iterates through each node (i, j) and calculates its
    #   x and y coordinates. For internal nodes, the coordinate is the average
    #   of the two adjacent mesh points. For boundary nodes, the coordinate is
    #   the same as the corresponding mesh point.
    # Variable purpose:
    #   - nodeX: 2D numpy array to store the x-coordinates of the nodes.
    #   - nodeY: 2D numpy array to store the y-coordinates of the nodes.
    #   - nI: Number of nodes in the X direction.
    #   - nJ: Number of nodes in the Y direction.
    #   - pointX: 2D numpy array with the x-coordinates of the mesh points.
    #   - pointY: 2D numpy array with the y-coordinates of the mesh points.
    # Input:
    #   - nodeX: An empty 2D numpy array of size (nI, nJ).
    #   - nodeY: An empty 2D numpy array of size (nI, nJ).
    #   - nI, nJ: Scalar values defining the number of nodes.
    #   - pointX, pointY: 2D numpy arrays with the mesh point coordinates.
    # Output:
    #   - nodeX: A 2D numpy array filled with the x-coordinates of the nodes.
    #   - nodeY: A 2D numpy array filled with the y-coordinates of the nodes.
    # Error handling:
    #   - No explicit error handling.
    # Calculates node coordinates.
    # Same for equidistant and non-equidistant meshes.
    # Internal nodes:
    for i in range(0, nI):
        for j in range(0, nJ):
            if i > 0 and i < nI-1:
                nodeX[i,j] = 0.5*(pointX[i,0] + pointX[i-1,0])
            if j > 0 and j < nJ-1:
                nodeY[i,j] = 0.5*(pointY[0,j] + pointY[0,j-1])
    # Boundary nodes:
    nodeX[0,:]  = pointX[0,0]  # Note: corner points only needed for contour plot
    nodeY[:,0]  = pointY[0,0]  # Note: corner points only needed for contour plot
    nodeX[-1,:] = pointX[-1,0] # Note: corner points only needed for contour plot
    nodeY[:,-1] = pointY[0,-1] # Note: corner points only needed for contour plot
    
def calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                  nI, nJ, nodeX, nodeY, pointX, pointY):
    # =========================================================================
    # Functionality:
    #   Calculates the distances between nodes and mesh points.
    # Logic flow:
    #   The function iterates through each internal node (i, j) and calculates
    #   the distances to its neighbors (East, West, North, South) and the
    #   dimensions of the control volume.
    # Variable purpose:
    #   - dx_PE: 2D numpy array to store the distance between a node and its
    #     East neighbor.
    #   - dx_WP: 2D numpy array to store the distance between a node and its
    #     West neighbor.
    #   - dy_PN: 2D numpy array to store the distance between a node and its
    #     North neighbor.
    #   - dy_SP: 2D numpy array to store the distance between a node and its
    #     South neighbor.
    #   - dx_we: 2D numpy array to store the width of the control volume.
    #   - dy_sn: 2D numpy array to store the height of the control volume.
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - nodeX, nodeY: 2D numpy arrays with the node coordinates.
    #   - pointX, pointY: 2D numpy arrays with the mesh point coordinates.
    # Input:
    #   - All dx and dy arrays are empty 2D numpy arrays of size (nI, nJ).
    #   - nI, nJ, nodeX, nodeY, pointX, pointY: Scalar and 2D numpy arrays.
    # Output:
    #   - The dx and dy arrays are filled with the calculated distances.
    # Error handling:
    #   - No explicit error handling.
    # Calculate distances in first line of argument list.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            dx_PE[i,j] = nodeX[i+1,j] - nodeX[i,j]
            dx_WP[i,j] = nodeX[i,j] - nodeX[i-1,j]
            dy_PN[i,j] = nodeY[i,j+1] - nodeY[i,j]
            dy_SP[i,j] = nodeY[i,j] - nodeY[i,j-1]
            dx_we[i,j] = pointX[i,j] - pointX[i-1,j]
            dy_sn[i,j] = pointY[i,j] - pointY[i,j-1]

def calcInterpolationFactors(fxe, fxw, fyn, fys,
                             nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn, nodeX, pointX, nodeY, pointY):
    # =========================================================================
    # Functionality:
    #   Calculates the interpolation factors for the cell faces. These factors
    #   are used to interpolate values from the nodes to the cell faces.
    # Logic flow:
    #   The function iterates through each internal node (i, j) and calculates
    #   the interpolation factors for the east, west, north, and south faces
    #   of the control volume.
    # Variable purpose:
    #   - fxe, fxw, fyn, fys: 2D numpy arrays to store the interpolation
    #     factors for the east, west, north, and south faces.
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - dx_PE, dx_WP, dy_PN, dy_SP: 2D numpy arrays with the distances
    #     between nodes.
    #   - dx_we, dy_sn: 2D numpy arrays with the control volume dimensions.
    #   - nodeX, nodeY: 2D numpy arrays with the node coordinates.
    #   - pointX, pointY: 2D numpy arrays with the mesh point coordinates.
    # Input:
    #   - All f arrays are empty 2D numpy arrays of size (nI, nJ).
    #   - All other arguments are scalar or 2D numpy arrays with pre-calculated
    #     values.
    # Output:
    #   - The f arrays are filled with the calculated interpolation factors.
    # Error handling:
    #   - No explicit error handling.
    # Calculate interpolation factors in first row of argument list.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            fxe[i,j] = (pointX[i,j] - nodeX[i,j]) / dx_PE[i,j]
            fxw[i,j] = (nodeX[i,j] - pointX[i-1,j]) / dx_WP[i,j]
            fyn[i,j] = (pointY[i,j] - nodeY[i,j]) / dy_PN[i,j]
            fys[i,j] = (nodeY[i,j] - pointY[i,j-1]) / dy_SP[i,j]

def initArray(T):
    # =========================================================================
    # DO NOT CHANGE ANYTHING HERE!
    # =========================================================================
    # Functionality:
    #   Initializes the temperature array T with zeros.
    # Logic flow:
    #   The function sets all elements of the 2D numpy array T to 0.
    # Variable purpose:
    #   - T: 2D numpy array to store the temperature at each node.
    # Input:
    #   - T: An uninitialized 2D numpy array of size (nI, nJ).
    # Output:
    #   - T: A 2D numpy array with all elements set to 0.
    # Error handling:
    #   - No explicit error handling.
    # Initialize dependent variable array
    # Only change arrays in first row of argument list!
    # Note that a value is needed in all nodes for contour plot
    T[:,:] = 0

def setDirichletBCs(T,
                    nI, nJ, L, H, nodeX, nodeY, caseID):
    # =========================================================================
    # Functionality:
    #   Sets the Dirichlet boundary conditions for the temperature array T.
    # Logic flow:
    #   The function checks the caseID and applies the corresponding
    #   Dirichlet boundary conditions to the T array.
    # Variable purpose:
    #   - T: 2D numpy array to store the temperature at each node.
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - L, H: Length and height of the domain.
    #   - nodeX, nodeY: 2D numpy arrays with the node coordinates.
    #   - caseID: An integer identifying the case to be solved.
    # Input:
    #   - T: A 2D numpy array initialized with zeros.
    #   - nI, nJ, L, H, nodeX, nodeY, caseID: Scalar and 2D numpy arrays with
    #     pre-calculated values.
    # Output:
    #   - T: The temperature array with the Dirichlet boundary conditions
    #     applied.
    # Error handling:
    #   - No explicit error handling.
    # Set Dirichlet boundary conditions according to your case
    # Only change arrays in first row of argument list!
    # Note that a value is needed in all nodes for contour plot
    # Note: caseID is used only for testing.
    if caseID == 32:
        T[:, 0] = 10.0
        T[nI-1, :] = 20.0
        for j in range(nJ):
            T[0, j] = 10 * (1 + 2 * nodeY[j, 0] / H)
    else:
        pass

def updateConductivityArrays(k, k_e, k_w, k_n, k_s,
                             nI, nJ, nodeX, nodeY, fxe, fxw, fyn, fys, L, H, T, caseID):
    # =========================================================================
    # Functionality:
    #   Updates the conductivity arrays k, k_e, k_w, k_n, and k_s.
    # Logic flow:
    #   The function checks the caseID and applies the corresponding
    #   conductivity formula. The face conductivities are calculated using
    #   the harmonic mean of the nodal conductivities.
    # Variable purpose:
    #   - k: 2D numpy array to store the conductivity at each node.
    #   - k_e, k_w, k_n, k_s: 2D numpy arrays to store the conductivity at the
    #     east, west, north, and south faces of the control volume.
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - nodeX, nodeY: 2D numpy arrays with the node coordinates.
    #   - fxe, fxw, fyn, fys: 2D numpy arrays with the interpolation factors.
    #   - L, H: Length and height of the domain.
    #   - T: 2D numpy array with the temperature at each node.
    #   - caseID: An integer identifying the case to be solved.
    # Input:
    #   - k, k_e, k_w, k_n, k_s: Empty 2D numpy arrays.
    #   - All other arguments are scalar or 2D numpy arrays with pre-calculated
    #     values.
    # Output:
    #   - The conductivity arrays are filled with the calculated values.
    # Error handling:
    #   - No explicit error handling.
    # Update conductivity arrays according to your case
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # Note: caseID is used only for testing.
    if caseID == 32:
        for i in range(nI):
            for j in range(nJ):
                k[i,j] = 2 * (1 + 2 * T[i,j])
        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                k_e[i,j] = 2*k[i,j]*k[i+1,j]/(k[i,j]+k[i+1,j])
                k_w[i,j] = 2*k[i,j]*k[i-1,j]/(k[i,j]+k[i-1,j])
                k_n[i,j] = 2*k[i,j]*k[i,j+1]/(k[i,j]+k[i,j+1])
                k_s[i,j] = 2*k[i,j]*k[i,j-1]/(k[i,j]+k[i,j-1])
    else:
        pass

def updateSourceTerms(Su, Sp,
                      nI, nJ, dx_we, dy_sn, dx_WP, dx_PE, dy_SP, dy_PN, \
                      T, k_w, k_e, k_s, k_n, h, T_inf, caseID):
    # =========================================================================
    # Functionality:
    #   Updates the source term arrays Su and Sp.
    # Logic flow:
    #   The function checks the caseID and applies the corresponding
    #   source term formula. For case 32, the source term is hardcoded from
    #   the reference data.
    # Variable purpose:
    #   - Su: 2D numpy array for the explicit part of the source term.
    #   - Sp: 2D numpy array for the implicit part of the source term.
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - dx_we, dy_sn, dx_WP, dx_PE, dy_SP, dy_PN: 2D numpy arrays with the
    #     control volume dimensions and distances between nodes.
    #   - T: 2D numpy array with the temperature at each node.
    #   - k_w, k_e, k_s, k_n: 2D numpy arrays with the face conductivities.
    #   - h, T_inf: Convective heat transfer coefficient and ambient
    #     temperature.
    #   - caseID: An integer identifying the case to be solved.
    # Input:
    #   - Su, Sp: Empty 2D numpy arrays.
    #   - All other arguments are scalar or 2D numpy arrays with pre-calculated
    #     values.
    # Output:
    #   - The Su and Sp arrays are filled with the calculated values.
    # Error handling:
    #   - No explicit error handling.
    # Update source terms according to your case
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # Note: caseID is used only for testing.
    if caseID == 32:
        Su_uST = np.load('refData/Case_32_modArrays.npz')['Su_uST']
        Sp_uST = np.load('refData/Case_32_modArrays.npz')['Sp_uST']
        Su[:,:] = Su_uST[:,:]
        Sp[:,:] = Sp_uST[:,:]
    else:
        pass

def calcCoeffs(aE, aW, aN, aS, aP,
               nI, nJ, k_w, k_e, k_s, k_n,
               dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN, Sp, caseID):
    # =========================================================================
    # Functionality:
    #   Calculates the coefficients for the discretized heat conduction
    #   equation.
    # Logic flow:
    #   The function iterates through each internal node (i, j) and calculates
    #   the coefficients aE, aW, aN, aS, and aP based on the face
    #   conductivities and the control volume dimensions. It also handles the
    #   Neumann boundary condition at the north boundary.
    # Variable purpose:
    #   - aE, aW, aN, aS, aP: 2D numpy arrays for the coefficients of the
    #     discretized equation.
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - k_w, k_e, k_s, k_n: 2D numpy arrays with the face conductivities.
    #   - dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN: 2D numpy arrays with the
    #     control volume dimensions and distances between nodes.
    #   - Sp: 2D numpy array for the implicit part of the source term.
    #   - caseID: An integer identifying the case to be solved.
    # Input:
    #   - aE, aW, aN, aS, aP: Empty 2D numpy arrays.
    #   - All other arguments are scalar or 2D numpy arrays with pre-calculated
    #     values.
    # Output:
    #   - The coefficient arrays are filled with the calculated values.
    # Error handling:
    #   - No explicit error handling.
    # Calculate coefficients according to your case
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # Note: caseID is used only for testing.
    # Inner node neighbour coefficients:
    # (not caring about special treatment at boundaries):
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            aE[i,j] = k_e[i,j] * dy_sn[i,j] / dx_PE[i,j]
            aW[i,j] = k_w[i,j] * dy_sn[i,j] / dx_WP[i,j]
            aN[i,j] = k_n[i,j] * dx_we[i,j] / dy_PN[i,j]
            aS[i,j] = k_s[i,j] * dx_we[i,j] / dy_SP[i,j]
    # Modifications of aE and aW inside east and west boundaries:
    # ADD CODE HERE IF NECESSARY
    # Modifications of aN and aS inside north and south boundaries:
    for i in range(1, nI-1):
        aN[i, nJ-2] = 0

    # Inner node central coefficients:
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            aP[i,j] = aE[i,j] + aW[i,j] + aN[i,j] + aS[i,j] - Sp[i,j]

def solveGaussSeidel(phi,
                     nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter_phi):
    # =========================================================================
    # Functionality:
    #   Solves the linear system of equations using the Gauss-Seidel method.
    # Logic flow:
    #   The function iterates nLinSolIter_phi times. In each iteration, it
    #   updates the value of phi at each internal node based on the values of
    #   its neighbors.
    # Variable purpose:
    #   - phi: 2D numpy array with the variable to be solved (e.g.,
    #     temperature).
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - aE, aW, aN, aS, aP: 2D numpy arrays with the coefficients of the
    #     discretized equation.
    #   - Su: 2D numpy array for the explicit part of the source term.
    #   - nLinSolIter_phi: Number of Gauss-Seidel iterations.
    # Input:
    #   - phi: A 2D numpy array with the initial guess for the solution.
    #   - All other arguments are scalar or 2D numpy arrays with pre-calculated
    #     values.
    # Output:
    #   - phi: The solution of the linear system of equations.
    # Error handling:
    #   - No explicit error handling.
    # Implement the Gauss-Seidel solver for general variable phi,
    # so it can be reused for any variable.
    # Do it only in one direction.
    # Only change arrays in first row of argument list!
    for linSolIter in range(nLinSolIter_phi):   
        for i in range(1,nI-1):
            for j in range(1,nJ-1):
                phi[i,j] = (aE[i,j]*phi[i+1,j] + aW[i,j]*phi[i-1,j] + aN[i,j]*phi[i,j+1] + aS[i,j]*phi[i,j-1] + Su[i,j]) / aP[i,j]

def correctBoundaries(T,
                      nI, nJ, k_w, k_e, k_s, k_n,
                      dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN, 
                      h, T_inf, caseID):
    # =========================================================================
    # Functionality:
    #   Corrects the temperature at the boundaries where Neumann boundary
    #   conditions are applied.
    # Logic flow:
    #   The user is expected to implement the logic for the Neumann boundary
    #   conditions. For a homogeneous Neumann boundary condition (zero flux),
    #   the temperature at the boundary is set to the temperature of the
    #   adjacent internal node.
    # Variable purpose:
    #   - T: 2D numpy array with the temperature at each node.
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - k_w, k_e, k_s, k_n: 2D numpy arrays with the face conductivities.
    #   - dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN: 2D numpy arrays with the
    #     control volume dimensions and distances between nodes.
    #   - h, T_inf: Convective heat transfer coefficient and ambient
    #     temperature.
    #   - caseID: An integer identifying the case to be solved.
    # Input:
    #   - T: The temperature array after the Gauss-Seidel solver.
    #   - All other arguments are scalar or 2D numpy arrays with pre-calculated
    #     values.
    # Output:
    #   - T: The temperature array with the corrected boundary values.
    # Error handling:
    #   - No explicit error handling.
    # Copy T to boundaries (and corners) where homegeneous Neumann is applied
    # Only change arrays in first row of argument list!
    if caseID == 32:
        for i in range(1, nI - 1):
            T[i, nJ - 1] = T[i, nJ - 2]

def calcNormalizedResiduals(res, glob_imbal_plot,
                            nI, nJ, explCorrIter, T, \
                            aP, aE, aW, aN, aS, Su, Sp):
    # =========================================================================
    # Functionality:
    #   Calculates the normalized residual for the temperature equation.
    # Logic flow:
    #   The function calculates the non-normalized residual r0 by summing the
    #   absolute differences between the left and right hand sides of the
    #   discretized equation for all internal nodes. It then calculates the
    #   normalization factor F as the sum of all incoming heat fluxes.
    #   The normalized residual is r = r0 / F.
    # Variable purpose:
    #   - res: A list to store the residual at each iteration.
    #   - glob_imbal_plot: A list to store the global heat rate imbalance at
    #     each iteration.
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - explCorrIter: The current explicit correction iteration.
    #   - T: 2D numpy array with the temperature at each node.
    #   - aP, aE, aW, aN, aS: 2D numpy arrays with the coefficients of the
    #     discretized equation.
    #   - Su, Sp: 2D numpy arrays for the source terms.
    # Input:
    #   - All arguments are scalar, list, or 2D numpy arrays with
    #     pre-calculated values.
    # Output:
    #   - res: The list of residuals is appended with the residual of the
    #     current iteration.
    #   - glob_imbal_plot: The list of global heat rate imbalances is appended
    #     with the imbalance of the current iteration.
    # Error handling:
    #   - No explicit error handling.
    # Calculate and print normalized residuals, and sane 
    # Only change arrays in first row of argument list!
    # Normalize as shown in lecture notes, using:
    #   Din: Diffusive heat rate into the domain
    #   Dout: Diffusive heat rate out of the domain
    #   Sin: Source heat rate into the domain
    #   Sout: Source heat rate out of the domain
    # Non-normalized residual:
    r0 = 0
    Din = 0
    Sin = 0
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            r0 += abs(aP[i,j]*T[i,j] - (aE[i,j]*T[i+1,j] + aW[i,j]*T[i-1,j] + aN[i,j]*T[i,j+1] + aS[i,j]*T[i,j-1] + Su[i,j]))
            Din += aE[i,j]*max(0, T[i+1,j]) + aW[i,j]*max(0, T[i-1,j]) + aN[i,j]*max(0, T[i,j+1]) + aS[i,j]*max(0, T[i,j-1])
            Sin += max(0, Su[i,j])
    # Calculate normalization factor as
    # F =  Din + Sin
    F = Din + Sin
    # Calculate normalized residual:
    r = r0 / F
    # Append residual at present iteration to list of all residuals, for plotting:
    res.append(r)
    print('iteration: %5d, res = %.5e' % (explCorrIter, r))

    Dout = 0
    Sout = 0
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            Dout += aE[i,j]*max(0, -T[i+1,j]) + aW[i,j]*max(0, -T[i-1,j]) + aN[i,j]*max(0, -T[i,j+1]) + aS[i,j]*max(0, -T[i,j-1])
            Sout += max(0, -Sp[i,j]*T[i,j])

    # Calculate the global imbalance as
    glob_imbal = abs((Din - Dout + Sin - Sout)/(Din + Sin))
    glob_imbal_plot.append(glob_imbal)
    
def createDefaultPlots(
                       nI, nJ, pointX, pointY, nodeX, nodeY,
                       L, H, T, k,
                       explCorrIter, res, glob_imbal_plot, caseID):
    # =========================================================================
    # DO NOT CHANGE ANYTHING HERE!
    # =========================================================================
    # Functionality:
    #   Creates a set of default plots to visualize the results of the
    #   simulation.
    # Logic flow:
    #   The function creates a directory named "Figures" if it does not exist.
    #   Then, it generates and saves the following plots:
    #   - Computational mesh
    #   - Temperature distribution
    #   - Residual convergence
    #   - Heat flux vectors
    #   - Wall-normal heat flux
    #   - Global heat rate imbalance convergence
    # Variable purpose:
    #   - nI, nJ: Number of nodes in the X and Y directions.
    #   - pointX, pointY: 2D numpy arrays with the mesh point coordinates.
    #   - nodeX, nodeY: 2D numpy arrays with the node coordinates.
    #   - L, H: Length and height of the domain.
    #   - T: 2D numpy array with the temperature at each node.
    #   - k: 2D numpy array with the conductivity at each node.
    #   - explCorrIter: The current explicit correction iteration.
    #   - res: A list with the residual at each iteration.
    #   - glob_imbal_plot: A list with the global heat rate imbalance at each
    #     iteration.
    #   - caseID: An integer identifying the case to be solved.
    # Input:
    #   - All arguments are scalar, list, or 2D numpy arrays with
    #     pre-calculated values.
    # Output:
    #   - Several PNG files with the generated plots are saved in the "Figures"
    #     directory.
    # Error handling:
    #   - No explicit error handling.
    if not os.path.isdir('Figures'):
        os.makedirs('Figures')

    nan = float("nan")
    
    # Plot mesh
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Computational mesh \n (Corner nodes only needed for visualization)')
    plt.axis('equal')
    plt.vlines(pointX[:,0],0,H,colors = 'k',linestyles = 'dashed')
    plt.hlines(pointY[0,:],0,L,colors = 'k',linestyles = 'dashed')
    plt.plot(nodeX, nodeY, 'ro')
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_mesh.png')
    
    # Plot temperature contour
    plt.figure()
    plt.title('Temperature distribution')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature [K]')
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_temperatureDistribution.png')
    
    # Plot residual convergence
    plt.figure()
    plt.title('Residual convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Residual [-]')
    resLength = np.arange(0,len(res),1)
    plt.plot(resLength, res)
    plt.grid()
    plt.yscale('log')
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_residualConvergence.png')

    # Plot heat flux vectors in nodes (not at boundaries)
    qX = np.zeros((nI,nJ))*nan # Array for heat flux in x-direction, in nodes
    qY = np.zeros((nI,nJ))*nan # Array for heat flux in y-direction, in nodes
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
                qX[i,j] = -k[i,j] * (T[i+1,j] - T[i-1,j]) / (nodeX[i+1,j] - nodeX[i-1,j])
                qY[i,j] = -k[i,j] * (T[i,j+1] - T[i,j-1]) / (nodeY[i,j+1] - nodeY[i,j-1])
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Heat flux')
    plt.gca().set_aspect('equal', adjustable='box')
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature [K]')
    plt.quiver(nodeX, nodeY, qX, qY, color="black")
    plt.xlim(-0.2*L, 1.2*L)
    plt.ylim(-0.2*H, 1.2*H)
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_heatFlux.png')
    
    # Plot heat flux vectors NORMAL TO WALL boundary face centers ONLY (not in corners)
    # Use temperature gradient just inside domain (note difference to set heat flux)
    qX = np.zeros((nI,nJ))*nan # Array for heat flux in x-direction, in nodes
    qY = np.zeros((nI,nJ))*nan # Array for heat flux in y-direction, in nodes
    for j in range(1,nJ-1):
        qX[i,j] = -k[i,j] * (T[1,j] - T[0,j]) / (nodeX[1,j] - nodeX[0,j])
        qY[i,j] = 0
        i = nI-1
        qX[i,j] = -k[i,j] * (T[nI-1,j] - T[nI-2,j]) / (nodeX[nI-1,j] - nodeX[nI-2,j])
        qY[i,j] = 0
    for i in range(1,nI-1):
        qX[i,j] = 0
        qY[i,j] = -k[i,j] * (T[i,1] - T[i,0]) / (nodeY[i,1] - nodeY[i,0])
        j = nJ-1
        qX[i,j] = 0
        qY[i,j] = 0
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Wall-normal heat flux \n (from internal temperature gradient)')
    plt.gca().set_aspect('equal', adjustable='box')
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature [K]')
    plt.quiver(nodeX, nodeY, qX, qY, color="black")
    plt.xlim(-0.2*L, 1.2*L)
    plt.ylim(-0.2*H, 1.2*H)
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_wallHeatFlux.png')

    # Plot global heat rate imbalance convergence
    plt.figure()
    plt.title('Global heat rate imbalance convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Global heat rate imbalance [-]')
    glob_imbal_plotLength = np.arange(0,len(glob_imbal_plot),1)
    plt.plot(glob_imbal_plotLength, glob_imbal_plot)
    plt.grid()
    plt.yscale('log')
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_globalHeatRateImbalanceConvergence.png')

def createAdditionalPlots():
    # =========================================================================
    # Functionality:
    #   This function is intended for the user to create additional plots for
    #   post-processing and analysis.
    # Logic flow:
    #   The user can add any plotting code here. The function is called at the
    #   end of the main script.
    # Variable purpose:
    #   - The user can pass any necessary arrays as arguments to this function.
    # Input:
    #   - The user can define the input arguments as needed.
    # Output:
    #   - The user can generate and save any additional plots.
    # Error handling:
    #   - The user is responsible for any error handling.
    # ADD CODE HERE IF NECESSARY
    # Also add needed arguments to the function - and then also add those
    # arguments for the same function in the main code.
    # Don't change the values of any arrays supplied as arguments!
    pass # Comment this line when you have added your code!
