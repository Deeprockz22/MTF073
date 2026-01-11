# Chapter 9: Boundary Conditions for Navier-Stokes

## Exam Importance: 57.1% (4/7 exams)

---

## 1. Types of Flow Boundaries

### Common Boundaries in CFD:
1. **Inlet** - specified velocity, turbulence
2. **Outlet** - pressure specified or extrapolated
3. **Wall** - no-slip (u=0), wall functions
4. **Symmetry** - mirror conditions
5. **Periodic** - cyclic conditions

---

## 2. Fully Developed Flow Outlet

### Definition:
- Flow properties don't change in flow direction
- ∂φ/∂n = 0 for all variables (except pressure)

### Physical Meaning:
```
Fully developed: ∂u/∂x = 0, ∂v/∂x = 0, ∂T/∂x = 0, etc.

Flow pattern established, no further development
```

---

## 3. Outlet Boundary Conditions

### For Velocity:
```
∂u/∂n = 0  (zero gradient normal to boundary)

Implementation:
u_outlet = u_interior
v_outlet = v_interior
```

### Why This Works:
- Flow is leaving domain
- Information travels downstream
- No upstream influence from outlet

### Alternative (Convective BC):
```
∂φ/∂t + U_convective·∂φ/∂n = 0

Allows disturbances to leave domain smoothly
```

---

## 4. Pressure Correction at Outlet

### Boundary Condition:
```
p' = 0  at outlet

Or: ∂p'/∂n = 0
```

### Why p' = 0?

#### Reasoning:
1. Outlet is reference for pressure
2. Absolute pressure level arbitrary in incompressible flow
3. Setting p'_outlet = 0 fixes pressure level

#### In SIMPLE Algorithm:
```
Pressure correction equation:
∇·(d∇p') = ∇·(ρu*)  (continuity error)

At outlet:
p'_outlet = 0  → fixes one pressure level
Interior p' adjusted to satisfy continuity
```

### Why ∂p'/∂n = 0 Alternative:
```
If flow is truly fully developed:
- No pressure gradient normal to flow
- Natural boundary condition
- Allows pressure correction to float
```

---

## 5. Global Continuity Guarantee

### The Problem:
- Mass must be conserved globally
- Outlet mass flux = Inlet mass flux

### In SIMPLE Algorithm:

#### How It's Guaranteed:

1. **Pressure correction equation** derived from continuity:
```
∇·(ρu) = 0

Discretized:
(ρu·A)_e - (ρu·A)_w + (ρv·A)_n - (ρv·A)_s = 0

Source term in p' equation:
b^p = (ρu*·A)_w - (ρu*·A)_e + ...  (mass imbalance)
```

2. **Sum over all cells**:
```
Σ_cells b^p = Σ(inlet mass flux) - Σ(outlet mass flux)

If global continuity violated: b^p ≠ 0 globally
```

3. **Pressure correction adjusts velocities**:
```
u = u* + d_u(p_w' - p_e')

Correction ensures: Σ(ρu·A)_outlet = Σ(ρu·A)_inlet
```

#### When in SIMPLE:
- **Every iteration** of pressure correction
- p' equation solved → velocities corrected → continuity improved
- Converged solution: global continuity satisfied

---

## 6. Symmetry Boundary Conditions

### Physical Meaning:
- Flow is mirror image across symmetry plane
- No flow across plane
- No gradient normal to plane

### Conditions:

#### Velocity:
```
Normal component: u_n = 0  (no penetration)
Tangential components: ∂u_t/∂n = 0  (no gradient)
```

#### Scalars (T, k, ε, etc.):
```
∂φ/∂n = 0  (zero gradient)
```

#### Pressure:
```
∂p/∂n = 0
```

---

## 7. Symmetry BC for Different Mesh Orientations

### Vertical Symmetry Plane (x-constant):

```
u = 0  (normal component)
∂v/∂x = 0
∂w/∂x = 0
∂T/∂x = 0
∂p/∂x = 0
```

### Horizontal Symmetry Plane (y-constant):

```
v = 0  (normal component)
∂u/∂y = 0
∂w/∂y = 0
∂T/∂y = 0
∂p/∂y = 0
```

### Implementation:
```
Ghost cell approach:
u_ghost = -u_P  (normal component reflected)
v_ghost = v_P   (tangential component mirrored)
T_ghost = T_P   (scalar mirrored)
```

---

## 8. Properties of Symmetry

### For Velocity:
- **Normal component antisymmetric**: u_n(left) = -u_n(right) → u_n = 0 at plane
- **Tangential components symmetric**: u_t(left) = u_t(right)

### For Scalars:
- **Symmetric**: T(left) = T(right)
- **Zero gradient**: ∂T/∂n = 0

### Why Use Symmetry:
1. **Reduce computational domain** by 2×, 4×, 8×
2. **Faster solution**
3. **Less memory**
4. **Physically accurate** if geometry/flow is symmetric

---

## 9. Inlet Boundary Conditions

### Velocity Inlet:
```
u = u_specified
v = v_specified
T = T_inlet
k = k_inlet (if turbulent)
ε = ε_inlet
```

### Turbulence at Inlet:
```
k_inlet = 1.5(U_inlet·I)²
ε_inlet = C_μ^{3/4}·k^{3/2}/l_t

Where:
I = turbulence intensity (%)
l_t = turbulence length scale
```

### Pressure at Inlet:
```
Usually: ∂p/∂n = 0 (extrapolated from interior)

Or from momentum equation (if pressure inlet)
```

---

## 10. Wall Boundary Conditions

### No-Slip (Viscous):
```
u_wall = 0
v_wall = 0
w_wall = 0
```

### Temperature:
```
Specified: T_wall = T_specified
Or adiabatic: ∂T/∂n = 0
Or convective: -k(∂T/∂n) = h(T_wall - T_∞)
```

### Turbulence (with wall functions):
```
k_wall = 0
ε_wall from law of wall
u_P from law of wall: u+ = (1/κ)ln(y+) + B
```

---

## 11. Pressure Outlet

### Conditions:
```
p = p_specified (usually p_∞ or gauge pressure = 0)

∂u/∂n = 0
∂v/∂n = 0
∂T/∂n = 0
```

### Use When:
- Outlet pressure known
- Multiple inlets/outlets
- Natural convection

---

## 12. Boundary Condition Implementation

### Ghost Cell Method:
```
Create fictitious cells beyond boundary
Set ghost cell values to enforce BC

Example - Symmetry:
T_ghost = T_P  → zero gradient
u_ghost = -u_P → zero velocity at face
```

### Direct Specification:
```
Set boundary face value directly
Modify discretized equation coefficients

Example - Wall:
T_wall = T_specified
Becomes known value in source term
```

---

## Common Exam Questions:

1. **Fully developed outlet BC** - MOST COMMON! Velocity and pressure correction
2. **Global continuity guarantee** - How and when in SIMPLE
3. **Symmetry BC for velocity and scalar** - Different components
4. **Properties of symmetry** - Antisymmetric vs symmetric
5. **Pressure correction at outlet** - Why p' = 0 or ∂p'/∂n = 0
6. **Different mesh orientations** - How BCs change with plane orientation

---

## Key Concepts to Remember:

```
# Fully developed outlet
∂φ/∂n = 0  for all variables except p

# Pressure correction
p'_outlet = 0  (fixes pressure level)
Or: ∂p'/∂n = 0  (natural BC)

# Symmetry
Normal velocity: u_n = 0
Tangential velocity: ∂u_t/∂n = 0
Scalars: ∂φ/∂n = 0

# Global continuity
Σ(ρu·A)_outlet = Σ(ρu·A)_inlet
Enforced every iteration via p' equation

# Wall (no-slip)
u = v = w = 0
T specified, adiabatic, or convective
```

---

## Tips for Exam:
- Fully developed outlet appears in 3/7 recent exams - high priority!
- Always explain "fully developed means ∂/∂n = 0"
- For symmetry: distinguish normal vs tangential components
- Global continuity: explain it's enforced by pressure correction
- p' = 0 at outlet fixes the pressure level (arbitrary constant)
- Know difference between velocity inlet and pressure outlet
- Remember: information travels downstream (outlet BCs extrapolated)
