# Chapter 11: Advanced Pressure-Velocity Coupling

## Exam Importance: 85.7% (6/7 exams) - CRITICAL CHAPTER!

---

## 1. Staggered vs Collocated Grids

### Staggered Grid:
```
- p stored at cell centers (i, j)
- u stored at vertical faces (i+1/2, j)
- v stored at horizontal faces (i, j+1/2)
```

#### Advantages:
- Natural coupling between p and velocities
- No checkerboard pressure
- Pressure gradient directly available at face

#### Disadvantages:
- Complex data structure
- Difficult for non-orthogonal meshes
- Multiple control volumes

### Collocated Grid:
```
- All variables (p, u, v, T, etc.) at cell centers (i, j)
```

#### Advantages:
- Simple data structure
- Easy for complex geometry
- Single control volume for all variables

#### Disadvantages:
- Checkerboard pressure problem
- Need Rhie-Chow interpolation

---

## 2. Checkerboard Pressure Problem

### What Is It?
```
Oscillating pressure field: high-low-high-low pattern
Physically unrealistic
```

### Why Does It Happen?

#### Pressure Gradient at Face:
```
On collocated grid:
(∂p/∂x)_e = (p_E - p_P)/δx

If p alternates: p_W = p_E ≠ p_P
Then: (p_E - p_P)/δx ≈ 0 even with large oscillations!
```

#### In Continuity:
```
∇·u = 0 satisfied cell-by-cell
But cannot detect checker pattern
```

### Solution: Rhie-Chow Interpolation

---

## 3. Rhie-Chow Interpolation

### Purpose:
- Prevent checkerboard pressure on collocated grids
- Couple pressure and velocity correctly at faces

### Standard Face Velocity:
```
u_e = (u_E + u_P)/2

Problem: doesn't account for pressure gradient difference
```

### Rhie-Chow Formula:
```
u_e = (u_E + u_P)/2 + d_u[(∇p)_avg - (∇p)_direct]

Where:
(∇p)_avg = [(∇p)_E + (∇p)_P]/2  (averaged from cell centers)
(∇p)_direct = (p_E - p_P)/δx  (direct gradient across face)
d_u = Δy/a_P^u  (velocity correction factor)
```

### Full Expression:
```
u_e = (u_E + u_P)/2 + 
      d_u{[(∂p/∂x)_E + (∂p/∂x)_P]/2 - (p_E - p_P)/δx}
```

### Why It Works:
- Adds pressure gradient correction
- Smooth pressure → (∇p)_avg ≈ (∇p)_direct → small correction
- Checkerboard → (∇p)_avg ≠ (∇p)_direct → large correction suppresses oscillation

---

## 4. Derivation of Face Velocity for Collocated

### Start with Momentum Equation:
```
a_P^u u_P = Σ a_nb^u u_nb + (p_w - p_e)·A

Solve for u_P:
u_P = (1/a_P^u)[Σ a_nb^u u_nb] + (1/a_P^u)(p_w - p_e)·A

u_P = u_P^H + d_u(p_w - p_e)

Where:
u_P^H = (1/a_P^u)[Σ a_nb^u u_nb]  (H = without pressure)
d_u = A/a_P^u
```

### Face Velocity:
```
Similar equation at face 'e':
u_e = u_e^H + d_e(p_w,e - p_e,e)

But p at faces not available on collocated grid!
```

### Interpolate:
```
u_e^H ≈ (u_P^H + u_E^H)/2

d_e ≈ (d_P + d_E)/2

Pressures at faces of face 'e':
p_w,e = p_P  (west of face 'e')
p_e,e = p_E  (east of face 'e')
```

### Result:
```
u_e = (u_P^H + u_E^H)/2 + d_e(p_P - p_E)

d_e = (d_P + d_E)/2 = A/(2)·(1/a_P^u + 1/a_E^u)
```

### Rhie-Chow Correction:
```
Add/subtract average pressure gradient:

u_e = (u_P + u_E)/2 + 
      d_e{[(p_w - p_e)_P + (p_w - p_e)_E]/2 - (p_P - p_E)}
```

---

## 5. Rhie-Chow at Boundary

### Inlet Boundary:
```
u_inlet specified directly
No interpolation needed
```

### Outlet Boundary:
```
Use one-sided interpolation:

u_outlet = u_P + d_P(p_w - p_e)_P

Or: u_outlet = u_P  (if fully developed)
```

### Wall Boundary:
```
u_wall = 0  (no-slip)

Or with wall functions:
u_wall from law of wall
```

### Symmetry:
```
u_n = 0  (normal component)
u_t from interior (tangential)
```

---

## 6. Velocity Correction on Collocated Grid

### SIMPLE for Collocated:

#### Momentum (starred values):
```
a_P^u u_P* = Σ a_nb^u u_nb* + (p_w* - p_e*)·A
```

#### Velocity Correction:
```
u = u* + u'
p = p* + p'
```

#### At Cell Centers:
```
a_P^u u_P' = Σ a_nb^u u_nb' + (p_w' - p_e')·A

SIMPLE approximation (neglect Σ a_nb u_nb'):
a_P^u u_P' ≈ (p_w' - p_e')·A

u_P' = d_u(p_w' - p_e')

Where: d_u = A/a_P^u
```

#### At Faces (for continuity):
```
Face velocity correction:
u_e' = d_e(p_w,e' - p_e,e')
     = d_e(p_P' - p_E')

Where: d_e = interpolated from d_P and d_E
```

---

## 7. Pressure Correction for Collocated

### Substitute into Continuity:
```
∇·(ρu) = 0

At face 'e':
(ρu)_e = ρ[u_e* + d_e(p_P' - p_E')]
```

### Discretized Continuity:
```
(ρA)_e[u_e* + d_e(p_P' - p_E')] - 
(ρA)_w[u_w* + d_w(p_W' - p_P')] + 
(ρA)_n[v_n* + d_n(p_P' - p_N')] - 
(ρA)_s[v_s* + d_s(p_S' - p_P')] = 0
```

### Rearrange to Pressure Correction Equation:
```
a_P^p p_P' = a_E^p p_E' + a_W^p p_W' + a_N^p p_N' + a_S^p p_S' + b^p

Where:
a_E^p = (ρd_e A)_e
a_W^p = (ρd_w A)_w
a_N^p = (ρd_n A)_n
a_S^p = (ρd_s A)_s
a_P^p = a_E^p + a_W^p + a_N^p + a_S^p

b^p = (ρu*A)_w - (ρu*A)_e + (ρv*A)_s - (ρv*A)_n  (mass imbalance)
```

---

## 8. Interpolation When Adding Pressure Gradient at Face

### Problem:
```
Need pressure gradient at face for momentum
On collocated grid: p only at cell centers
```

### Solution: Interpolate

#### Method 1: Direct Gradient
```
(∂p/∂x)_e = (p_E - p_P)/δx

Simple but causes checkerboarding
```

#### Method 2: Average Cell Gradients
```
(∂p/∂x)_e = [(∂p/∂x)_P + (∂p/∂x)_E]/2

Where (∂p/∂x)_P calculated using neighboring cells
```

#### Method 3: Rhie-Chow (BEST)
```
Already includes pressure gradient correction
Prevents checkerboarding
```

---

## 9. SIMPLER Algorithm

### Difference from SIMPLE:
```
SIMPLE: pressure correction updates p
SIMPLER: solve pressure equation directly
```

### Steps:
1. Guess u*, v*
2. Solve **pressure equation** (not correction) → p
3. Solve momentum with new p → u**, v**
4. Solve pressure correction → p'
5. Correct velocities: u = u** + u'
6. No pressure correction (p already correct)

### Advantages:
- Faster convergence
- Better for steady problems

### Disadvantages:
- More complex
- Two pressure equations per iteration

---

## 10. PISO Algorithm

### Predictor-Corrector:
1. Predictor: solve momentum → u*, v*
2. First corrector: p' correction
3. Second corrector: additional p' correction
4. Can use larger time steps

### Use:
- Unsteady flows
- Faster than SIMPLE for transient

---

## Common Exam Questions:

1. **Rhie-Chow interpolation** - VERY COMMON! Formula and why needed
2. **Derive face velocity** - From momentum equation
3. **Rhie-Chow at boundary** - Different boundary types
4. **Velocity correction collocated** - At cells vs faces
5. **Why checkerboard** - And how Rhie-Chow prevents it
6. **Interpolation adding pressure gradient** - Different methods

---

## Key Formulas to Memorize:

```
# Rhie-Chow interpolation
u_e = (u_E + u_P)/2 + d_e[(∇p)_avg - (∇p)_direct]

(∇p)_avg = [(∂p/∂x)_E + (∂p/∂x)_P]/2
(∇p)_direct = (p_E - p_P)/δx

# Velocity correction factor
d_u = A/a_P^u

# Velocity correction (collocated)
u_P' = d_u(p_w' - p_e')  (at cell center)
u_e' = d_e(p_P' - p_E')  (at face)

# Pressure correction coefficients
a_E^p = (ρd_e A)_e
b^p = (ρu*A)_w - (ρu*A)_e + ... (mass imbalance)
```

---

## Tips for Exam:
- Chapter 11 appears in 85.7% of exams - CRITICAL!
- Rhie-Chow is THE most important topic - memorize formula
- Understand why checkerboarding happens
- Know difference: staggered (natural) vs collocated (needs RC)
- Boundary conditions for Rhie-Chow asked frequently
- Face velocity derivation shows up multiple times
- d = A/a_P is the key velocity correction factor
