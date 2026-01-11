# Chapter 6: Pressure-Velocity Coupling & SIMPLE Algorithm

## Exam Importance: 57.1% (4/7 exams)

---

## 1. Why Pressure-Velocity Coupling is Needed

### The Problem:
- Momentum equations contain pressure gradient
- Continuity equation does NOT contain pressure
- **No direct equation for pressure!**

### The Solution:
- Use continuity as constraint on velocity field
- Derive pressure correction equation
- Iterative coupling: SIMPLE, SIMPLER, PISO

---

## 2. Discretized Momentum Equation

### General Form:
```
a_P^u u_P = Σ a_nb^u u_nb + (p_w - p_e)·Δy + S_u
```

### In Detail:
```
a_P^u u_P = a_E^u u_E + a_W^u u_W + a_N^u u_N + a_S^u u_S + (p_I-1,J - p_I,J)·Δy
```

### Where:
- a_P^u = Σ a_nb^u - S_P^u
- S_P^u represents source term linearization
- Pressure term acts as a source

---

## 3. Physical Significance of S_P

### From Chapter 5:
```
S_P^u = -(F_e - F_w) - (F_n - F_s)
```

### Physical Meaning:
- **Net mass flux OUT of the control volume**
- If S_P < 0: acts as sink (good for stability)
- If S_P > 0: acts as source (can cause instability)

### Implementation for Stability:
```
Always calculate S_P to be NEGATIVE or ZERO

S_P = -|F_e - F_w| - |F_n - F_s|

Or linearize source terms: S = S_C + S_P·φ, where S_P < 0
```

---

## 4. Under-Relaxation

### Why Needed:
- Non-linear equations
- Prevent divergence
- Smooth convergence

### Explicit Under-Relaxation:
```
φ^new = φ^old + α·(φ^calculated - φ^old)
      = α·φ^calculated + (1-α)·φ^old

Where 0 < α < 1
```

### Implicit Under-Relaxation (BETTER!):

#### For u-momentum:
```
Standard equation:
a_P^u u_P = Σ a_nb^u u_nb + (p_w - p_e)·Δy

Under-relaxed:
(a_P^u/α) u_P = Σ a_nb^u u_nb + (p_w - p_e)·Δy + ((1-α)/α)·a_P^u·u_P^old

Rearranged:
a_P^u u_P = α[Σ a_nb^u u_nb + (p_w - p_e)·Δy] + (1-α)·a_P^u·u_P^old
```

#### New coefficient:
```
a_P^new = a_P^u / α

Additional source:
S_extra = ((1-α)/α)·a_P^u·u_P^old
```

### Why Implicit is Better:
- More stable
- Larger α possible
- Better convergence
- Couples with matrix solution

---

## 5. Meaning of Non-Linear Momentum Equations

### Sources of Non-Linearity:

#### 1. Convection Term:
```
∇·(ρuu) 

Non-linear because u appears twice: u·∇u
```

#### 2. Turbulent Viscosity:
```
μ_t = ρC_μ k²/ε

μ_t depends on k and ε, which depend on u
```

#### 3. Variable Properties:
```
ρ = ρ(p, T)
μ = μ(T)
```

### Consequences:
- **Iteration required** - cannot solve directly
- **Under-relaxation needed** for stability
- **Coupling** between equations

---

## 6. SIMPLE Algorithm (Semi-Implicit Method for Pressure-Linked Equations)

### Overview:
1. Guess pressure field p*
2. Solve momentum equations → u*, v*
3. Solve pressure correction p'
4. Correct velocities and pressure
5. Solve other equations (T, k, ε, etc.)
6. Check convergence, repeat if needed

### Detailed Steps:

#### Step 1: Solve Momentum with p*
```
a_P^u u_P* = Σ a_nb^u u_nb* + (p_w* - p_e*)·Δy
```

#### Step 2: Define Corrections
```
u = u* + u'
v = v* + v'
p = p* + p'

Where u', v', p' are corrections
```

#### Step 3: Subtract Momentum Equations
```
a_P^u u_P = Σ a_nb^u u_nb + (p_w - p_e)·Δy
a_P^u u_P* = Σ a_nb^u u_nb* + (p_w* - p_e*)·Δy

Subtract:
a_P^u u_P' = Σ a_nb^u u_nb' + (p_w' - p_e')·Δy
```

#### Step 4: SIMPLE Approximation
```
NEGLECT: Σ a_nb^u u_nb'

Therefore:
a_P^u u_P' = (p_w' - p_e')·Δy

u_P' = d_u(p_w' - p_e')

Where: d_u = Δy/a_P^u
```

#### Step 5: Substitute into Continuity
```
∇·(ρu) = 0

Discretized:
(ρu·A)_e - (ρu·A)_w + (ρv·A)_n - (ρv·A)_s = 0

Substitute u = u* + u':
(ρA)_e[u_e* + d_u(p_P' - p_E')·...] = 0
```

#### Step 6: Pressure Correction Equation
```
a_P^p p_P' = a_E^p p_E' + a_W^p p_W' + a_N^p p_N' + a_S^p p_S' + b^p

Where:
a_E^p = (ρd_u A)_e
a_W^p = (ρd_u A)_w
a_N^p = (ρd_v A)_n
a_S^p = (ρd_v A)_s
a_P^p = a_E^p + a_W^p + a_N^p + a_S^p

b^p = (ρu*A)_w - (ρu*A)_e + (ρv*A)_s - (ρv*A)_n  (mass imbalance)
```

#### Step 7: Correct Velocities and Pressure
```
u_P = u_P* + d_u(p_w' - p_e')
v_P = v_P* + d_v(p_s' - p_n')
p = p* + α_p·p'  (under-relaxed)

Where α_p < 1 (typically 0.2-0.3)
```

---

## 7. SIMPLE for Collocated Grids

### Problem with Collocated:
- All variables at cell centers
- Can get checkerboard pressure field
- Need interpolation for face velocities

### Rhie-Chow Interpolation:
```
u_e = (u_E + u_P)/2 + d_u[(p_w - p_e)_avg - (p_E - p_P)/δx]
```

### Pressure correction for collocated:
```
u_P' = d_u(p_w' - p_e')
v_P' = d_v(p_s' - p_n')

Where p_w', p_e' are face values interpolated from nodes
```

---

## 8. Staggered vs Collocated

### Staggered Grid:
- u on vertical faces
- v on horizontal faces
- p at cell centers
- Natural coupling
- No checkerboarding

### Collocated Grid:
- All variables at centers
- Simpler data structure
- Needs Rhie-Chow
- More flexible for complex geometry

---

## Common Exam Questions:

1. **Derive implicit under-relaxation** - Show new a_P and source term
2. **Physical significance of S_P** - Net mass flux, stability
3. **SIMPLE algorithm steps** - Complete procedure
4. **Pressure correction derivation** - Key approximation
5. **Meaning of non-linear** - Sources and consequences
6. **Collocated vs staggered** - Advantages and issues

---

## Key Formulas to Memorize:

```
# Momentum equation
a_P^u u_P = Σ a_nb^u u_nb + (p_w - p_e)·Δy

# Velocity correction
u_P' = d_u(p_w' - p_e')
d_u = Δy/a_P^u

# Pressure correction coefficients
a_E^p = (ρd_u A)_e
b^p = (ρu*A)_w - (ρu*A)_e + ... (mass imbalance)

# Under-relaxation (implicit)
a_P^new = a_P/α
S_extra = ((1-α)/α)·a_P·φ^old

# Corrections
u = u* + u'
p = p* + α_p·p'
```

---

## Tips for Exam:
- Know the SIMPLE algorithm steps by heart
- Understand the key approximation: neglecting Σ a_nb u_nb'
- Implicit under-relaxation derivation is common
- Remember: d_u = Δy/a_P^u (velocity correction factor)
- b^p is the mass imbalance (continuity residual)
- S_P should always be negative for stability
