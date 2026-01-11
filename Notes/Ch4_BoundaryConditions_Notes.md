# Chapter 4: Boundary Conditions & Finite Volume Method

## Exam Importance: 85.7% (6/7 exams) - CRITICAL CHAPTER!

---

## 1. Linear Interpolation & Interpolation Factors

### Standard Form:
```
φ_e = f_x·φ_E + (1 - f_x)·φ_P
```

### Derivation (MUST KNOW FOR EXAM):

#### Assumptions:
- Linear variation between nodes
- Face 'e' is between nodes P and E

#### Step 1: Similar triangles
```
(φ_e - φ_P)/(x_e - x_P) = (φ_E - φ_P)/(x_E - x_P)
```

#### Step 2: Solve for φ_e
```
φ_e - φ_P = (φ_E - φ_P)·(x_e - x_P)/(x_E - x_P)

φ_e = φ_P + (φ_E - φ_P)·(x_e - x_P)/(x_E - x_P)
```

#### Step 3: Rearrange to standard form
```
Let f_x = (x_e - x_P)/(x_E - x_P)

φ_e = φ_P + f_x(φ_E - φ_P)
    = φ_P(1 - f_x) + f_x·φ_E
    = f_x·φ_E + (1 - f_x)·φ_P
```

### Verification:
- If face 'e' is at node P (x_e = x_P): f_x = 0 → φ_e = φ_P ✓
- If face 'e' is at node E (x_e = x_E): f_x = 1 → φ_e = φ_E ✓
- For uniform mesh: f_x = 0.5 → φ_e = (φ_P + φ_E)/2 ✓

---

## 2. Neumann Boundary Conditions

### Definition:
```
∂φ/∂n = specified value
```

### Types:

#### Homogeneous Neumann (Most Common):
```
∂φ/∂n = 0  (zero flux, adiabatic wall, symmetry)
```

#### Non-Homogeneous Neumann:
```
∂φ/∂n = q_specified  (specified flux)
```

---

## 3. Implicit vs Explicit BC Implementation

### Definitions:

#### Explicit:
- Boundary value calculated **independently**
- Applied as a **known value**
- Does NOT affect coefficient matrix

#### Implicit:
- Boundary value **coupled** with interior nodes
- Contributes to **coefficient matrix**
- More stable and accurate

---

## 4. Thermally Insulated Boundary (Implicit Implementation)

### Most Common Exam Question!

### Boundary Condition:
```
∂T/∂n = 0  at boundary
```

### Implementation for West Boundary:

#### Method 1: Ghost Cell
```
T_ghost = T_P  (mirror temperature)

Flux at west face:
q_w = k(T_P - T_ghost)/δx = k(T_P - T_P)/δx = 0 ✓
```

#### Method 2: Zero Gradient Approximation
```
dT/dx|_w ≈ (T_P - T_W)/δx = 0

Therefore: T_W = T_P
```

#### Effect in Discretized Equation:
```
Standard internal node:
a_P T_P = a_E T_E + a_W T_W + b

At west boundary (T_W = T_P):
a_P T_P = a_E T_E + a_W T_P + b

Rearrange:
(a_P - a_W) T_P = a_E T_E + b

New a_P = a_P - a_W  (effectively removes west connection)
```

### Why Implicit is Better:
1. **More stable** - no explicit calculation error
2. **Automatically conservative** - no artificial sources/sinks
3. **Second-order accurate** - maintains discretization order

---

## 5. Convective Boundary Condition

### Physical Setup:
```
-k(∂T/∂n) = h(T - T_∞)

Where: h = heat transfer coefficient
       T_∞ = ambient temperature
```

### Implementation at East Boundary:

#### Discretize at face 'e':
```
-k(T_E - T_P)/δx = h(T_e - T_∞)

Assume T_e ≈ T_E (boundary temperature):
-k(T_E - T_P)/δx = h(T_E - T_∞)
```

#### Rearrange:
```
-k(T_E - T_P) = hδx(T_E - T_∞)
-kT_E + kT_P = hδxT_E - hδxT_∞
T_E(k + hδx) = kT_P + hδxT_∞

T_E = [kT_P + hδxT_∞]/(k + hδx)
```

#### In Discretized Equation:
```
Source term contribution:
S_C = hT_∞δyδz
S_P = -hδyδz

Modified east coefficient:
a_E_modified = a_E·k/(k + hδx)
```

---

## 6. Global Conservation & Heat Balance

### Most Important Concept!

### 1D Diffusion with Source:
```
d/dx(k dT/dx) + q = 0
```

### Boundary Conditions:
- West (x=0): T = 0
- East (x=L): ∂T/∂x = 0 (adiabatic)

### Global Balance Derivation:

#### Integrate from 0 to L:
```
∫₀ᴸ d/dx(k dT/dx)dx + ∫₀ᴸ q dx = 0

[k dT/dx]₀ᴸ + qL = 0

k(dT/dx)|_{x=L} - k(dT/dx)|_{x=0} + qL = 0
```

#### Apply BCs:
```
At x=L: dT/dx = 0 (adiabatic)
Therefore:
-k(dT/dx)|_{x=0} + qL = 0

Heat flux at west wall = Total heat generated
```

### Physical Interpretation:
- All heat generated (qL) must exit through west wall
- Global balance MUST be satisfied
- East wall has zero flux (insulated)

### Verification in Code:
```python
# Calculate total heat generated
Q_generated = q * (pointX[-1] - pointX[0])

# Calculate west wall flux
Q_west = -k * (T[0] - T_ghost)/dx

# Calculate east wall flux  
Q_east = -k * (T[N] - T[N-1])/dx

# Global balance check
globalCheck = Q_generated - (Q_east - Q_west)
# Should be ≈ 0
```

---

## 7. Discretization in Corner Cells (2D)

### Setup:
```
∂/∂x(k ∂T/∂x) + ∂/∂y(k ∂T/∂y) = 0
```

### Corner Cell (e.g., SW corner):
- West and South faces on boundary
- East and North faces interior

### Discretized Form:
```
East face flux: k(T_E - T_P)/δx · Δy
West face flux: (BC dependent)
North face flux: k(T_N - T_P)/δy · Δx
South face flux: (BC dependent)

Total:
a_P T_P = a_E T_E + a_N T_N + a_W T_W + a_S T_S + b

Where coefficients modified by BCs
```

---

## Common Exam Questions:

1. **Derive interpolation factor** - VERY COMMON! Show full derivation
2. **Implicit insulated boundary** - Implementation and why better than explicit
3. **Show global balance** - Heat flux = heat generated
4. **Convective BC** - Discretization and implementation
5. **Corner cell discretization** - How BCs modify coefficients

---

## Key Formulas to Memorize:

```
# Interpolation
f_x = (x_e - x_P)/(x_E - x_P)
φ_e = f_x·φ_E + (1 - f_x)·φ_P

# Insulated boundary (implicit)
T_boundary = T_P
a_P_new = a_P - a_boundary

# Convective BC
-k(∂T/∂n) = h(T - T_∞)

# Global balance
-k(dT/dx)|_wall = ∫q dx
```

---

## Tips for Exam:
- Chapter 4 appears in 85.7% of exams - HIGHEST PRIORITY
- Interpolation factor derivation is asked frequently - memorize steps
- Always explain "implicit means coupled with interior"
- For insulated BC: show T_ghost = T_P or equivalent
- Global balance: integrate governing equation over entire domain
- Verify formulas by checking limit cases
