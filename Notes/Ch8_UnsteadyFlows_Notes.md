# Chapter 8: Unsteady Flows & Time Discretization

## Exam Importance: 85.7% (6/7 exams) - CRITICAL CHAPTER!

---

## 1. Generic Unsteady Equation

### Standard Form:
```
∂(ρφ)/∂t + ∇·(ρuφ) = ∇·(Γ∇φ) + S
  [Time]   [Convection]  [Diffusion]  [Source]
```

### 1D Unsteady Diffusion:
```
∂(ρφ)/∂t = ∂/∂x(Γ ∂φ/∂x) + q

Special case:
- No convection
- Source term q can be φ-dependent or constant
```

---

## 2. Time Discretization Schemes

### Notation:
```
φ^n = value at time level n (current time t)
φ^{n+1} = value at time level n+1 (new time t + Δt)
```

### Three Main Schemes:

#### 1. Forward Euler (Explicit):
```
(φ^{n+1} - φ^n)/Δt = f(φ^n)

Time derivative: use φ^n for RHS
```

#### 2. Backward Euler (Implicit):
```
(φ^{n+1} - φ^n)/Δt = f(φ^{n+1})

Time derivative: use φ^{n+1} for RHS
```

#### 3. Crank-Nicolson (Semi-Implicit):
```
(φ^{n+1} - φ^n)/Δt = [f(φ^n) + f(φ^{n+1})]/2

Time derivative: average of old and new time levels
θ = 0.5
```

---

## 3. Discretization of Time Term

### 1D Unsteady Diffusion:
```
∂(ρφ)/∂t = ∂/∂x(Γ ∂φ/∂x) + q
```

### Integrate Over Control Volume and Time Step:

#### Volume Integration:
```
∫_V ∂(ρφ)/∂t dV = ∫_V ∂/∂x(Γ ∂φ/∂x) dV + ∫_V q dV
```

#### Time Integration (Δt = t^{n+1} - t^n):
```
∫_{t^n}^{t^{n+1}} [∫_V ∂(ρφ)/∂t dV] dt = ∫_{t^n}^{t^{n+1}} [RHS] dt
```

#### Left Side:
```
∫_{t^n}^{t^{n+1}} ∫_V ∂(ρφ)/∂t dV dt = V(ρφ|^{n+1} - ρφ|^n)

For constant ρ and V:
= ρV(φ_P^{n+1} - φ_P^n)
```

#### Right Side (Crank-Nicolson):
```
∫_{t^n}^{t^{n+1}} RHS dt ≈ Δt/2 [RHS^n + RHS^{n+1}]
```

---

## 4. Complete Crank-Nicolson Discretization

### 1D Unsteady Diffusion with Source:
```
∂(ρφ)/∂t = ∂/∂x(Γ ∂φ/∂x) + q
```

### Assumptions:
- Constant ρ, Γ
- Equidistant mesh (δx uniform)
- q = constant (spatially varying but time-independent)

### Discretized Form:

#### Time term:
```
ρV(φ_P^{n+1} - φ_P^n)/Δt
```

#### Diffusion term (Crank-Nicolson):
```
Δt/2 [D_e(φ_E^n - φ_P^n) - D_w(φ_P^n - φ_W^n)]  +
Δt/2 [D_e(φ_E^{n+1} - φ_P^{n+1}) - D_w(φ_P^{n+1} - φ_W^{n+1})]

Where: D = Γ·A/δx
```

#### Source term:
```
Δt·q·V
```

### Rearranged to Standard Form:
```
a_P φ_P^{n+1} = a_E φ_E^{n+1} + a_W φ_W^{n+1} + b

Where:
a_E = Δt/2 · D_e
a_W = Δt/2 · D_w
a_P^0 = ρV/Δt  (time term coefficient)
a_P = a_P^0 + a_E + a_W

b = a_P^0 φ_P^n + Δt/2[D_e(φ_E^n - φ_P^n) - D_w(φ_P^n - φ_W^n)] + Δt·q·V
  = [a_P^0 - a_E - a_W]φ_P^n + a_E φ_E^n + a_W φ_W^n + Δt·q·V
```

### Compact Form:
```
a_P φ_P^{n+1} = a_E φ_E^{n+1} + a_W φ_W^{n+1} + 
                a_E φ_E^n + a_W φ_W^n + 
                [a_P^0 - a_E - a_W]φ_P^n + Δt·q·V

Components:
- New time level (implicit): a_P φ_P^{n+1} = a_E φ_E^{n+1} + a_W φ_W^{n+1} + ...
- Old time level (explicit): ... + a_E φ_E^n + a_W φ_W^n + [a_P^0 - a_E - a_W]φ_P^n
- Source: ... + Δt·q·V
```

---

## 5. System Matrix Components (Ax = b)

### For Crank-Nicolson:

#### Matrix A (left side, implicit part):
```
Contains: a_P, a_E, a_W for time level n+1

Diagonal: a_P = ρV/Δt + Δt/2(D_e + D_w)
Off-diagonals: -Δt/2·D_e, -Δt/2·D_w
```

#### Vector b (right side, explicit part + source):
```
Contains: contributions from time level n

b_P = [ρV/Δt - Δt/2(D_e + D_w)]φ_P^n + 
      Δt/2·D_e·φ_E^n + 
      Δt/2·D_w·φ_W^n + 
      Δt·q·V
```

---

## 6. Order of Accuracy Analysis

### Using Taylor Expansion:

#### Forward Difference (Forward Euler):
```
∂φ/∂t ≈ (φ^{n+1} - φ^n)/Δt

Taylor expansion:
φ^{n+1} = φ^n + Δt·∂φ/∂t + (Δt²/2)·∂²φ/∂t² + O(Δt³)

Therefore:
(φ^{n+1} - φ^n)/Δt = ∂φ/∂t + (Δt/2)·∂²φ/∂t² + O(Δt²)

Truncation error: O(Δt) → FIRST ORDER
```

#### Backward Difference (Backward Euler):
```
∂φ/∂t ≈ (φ^{n+1} - φ^n)/Δt

Taylor expansion (backward):
φ^n = φ^{n+1} - Δt·∂φ/∂t + (Δt²/2)·∂²φ/∂t² + O(Δt³)

Therefore:
(φ^{n+1} - φ^n)/Δt = ∂φ/∂t - (Δt/2)·∂²φ/∂t² + O(Δt²)

Truncation error: O(Δt) → FIRST ORDER
```

#### Central Difference (Crank-Nicolson):
```
∂φ/∂t ≈ (φ^{n+1} - φ^n)/Δt

Average RHS at n and n+1:
f(φ) ≈ [f(φ^n) + f(φ^{n+1})]/2

Taylor expansions:
φ^{n+1} = φ^n + Δt·∂φ/∂t + (Δt²/2)·∂²φ/∂t² + (Δt³/6)·∂³φ/∂t³ + ...
φ^{n-1} = φ^n - Δt·∂φ/∂t + (Δt²/2)·∂²φ/∂t² - (Δt³/6)·∂³φ/∂t³ + ...

Add:
φ^{n+1} + φ^{n-1} = 2φ^n + Δt²·∂²φ/∂t² + O(Δt⁴)

Therefore:
∂²φ/∂t² = (φ^{n+1} - 2φ^n + φ^{n-1})/Δt² + O(Δt²)

For Crank-Nicolson averaging:
Truncation error: O(Δt²) → SECOND ORDER
```

---

## 7. Stability Analysis

### Forward Euler (Explicit):
```
Stability criterion:
Δt ≤ ρ(δx)²/(2Γ)  (1D diffusion)

Conditional stability - restrictive time step!
```

### Backward Euler (Implicit):
```
Unconditionally stable for any Δt

Can use large time steps
```

### Crank-Nicolson (Semi-Implicit):
```
Unconditionally stable

BUT: can produce oscillations if Δt too large
Best compromise: 2nd order + stable
```

---

## 8. Implementation Details

### Initialization:
```
At t = 0: specify φ^0 everywhere
```

### Time Marching:
```
For each time step:
1. Assemble coefficient matrix using φ^n
2. Solve for φ^{n+1}
3. Update: φ^n = φ^{n+1}
4. Advance to next time step
```

### Transient vs Steady:
```
Transient: continue until t_final
Steady: continue until |φ^{n+1} - φ^n| < tolerance
```

---

## 9. 3D Extension

### 3D Unsteady Diffusion:
```
∂(ρφ)/∂t = ∂/∂x(Γ ∂φ/∂x) + ∂/∂y(Γ ∂φ/∂y) + ∂/∂z(Γ ∂φ/∂z) + q
```

### Discretized (Crank-Nicolson):
```
a_P φ_P^{n+1} = a_E φ_E^{n+1} + a_W φ_W^{n+1} + 
                a_N φ_N^{n+1} + a_S φ_S^{n+1} + 
                a_T φ_T^{n+1} + a_B φ_B^{n+1} + 
                [contributions from old time level] + source

Where:
a_E = Δt/2 · D_e = Δt/2 · (Γ·A_e/δx_e)
a_P^0 = ρV/Δt
a_P = a_P^0 + Σ a_nb
```

---

## Common Exam Questions:

1. **Discretize 1D unsteady with CN** - MOST COMMON! Full derivation
2. **Identify components in Ax = b** - What goes where
3. **Order of accuracy** - Taylor expansion proof
4. **Discretization of time term** - Integration over volume and time
5. **Stability comparison** - Explicit vs implicit schemes
6. **3D cell discretization** - Extension to 3D

---

## Key Formulas to Memorize:

```
# Time term (always)
a_P^0 = ρV/Δt
Contribution: (ρV/Δt)(φ^{n+1} - φ^n)

# Crank-Nicolson
a_E = Δt/2 · D_e
a_W = Δt/2 · D_w
a_P = a_P^0 + a_E + a_W

b = [a_P^0 - a_E - a_W]φ_P^n + a_E φ_E^n + a_W φ_W^n + Δt·q·V

# Forward Euler: O(Δt) - 1st order
# Backward Euler: O(Δt) - 1st order  
# Crank-Nicolson: O(Δt²) - 2nd order

# Stability
Forward Euler: Δt ≤ ρ(δx)²/(2Γ)
Backward Euler: unconditionally stable
Crank-Nicolson: unconditionally stable
```

---

## Tips for Exam:
- Chapter 8 appears in 85.7% of exams - CRITICAL!
- Crank-Nicolson discretization is THE most asked question
- Know where each term goes in Ax = b system
- Time term ALWAYS contributes to diagonal: ρV/Δt
- Taylor expansion for order of accuracy - practice this!
- Remember: θ = 0 (explicit), θ = 1 (implicit), θ = 0.5 (CN)
- Show all assumptions: constant ρ, Γ, uniform mesh, etc.
