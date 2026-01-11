# Chapter 2: Fundamentals of CFD

## Exam Importance: 71.4% (5/7 exams)

---

## 1. Eulerian Approach

### What it means:
- **Fixed control volumes** in space
- Fluid flows **through** the control volumes
- Opposite of Lagrangian (following fluid particles)

### Key Concept:
- Control volumes are stationary
- We observe properties as fluid passes through
- Used in the Finite Volume Method (FVM)

---

## 2. Physical Role of Pressure Gradient

### In Momentum Equations:
```
∇p = pressure gradient term
```

### Physical Significance:
1. **Drives fluid motion** - creates acceleration
2. Acts as a **force per unit volume**
3. Couples velocity components together
4. Responsible for pressure-driven flows

### In FVM:
- Pressure gradient acts on control volume **faces**
- Creates net force: (p_west - p_east) × Area
- Must be carefully interpolated at faces

---

## 3. Treatment of Terms in Governing Equations

### Standard Form:
```
∂(ρφ)/∂t + ∇·(ρuφ) = ∇·(Γ∇φ) + S
  [Time]    [Convection]   [Diffusion]  [Source]
```

### How Each Term is Treated:

#### Time Term:
- Forward Euler (explicit)
- Backward Euler (implicit)
- Crank-Nicolson (semi-implicit)

#### Convection Term:
- Upwind schemes
- Central differencing
- Higher-order schemes (QUICK, Van Leer)

#### Diffusion Term:
- Always central differencing
- Second-order accurate

#### Source Term:
- Linearized: S = S_C + S_P·φ
- S_P should be **negative** for stability

---

## 4. Similarity of Governing Equations

### Generic Transport Equation:
```
∂(ρφ)/∂t + ∇·(ρuφ) = ∇·(Γ∇φ) + S
```

### What Changes for Different φ:
| Variable φ | Γ (Diffusion Coeff) | S (Source Term) |
|------------|---------------------|-----------------|
| u-velocity | μ | -∂p/∂x + Su |
| v-velocity | μ | -∂p/∂y + Sv |
| Temperature| k/cp | q̇ |
| Species | ρD | R |
| Turbulence k | μ_t/σ_k | P_k - ρε |

### Why This Matters:
- **One solver** can handle all equations
- Same discretization approach for all variables
- Only coefficients change

---

## 5. Derivation of FVM from Governing Equations

### Steps:
1. **Integrate** governing equation over control volume
2. **Apply divergence theorem** (Gauss theorem)
3. **Convert volume integrals** to surface integrals
4. **Discretize** face fluxes and source terms

### Example - 1D Diffusion:
```
Governing: d/dx(Γ dφ/dx) = 0

Integrate over CV:
∫[d/dx(Γ dφ/dx)]dV = 0

Apply divergence theorem:
Γ_e(dφ/dx)_e - Γ_w(dφ/dx)_w = 0

Discretize:
Γ_e(φ_E - φ_P)/δx_e - Γ_w(φ_P - φ_W)/δx_w = 0
```

---

## Key Formulas to Remember:

### Conservation Principle:
```
Rate of change + Net convection = Net diffusion + Generation
```

### Peclet Number:
```
Pe = (Convection strength)/(Diffusion strength) = ρuL/Γ
```

### Continuity Equation:
```
∂ρ/∂t + ∇·(ρu) = 0
```

---

## Common Exam Questions:

1. **Define Eulerian approach** - Fixed CV, fluid flows through
2. **Physical role of ∇p** - Force per volume, drives flow
3. **How does ∇p act in FVM** - Through face pressures, creates net force
4. **Generic transport equation** - Know all terms and their treatment
5. **Derive FVM discretization** - Integrate → Divergence theorem → Discretize

---

## Tips for Exam:
- Always mention "control volume" when discussing FVM
- State that diffusion uses central differencing
- Remember: S_P < 0 for stability
- Know the difference between Eulerian (fixed) and Lagrangian (moving)
