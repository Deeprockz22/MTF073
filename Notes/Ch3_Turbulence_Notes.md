# Chapter 3: Turbulence Modeling

## Exam Importance: 71.4% (5/7 exams)

---

## 1. Concept of Eddy Viscosity

### Basic Idea:
- Turbulence enhances mixing → acts like increased viscosity
- **Eddy viscosity (μ_t)** represents turbulent momentum transport

### Boussinesq Approximation:
```
τ_turbulent = -ρu'v' = μ_t(∂u/∂y)
```

### Key Points:
- μ_t is **NOT a fluid property** (unlike molecular viscosity μ)
- μ_t varies with flow conditions (position, velocity gradients)
- μ_t >> μ in turbulent flows

### Effective Viscosity:
```
μ_eff = μ + μ_t
```

---

## 2. k-ε Turbulence Model

### Two Transport Equations:

#### Turbulent Kinetic Energy (k):
```
∂(ρk)/∂t + ∇·(ρuk) = ∇·((μ + μ_t/σ_k)∇k) + P_k - ρε
```

#### Dissipation Rate (ε):
```
∂(ρε)/∂t + ∇·(ρuε) = ∇·((μ + μ_t/σ_ε)∇ε) + (C_1ε P_k - C_2ε ρε)ε/k
```

### Eddy Viscosity Formula:
```
μ_t = ρC_μ k²/ε
```

### Standard Constants:
- C_μ = 0.09
- C_1ε = 1.44
- C_2ε = 1.92
- σ_k = 1.0
- σ_ε = 1.3

### Production Term:
```
P_k = μ_t(∂u_i/∂x_j + ∂u_j/∂x_i)(∂u_i/∂x_j)
```

---

## 3. Wall Functions

### Why Wall Functions?
- Near-wall region has **very high gradients**
- Would need extremely fine mesh (y+ < 1)
- Wall functions **bridge** log-layer to first cell

### Law of the Wall:
```
u+ = u/u_τ
y+ = ρu_τy/μ

If y+ < 11.63 (viscous sublayer):
  u+ = y+

If y+ > 11.63 (log layer):
  u+ = (1/κ)ln(y+) + B
  
Where: κ = 0.41 (von Karman constant)
       B = 5.0
```

### Friction Velocity:
```
u_τ = √(τ_wall/ρ)
```

---

## 4. Source Terms for Wall Function BC (U-momentum)

### Most Common Exam Question!

### Setup:
- First cell center at distance y from wall
- Wall shear stress τ_wall acts on cell

### Source Term Derivation:

#### Wall Shear Stress:
```
τ_wall = ρu_τ² = μ_eff(∂u/∂y)|_wall
```

#### Using Law of Wall:
```
u_P = (u_τ/κ)ln(y_P+) + Bu_τ

Rearrange:
u_τ = κu_P / [ln(y_P+) + κB]
```

#### Source Term for Momentum Equation:
```
S_u = -τ_wall × A_wall / V_cell

Where A_wall is the wall face area
      V_cell is cell volume
```

#### Linearized Form:
```
S_u = S_C + S_P·u_P

Where:
S_C = 0
S_P = -ρu_τ²A_wall/(u_P·V_cell)
    = -ρκ²A_wall / [V_cell(ln(y+) + κB)²]
```

### Key Point:
- S_P must be **NEGATIVE** for stability
- Acts as implicit drag on the fluid

---

## 5. Turbulence Boundary Conditions

### At Walls:

#### k-ε Model:
```
k_wall = 0 (no production at wall)
ε_wall = calculated from wall function
u_wall = 0 (no-slip)
```

#### k-ω Model:
```
k_wall = 0
ω_wall = very large value
```

### At Inlet:
```
k_inlet = 1.5(U_inlet·I)²
ε_inlet = C_μ^(3/4) k^(3/2) / l_t

Where: I = turbulence intensity (typically 1-10%)
       l_t = turbulence length scale
```

### At Outlet:
```
∂k/∂n = 0 (zero gradient)
∂ε/∂n = 0
```

---

## 6. y+ Requirements

### For Wall Functions:
```
30 < y+ < 300  (ideal range)
```

### For Wall-Resolved:
```
y+ < 1  (very fine mesh needed)
```

### How to Control y+:
```
y+ = ρu_τΔy/μ

To achieve target y+:
Δy = y+·μ/(ρu_τ)

Where u_τ ≈ 0.05·U_∞ (rough estimate)
```

---

## Common Exam Questions:

1. **Explain eddy viscosity concept** - Enhanced mixing, Boussinesq approximation
2. **k-ε equations** - Write both transport equations
3. **Source terms for wall function BC** - Most frequent! Know linearization
4. **Law of the wall** - Know both viscous and log-layer formulas
5. **y+ requirements** - Different for wall functions vs resolved

---

## Key Formulas to Memorize:

```
μ_t = ρC_μ k²/ε

u+ = (1/κ)ln(y+) + B    [y+ > 11.63]

y+ = ρu_τy/μ

u_τ = √(τ_wall/ρ)

S_P = -ρκ²A_wall / [V_cell(ln(y+) + κB)²]
```

---

## Tips for Exam:
- Wall functions appear in 3/7 exams - HIGH PRIORITY
- Always show S_P is negative for stability
- Know the difference between μ (molecular) and μ_t (turbulent)
- Remember: y+ = 30-300 for wall functions
- κ = 0.41, B = 5.0, C_μ = 0.09 (standard values)
