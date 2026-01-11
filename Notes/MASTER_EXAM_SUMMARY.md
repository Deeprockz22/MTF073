# MTF073 CFD Exam Preparation - Master Summary

## 📚 Based on 7 Years of Exam Analysis (2019-2025)

---

## 🎯 PRIORITY TOPICS (Must Know!)

### TOP 3 CRITICAL CHAPTERS:
1. **Chapter 5 (114.3%)** - Convection Schemes - APPEARS MULTIPLE TIMES PER EXAM
2. **Chapter 4 (85.7%)** - Boundary Conditions & Finite Volume
3. **Chapter 8 (85.7%)** - Unsteady Flows & Time Discretization
4. **Chapter 11 (85.7%)** - Pressure-Velocity Coupling

---

## 🔥 MOST FREQUENTLY ASKED QUESTIONS

### 1. QUICK Scheme 3rd Order Proof (Ch5)
**Frequency: Very High**
- Use Taylor expansion at face
- Show truncation error is O(δx³)
- Formula: φ_e = 6/8·φ_P + 3/8·φ_E - 1/8·φ_W

### 2. Crank-Nicolson Discretization (Ch8)
**Frequency: Very High**
- 1D unsteady diffusion with source
- Show components in Ax = b
- Order of accuracy proof

### 3. Wall Function Source Terms (Ch3)
**Frequency: High (3/7 exams)**
- Derive S_u for u-momentum
- Show S_P < 0 for stability
- Use law of wall

### 4. Interpolation Factor Derivation (Ch4)
**Frequency: High**
- Linear interpolation: φ_e = f_x·φ_E + (1-f_x)·φ_P
- Full derivation from similar triangles
- Verify limit cases

### 5. Rhie-Chow Interpolation (Ch11)
**Frequency: High**
- Prevents checkerboard pressure
- Formula with pressure correction
- Boundary implementations

### 6. Fully Developed Outlet BC (Ch9)
**Frequency: High (3/7 recent exams)**
- Velocity: ∂u/∂n = 0
- Pressure correction: p' = 0 or ∂p'/∂n = 0
- Global continuity guarantee

### 7. Van Leer = 2nd Order Upwind (Ch5)
**Frequency: Medium-High**
- Constant gradient assumption
- Show schemes are identical

### 8. Implicit Under-Relaxation (Ch6)
**Frequency: Medium**
- Derive new a_P and source term
- Why better than explicit

---

## 📖 CHAPTER-BY-CHAPTER QUICK REFERENCE

### Chapter 2: Fundamentals (71.4%)
**Key Topics:**
- Eulerian approach (fixed CV)
- Physical role of ∇p (force per volume)
- Treatment of terms in transport equation
- Generic transport equation form

**Key Formula:**
```
∂(ρφ)/∂t + ∇·(ρuφ) = ∇·(Γ∇φ) + S
```

---

### Chapter 3: Turbulence (71.4%)
**Key Topics:**
- Eddy viscosity concept: μ_t = ρC_μ k²/ε
- k-ε model equations
- Wall functions & law of wall
- Source terms for wall BC ⭐

**Key Formulas:**
```
u+ = (1/κ)ln(y+) + B
y+ = ρu_τy/μ
S_P = -ρκ²A_wall/[V_cell(ln(y+)+κB)²]
```

---

### Chapter 4: Boundary Conditions (85.7%) ⭐
**Key Topics:**
- Interpolation factor derivation ⭐
- Implicit insulated boundary
- Convective BC
- Global heat balance ⭐
- Corner cell discretization

**Key Formulas:**
```
f_x = (x_e - x_P)/(x_E - x_P)
φ_e = f_x·φ_E + (1-f_x)·φ_P
-k(dT/dx)|_wall = qL
```

---

### Chapter 5: Convection Schemes (114.3%) ⭐⭐⭐
**Key Topics:**
- QUICK 3rd order proof ⭐⭐⭐
- Van Leer = 2nd upwind ⭐⭐
- Face flux conservation ⭐
- Scheme coefficients
- Peclet number formulation

**Key Formulas:**
```
Pe = F/D = ρuδx/Γ
QUICK: φ_e = 6/8·φ_P + 3/8·φ_E - 1/8·φ_W
Van Leer: φ_e = φ_P + (φ_E-φ_P)/(φ_E-φ_W)·(φ_P-φ_W)
2nd Upwind: φ_e = 3/2·φ_P - 1/2·φ_W
```

---

### Chapter 6: SIMPLE Algorithm (57.1%)
**Key Topics:**
- Implicit under-relaxation ⭐
- S_P physical significance
- Non-linear momentum meaning
- SIMPLE steps
- Velocity correction

**Key Formulas:**
```
a_P^new = a_P/α
S_extra = ((1-α)/α)·a_P·φ^old
u_P' = d_u(p_w' - p_e')
d_u = Δy/a_P^u
```

---

### Chapter 7: Solution Algorithms (57.1%)
**Key Topics:**
- TDMA forward elimination
- Matrix structure change ⭐
- 2D to 1D representation
- Direct vs iterative methods

**Key Formulas:**
```
Standard: a_i T_i = b_i T_{i+1} + c_i T_{i-1} + d_i
After FE: T_i = P_i T_{i+1} + Q_i

P_i = b_i/(a_i - c_i·P_{i-1})
Q_i = (d_i + c_i·Q_{i-1})/(a_i - c_i·P_{i-1})
```

---

### Chapter 8: Unsteady Flows (85.7%) ⭐⭐
**Key Topics:**
- Crank-Nicolson discretization ⭐⭐⭐
- Components in Ax = b ⭐⭐
- Order of accuracy proof ⭐
- Time term discretization
- 3D extension

**Key Formulas:**
```
a_P^0 = ρV/Δt  (always!)
a_E = Δt/2·D_e  (CN)
a_P = a_P^0 + a_E + a_W

b = [a_P^0-a_E-a_W]φ_P^n + a_Eφ_E^n + a_Wφ_W^n + Δt·q·V
```

---

### Chapter 9: NS Boundary Conditions (57.1%)
**Key Topics:**
- Fully developed outlet ⭐⭐
- Pressure correction at outlet ⭐
- Global continuity guarantee ⭐
- Symmetry BC properties
- Different mesh orientations

**Key Conditions:**
```
Outlet: ∂φ/∂n = 0, p' = 0
Symmetry: u_n = 0, ∂u_t/∂n = 0
Wall: u = 0
```

---

### Chapter 11: Advanced Pressure (85.7%) ⭐⭐
**Key Topics:**
- Rhie-Chow interpolation ⭐⭐⭐
- Derive face velocity ⭐⭐
- Rhie-Chow at boundary ⭐
- Staggered vs collocated
- Checkerboard prevention

**Key Formulas:**
```
u_e = (u_E+u_P)/2 + d_e[(∇p)_avg - (∇p)_direct]

d_u = A/a_P^u
u_P' = d_u(p_w'-p_e')  (cell)
u_e' = d_e(p_P'-p_E')  (face)
```

---

## 🎓 EXAM STRATEGY

### Time Allocation (4 hours):
- Read all questions: 10 min
- Derivation questions (Ch5, Ch8): 90 min
- Boundary conditions (Ch4, Ch9): 60 min
- Algorithm questions (Ch6, Ch11): 60 min
- Review: 20 min

### Point Distribution Pattern:
- 6-10 questions per exam
- 79-86 points total (post-2021)
- Typically 4-12 points per question

### Common Question Structures:
1. **Prove/Show** (Ch5) - Taylor expansion, algebraic manipulation
2. **Derive/Discretize** (Ch4, Ch8) - Integrate, apply FVM
3. **Explain physical meaning** (Ch2, Ch3, Ch6, Ch9) - Conceptual
4. **Implement BC** (Ch4, Ch9) - Show effect on equations

---

## 💡 LAST-MINUTE CHECKLIST

### Memorize These:
- [ ] QUICK formula: 6/8, 3/8, -1/8
- [ ] 2nd upwind: 3/2, -1/2
- [ ] Rhie-Chow full formula
- [ ] CN coefficients: a_P^0 = ρV/Δt, a_E = Δt/2·D_e
- [ ] d_u = A/a_P^u
- [ ] Law of wall: u+ = (1/κ)ln(y+) + B, κ=0.41, B=5.0
- [ ] Pe = ρuδx/Γ
- [ ] f_x = (x_e-x_P)/(x_E-x_P)

### Practice These Derivations:
- [ ] QUICK 3rd order (Taylor expansion)
- [ ] Interpolation factor (similar triangles)
- [ ] CN discretization (integrate over CV and time)
- [ ] Van Leer = 2nd upwind (constant gradient)
- [ ] Implicit under-relaxation (rearrange equation)

### Understand Concepts:
- [ ] Why Rhie-Chow prevents checkerboard
- [ ] Why S_P < 0 for stability
- [ ] How SIMPLE ensures global continuity
- [ ] Physical role of pressure gradient
- [ ] Eulerian vs Lagrangian

---

## 📊 EXAM TRENDS

### Increasing Topics (2023-2025):
- Rhie-Chow at boundaries
- Fully developed outlet conditions
- Coefficient matrix structure (TDMA)

### Consistent Topics (Every Year):
- Discretization schemes (Ch5)
- Boundary conditions (Ch4)
- Time stepping (Ch8)
- Pressure-velocity coupling (Ch11)

### 2026 Prediction:
- Expect: QUICK proof, CN discretization, Rhie-Chow
- Watch: Outlet BCs, matrix analysis, face velocity derivation
- Possible: More algorithm details, stability analysis

---

## Good Luck! 🍀

**Remember:**
- Read the question carefully - note what's GIVEN vs ASKED
- State ALL assumptions (constant ρ, uniform mesh, etc.)
- Show your work - partial credit is significant
- Verify your answer (limit cases, dimensional analysis)
- Neat presentation - make it easy to grade
