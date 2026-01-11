# Chapter 5: Convection Schemes

## Exam Importance: 114.3% (8/7 exams) - MOST CRITICAL CHAPTER!

---

## 1. Overview of Discretization Schemes

### Generic Convection-Diffusion:
```
Convective flux: F = ρu
Diffusive flux: D = Γ/δx
Peclet number: Pe = F/D = ρuδx/Γ
```

### Scheme Comparison:

| Scheme | Order | Bounded | Conservative | Pe Limit |
|--------|-------|---------|--------------|----------|
| Central | 2nd | No | Yes | |Pe| < 2 |
| Upwind (1st) | 1st | Yes | Yes | All Pe |
| Upwind (2nd) | 2nd | No | Yes | All Pe |
| Hybrid | 1st | Yes | Yes | All Pe |
| QUICK | 3rd | No | Yes | All Pe |
| Van Leer | 2nd | Yes | Yes | All Pe |

---

## 2. Central Differencing Scheme (CDS)

### Face Value:
```
φ_e = (φ_P + φ_E)/2
```

### Net Flux at East Face:
```
q_e = F_e·φ_e - D_e(φ_E - φ_P)
    = F_e(φ_P + φ_E)/2 - D_e(φ_E - φ_P)
    = (F_e/2)φ_P + (F_e/2)φ_E - D_e·φ_E + D_e·φ_P
    = (F_e/2 + D_e)φ_P + (F_e/2 - D_e)φ_E
```

### Coefficients:
```
a_E = D_e - F_e/2
a_W = D_w + F_w/2
a_P = a_E + a_W + (F_e - F_w)
```

### Stability Requirement:
```
|Pe| = |F/D| < 2

Physical meaning: 
- Diffusion must dominate
- Fails for convection-dominated flows
```

---

## 3. First-Order Upwind Scheme

### Face Value:
```
If F_e > 0 (flow left to right):
  φ_e = φ_P

If F_e < 0 (flow right to left):
  φ_e = φ_E
```

### Coefficients:
```
a_E = D_e + max(0, -F_e)
a_W = D_w + max(0, F_w)
a_P = a_E + a_W + (F_e - F_w)
```

### Properties:
- **Unconditionally stable** (all Pe)
- **Bounded** (no over/undershoots)
- **Only 1st order** - numerical diffusion
- **Conservative**

---

## 4. Second-Order Upwind Scheme

### Face Value (F_e > 0):
```
φ_e = 3/2·φ_P - 1/2·φ_W
```

### Using Three Points:
- Upstream neighbor (W)
- Central node (P)  
- Downstream node (E)

### Properties:
- **2nd order accurate**
- NOT bounded (can overshoot)
- Less diffusive than 1st order upwind

---

## 5. QUICK Scheme (Quadratic Upstream Interpolation)

### Face Value (F_e > 0):
```
φ_e = 6/8·φ_P + 3/8·φ_E - 1/8·φ_W
```

### Alternative Form:
```
φ_e = φ_P + 1/8·[(φ_E - φ_P) - (φ_P - φ_W)]
```

### Proof of 3rd Order Accuracy:

#### Taylor Expansions at face 'e':
```
φ_P = φ_e - δx·φ'_e + (δx²/2)·φ''_e - (δx³/6)·φ'''_e + O(δx⁴)

φ_E = φ_e + δx·φ'_e + (δx²/2)·φ''_e + (δx³/6)·φ'''_e + O(δx⁴)

φ_W = φ_e - 2δx·φ'_e + (4δx²/2)·φ''_e - (8δx³/6)·φ'''_e + O(δx⁴)
```

#### Substitute into QUICK:
```
6/8·φ_P = 6/8·[φ_e - δx·φ'_e + (δx²/2)·φ''_e - (δx³/6)·φ'''_e]
3/8·φ_E = 3/8·[φ_e + δx·φ'_e + (δx²/2)·φ''_e + (δx³/6)·φ'''_e]
-1/8·φ_W = -1/8·[φ_e - 2δx·φ'_e + 2δx²·φ''_e - (4δx³/3)·φ'''_e]

Sum:
φ_e_QUICK = φ_e + O(δx³)  → 3rd order!
```

---

## 6. Van Leer Scheme (Flux Limiter)

### Formula (F_e > 0):
```
φ_e = φ_P + (φ_E - φ_P)/(φ_E - φ_W)·(φ_P - φ_W)   if |φ_E - 2φ_P + φ_W| ≤ |φ_E - φ_W|

φ_e = φ_P                                           otherwise
```

### Properties:
- **2nd order** in smooth regions
- **Bounded** (TVD scheme)
- **Reduces to 1st order** near discontinuities

---

## 7. Van Leer vs 2nd Order Upwind (IMPORTANT!)

### When are they identical?

#### Assumption: ∂φ/∂x ≈ constant

#### Van Leer formula:
```
φ_e = φ_P + (φ_E - φ_P)/(φ_E - φ_W)·(φ_P - φ_W)
```

#### If gradient is constant:
```
(φ_E - φ_P)/δx = (φ_P - φ_W)/δx = constant

Therefore:
φ_E - φ_P = φ_P - φ_W

Substitute:
φ_e = φ_P + (φ_P - φ_W)/(φ_E - φ_W)·(φ_P - φ_W)
    = φ_P + (φ_P - φ_W)²/(φ_E - φ_W)
```

#### For uniform mesh with constant gradient:
```
φ_E - φ_W = 2(φ_P - φ_W)

Therefore:
φ_e = φ_P + (φ_P - φ_W)²/(2(φ_P - φ_W))
    = φ_P + (φ_P - φ_W)/2
    = 3/2·φ_P - 1/2·φ_W  ← 2nd order upwind!
```

---

## 8. Face Flux Conservation

### Important Principle:
**Face flux must be same from both sides!**

### Proof for Central Differencing:

#### From node i (east face):
```
q_e^i = (F_e/2)(T_i + T_{i+1}) - D_e(T_{i+1} - T_i)
```

#### From node i+1 (west face):
```
q_w^{i+1} = (F_w/2)(T_i + T_{i+1}) - D_w(T_{i+1} - T_i)
```

#### Since e (of i) = w (of i+1):
```
F_e = F_w
D_e = D_w

Therefore:
q_e^i = q_w^{i+1} ✓
```

### Why This Matters:
- Ensures **global conservation**
- No artificial sources/sinks at faces
- Fundamental FVM property

---

## 9. Contribution to Coefficient Matrix

### Central Differencing:
```
a_E = D_e - F_e/2
a_W = D_w + F_w/2
```

### First-Order Upwind:
```
a_E = D_e + max(0, -F_e)
a_W = D_w + max(0, F_w)
```

### Second-Order Upwind (F > 0):
```
From: φ_e = 3/2·φ_P - 1/2·φ_W

Flux: F_e(3/2·φ_P - 1/2·φ_W)

Coefficients:
a_P contribution: +3/2·F_e
a_W contribution: -1/2·F_e
```

### QUICK (F > 0):
```
From: φ_e = 6/8·φ_P + 3/8·φ_E - 1/8·φ_W

Coefficients:
a_P contribution: +6/8·F_e
a_E contribution: +3/8·F_e  
a_W contribution: -1/8·F_e
```

---

## 10. Peclet Number & Hybrid Scheme

### Peclet Number:
```
Pe = F/D = (ρu)/(Γ/δx) = ρuδx/Γ
```

### Physical Meaning:
- Pe >> 1: Convection dominates
- Pe << 1: Diffusion dominates
- |Pe| = 2: Stability limit for CDS

### Hybrid Scheme:
```
If |Pe| < 2:
  Use Central Differencing (2nd order)
  
If |Pe| ≥ 2:
  Use First-Order Upwind (stable, bounded)
```

### Net Flux with Peclet:
```
q_w = F_w[(1/2)(1 + 2/Pe_w)·φ_W + (1/2)(1 - 2/Pe_w)·φ_P]

Where Pe_w = F_w/D_w
```

---

## 11. Bounded and Conservative Schemes

### Conservative:
- Face flux same from both sides
- All FVM schemes are conservative by construction

### Bounded:
- No over/undershoots
- Solution stays within physical limits
- Bounded schemes: Upwind (1st), Hybrid, Van Leer

### Transportive:
- Recognizes flow direction
- All upwind-based schemes are transportive

---

## Common Exam Questions:

1. **Show QUICK is 3rd order** - Taylor expansion proof
2. **Van Leer = 2nd order upwind** - Constant gradient assumption
3. **Face flux from both sides** - Conservation proof
4. **Scheme coefficients** - Contribution to a_P, a_E, a_W
5. **Peclet number formulation** - Derive flux using Pe
6. **Bounded vs conservative** - Definitions and examples

---

## Key Formulas to Memorize:

```
# Peclet Number
Pe = F/D = ρuδx/Γ

# Central (2nd order)
φ_e = (φ_P + φ_E)/2
a_E = D_e - F_e/2

# Upwind 1st (bounded)
φ_e = φ_P  (if F > 0)
a_E = D_e + max(0, -F_e)

# Upwind 2nd
φ_e = 3/2·φ_P - 1/2·φ_W

# QUICK (3rd order)
φ_e = 6/8·φ_P + 3/8·φ_E - 1/8·φ_W

# Van Leer
φ_e = φ_P + (φ_E-φ_P)/(φ_E-φ_W)·(φ_P-φ_W)  (with limiter)
```

---

## Tips for Exam:
- Chapter 5 appears MOST FREQUENTLY - absolute priority!
- QUICK 3rd order proof is very common - practice Taylor expansions
- Van Leer = 2nd upwind proof appears multiple times
- Know how to derive coefficients from face flux formula
- Remember: CDS stable only if |Pe| < 2
- Face flux conservation is a fundamental check
