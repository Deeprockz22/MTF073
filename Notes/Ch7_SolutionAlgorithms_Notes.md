# Chapter 7: Solution Algorithms & TDMA

## Exam Importance: 57.1% (4/7 exams)

---

## 1. System of Linear Equations

### Matrix Form:
```
Ax = b

Where:
A = coefficient matrix
x = solution vector (unknowns)
b = source vector (known values)
```

### For CFD:
```
a_P φ_P = a_E φ_E + a_W φ_W + a_N φ_N + a_S φ_S + b
```

---

## 2. Properties of CFD Coefficient Matrices

### Sparse:
- Most entries are zero
- Only neighbors have non-zero coefficients
- Storage: store only non-zero values

### Diagonal Dominant:
```
|a_P| ≥ Σ|a_nb|

Required for convergence of iterative methods
```

### Symmetric (for pure diffusion):
```
A = A^T

Not symmetric for convection-diffusion
```

---

## 3. Direct vs Iterative Methods

### Direct Methods:
- Gaussian elimination, LU decomposition
- Exact solution (within round-off)
- **Expensive** for large 3D problems: O(N³)
- TDMA is direct for 1D

### Iterative Methods:
- Jacobi, Gauss-Seidel, SOR, Multigrid
- Approximate solution (within tolerance)
- **Efficient** for large systems: O(N)
- Suitable for sparse matrices

---

## 4. TDMA (Tri-Diagonal Matrix Algorithm)

### Also Known As:
- Thomas Algorithm
- Tri-Diagonal Matrix Solver

### Standard 1D Equation:
```
a_i T_i = b_i T_{i+1} + c_i T_{i-1} + d_i

Where:
a_i = diagonal coefficient (a_P)
b_i = east coefficient (a_E)
c_i = west coefficient (a_W)
d_i = source term
```

### Matrix Form:
```
⎡ a₁  b₁   0   0  ...  0  ⎤   ⎡T₁⎤   ⎡d₁⎤
⎢ c₂  a₂  b₂   0  ...  0  ⎥   ⎢T₂⎥   ⎢d₂⎥
⎢  0  c₃  a₃  b₃  ...  0  ⎥ × ⎢T₃⎥ = ⎢d₃⎥
⎢ ...              ...     ⎥   ⎢..⎥   ⎢..⎥
⎣  0   0   0  ... c_N a_N ⎦   ⎣T_N⎦  ⎣d_N⎦
```

---

## 5. TDMA Algorithm

### Two Steps: Forward Elimination + Back Substitution

#### Forward Elimination:
```
Convert: a_i T_i = b_i T_{i+1} + c_i T_{i-1} + d_i
To:      T_i = P_i T_{i+1} + Q_i

Recurrence relations:
P_i = b_i / (a_i - c_i·P_{i-1})
Q_i = (d_i + c_i·Q_{i-1}) / (a_i - c_i·P_{i-1})

Starting values (boundary):
P_0 = 0  (no T_{-1})
Q_0 = T_0  (boundary value)
```

#### Back Substitution:
```
Starting from i = N (east boundary):
T_N = Q_N  (if b_N = 0)

Then march westward:
T_i = P_i·T_{i+1} + Q_i    for i = N-1, N-2, ..., 1
```

---

## 6. How TDMA Changes Coefficient Matrix

### Original Form (Tri-diagonal):
```
a_i T_i = b_i T_{i+1} + c_i T_{i-1} + d_i

Matrix has 3 diagonals:
- Main diagonal: a_i
- Upper diagonal: b_i
- Lower diagonal: c_i
```

### After Forward Elimination (Bi-diagonal):
```
T_i = P_i T_{i+1} + Q_i

Matrix has 2 diagonals:
- Main diagonal: 1
- Upper diagonal: P_i
- RHS: Q_i
```

### Structure Change:
```
Before:                After:
a₁  b₁                 1   P₁
c₂  a₂  b₂            0   1   P₂
    c₃  a₃  b₃           0   1   P₃
        c₄  a₄              0   1
```

### Why This Enables Direct Solution:
1. **Bi-diagonal system** is trivial to solve
2. **No coupling** with downstream nodes
3. Start at T_N, calculate T_{N-1}, T_{N-2}, ... T_1
4. **Single pass** through domain - O(N) operations

---

## 7. 2D to 1D Representation for TDMA

### 2D Equation:
```
a_P T_{i,j} = a_E T_{i+1,j} + a_W T_{i-1,j} + a_N T_{i,j+1} + a_S T_{i,j-1} + b_{i,j}
```

### Strategy: Line-by-Line Solution

#### Sweep in x-direction (along i):
```
Treat T_{i,j+1} and T_{i,j-1} as known (from previous iteration)

Rearrange:
a_P T_{i,j} = a_E T_{i+1,j} + a_W T_{i-1,j} + [a_N T_{i,j+1} + a_S T_{i,j-1} + b_{i,j}]
                                                 ⎣_________________________________________⎦
                                                              d_{i,j} (modified source)

1D form:
a_{i,j} T_{i,j} = b_{i,j} T_{i+1,j} + c_{i,j} T_{i-1,j} + d_{i,j}

Where:
a_{i,j} = a_P
b_{i,j} = a_E
c_{i,j} = a_W
d_{i,j} = a_N T_{i,j+1} + a_S T_{i,j-1} + b_{i,j}
```

#### Procedure:
1. For each j (row), solve 1D problem in i (columns)
2. Use TDMA for each line
3. Iterate until convergence

---

## 8. ADI (Alternating Direction Implicit)

### Concept:
- Split 2D problem into two 1D problems
- Alternate between x-sweeps and y-sweeps

### Half Time Step in x:
```
Implicit in x, explicit in y
Solve using TDMA in x-direction
```

### Half Time Step in y:
```
Implicit in y, explicit in x
Solve using TDMA in y-direction
```

### Advantages:
- Unconditionally stable
- Each direction: TDMA (fast)
- Better than explicit methods

---

## 9. Iterative Methods

### Jacobi:
```
T_P^{new} = (1/a_P)[Σ a_nb T_nb^{old} + b]

Use all OLD values
```

### Gauss-Seidel:
```
T_P^{new} = (1/a_P)[Σ a_nb T_nb^{latest} + b]

Use latest available values
Faster convergence than Jacobi
```

### SOR (Successive Over-Relaxation):
```
T_P^{new} = T_P^{old} + ω(T_P^{GS} - T_P^{old})

Where: 1 < ω < 2
Accelerates Gauss-Seidel
```

---

## 10. Convergence Criteria

### Residual:
```
R_P = |a_P T_P - Σ a_nb T_nb - b|

Normalized residual:
R_norm = R_P / max(|a_P T_P|, |Σ a_nb T_nb|, |b|)
```

### Convergence Check:
```
max(R_norm) < tolerance  (e.g., 10^-6)

Or:

Σ|R_P|/Σ|a_P T_P| < tolerance
```

---

## 11. Efficiency Considerations

### TDMA:
- **O(N)** operations per line
- Direct solution (no iteration)
- Ideal for 1D problems

### Line-by-Line with TDMA:
- **O(N²)** per iteration for 2D
- Requires outer iteration loop
- Better than direct Gaussian elimination: O(N³)

### Multigrid:
- **O(N)** total for 2D/3D
- Most efficient for large problems
- Complex to implement

---

## Common Exam Questions:

1. **TDMA forward elimination** - Derive P_i and Q_i formulas
2. **Change in coefficient matrix** - Tri-diagonal → bi-diagonal
3. **How TDMA provides direct solution** - Back substitution process
4. **2D to 1D representation** - Line-by-line, modified source term
5. **Diagonal dominance requirement** - Why needed for convergence
6. **ADI method** - Alternating direction approach

---

## Key Formulas to Memorize:

```
# TDMA standard form
a_i T_i = b_i T_{i+1} + c_i T_{i-1} + d_i

# Forward elimination
T_i = P_i T_{i+1} + Q_i

P_i = b_i / (a_i - c_i·P_{i-1})
Q_i = (d_i + c_i·Q_{i-1}) / (a_i - c_i·P_{i-1})

# Back substitution
T_i = P_i·T_{i+1} + Q_i

# 2D to 1D
d_{i,j} = a_N T_{i,j+1} + a_S T_{i,j-1} + b_{i,j}

# Diagonal dominance
|a_P| ≥ Σ|a_nb|
```

---

## Tips for Exam:
- TDMA appears in 57% of exams - important topic
- Understand structure change: tri-diagonal → bi-diagonal
- Know why bi-diagonal enables direct solution
- 2D to 1D: treat perpendicular direction as source
- Forward elimination modifies coefficients
- Back substitution is trivial: one sweep
- Complexity: O(N) for TDMA vs O(N³) for Gauss elimination
