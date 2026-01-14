# MTF073 Computational Fluid Dynamics - Comprehensive Study Notes

## 📘 About This Repository

This repository contains comprehensive study notes for **MTF073 Computational Fluid Dynamics** at Chalmers University of Technology, based on detailed analysis of **7 years of exam papers (2019-2025)**.

---

## 📊 Exam Analysis Summary

### Key Statistics:
- **7 exam papers analyzed** (2019-2025)
- **Pattern identified**: 79-86 points, 6-10 questions per exam
- **Most tested chapters identified** with frequency analysis
- **Recurring topics** mapped across years

### Chapter Coverage in Exams:
| Chapter | Frequency | Priority |
|---------|-----------|----------|
| Ch 5: Convection Schemes | 114.3% | ⭐⭐⭐ CRITICAL |
| Ch 4: Boundary Conditions | 85.7% | ⭐⭐ VERY HIGH |
| Ch 8: Unsteady Flows | 85.7% | ⭐⭐ VERY HIGH |
| Ch 11: Pressure-Velocity | 85.7% | ⭐⭐ VERY HIGH |
| Ch 2: Fundamentals | 71.4% | ⭐ HIGH |
| Ch 3: Turbulence | 71.4% | ⭐ HIGH |
| Ch 6: SIMPLE Algorithm | 57.1% | MEDIUM |
| Ch 7: Solution Algorithms | 57.1% | MEDIUM |
| Ch 9: NS Boundary Conditions | 57.1% | MEDIUM |

---

## 📚 Study Notes by Chapter

### Core Notes:
1. **[Ch2_Fundamentals_Notes.md](Notes/Ch2_Fundamentals_Notes.md)** - Eulerian approach, pressure gradients, FVM basics
2. **[Ch3_Turbulence_Notes.md](Notes/Ch3_Turbulence_Notes.md)** - Eddy viscosity, k-ε model, wall functions
3. **[Ch4_BoundaryConditions_Notes.md](Notes/Ch4_BoundaryConditions_Notes.md)** - Interpolation, Neumann BCs, global balance
4. **[Ch5_ConvectionSchemes_Notes.md](Notes/Ch5_ConvectionSchemes_Notes.md)** - QUICK, Van Leer, upwind schemes
5. **[Ch6_PressureVelocity_Notes.md](Notes/Ch6_PressureVelocity_Notes.md)** - SIMPLE algorithm, under-relaxation
6. **[Ch7_SolutionAlgorithms_Notes.md](Notes/Ch7_SolutionAlgorithms_Notes.md)** - TDMA, iterative methods
7. **[Ch8_UnsteadyFlows_Notes.md](Notes/Ch8_UnsteadyFlows_Notes.md)** - Time discretization, Crank-Nicolson
8. **[Ch9_BoundaryConditionsNS_Notes.md](Notes/Ch9_BoundaryConditionsNS_Notes.md)** - Outlet BCs, symmetry
9. **[Ch11_AdvancedPressure_Notes.md](Notes/Ch11_AdvancedPressure_Notes.md)** - Rhie-Chow, collocated grids

### Master Summary:
- **[MASTER_EXAM_SUMMARY.md](Notes/MASTER_EXAM_SUMMARY.md)** - Quick reference guide with all critical topics
- **[REPEATED_QUESTIONS_ANALYSIS.md](Notes/REPEATED_QUESTIONS_ANALYSIS.md)** - ⚡ Analysis of repeated questions (2020-2025)

---

## ⚡ REPEATED QUESTIONS DISCOVERED!

### 🚨 **Major Finding: Significant Question Repetition**

Analysis reveals **11 major question patterns** that repeat across years:

**Top 5 Most Repeated:**
1. **SIMPLE/Rhie-Chow Topics** - 4 times (2020, 2021, 2023, 2024)
2. **Crank-Nicolson Discretization** - 3 times (2020, 2024, 2025)
3. **Wall Function Source Terms** - 3 times (2020, 2022, 2024)
4. **Time Discretization Details** - 3 times (2021, 2022, 2023)
5. **TDMA/Matrix Structure** - 3 times (2023, 2024, 2025)

**🆕 Emerging Trends (Back-to-Back 2024-2025):**
- Rhie-Chow at Boundary
- Fully Developed Outlet BC

**📊 See full analysis:** [REPEATED_QUESTIONS_ANALYSIS.md](Notes/REPEATED_QUESTIONS_ANALYSIS.md)

---

## 🔥 Most Frequently Asked Exam Questions

### Top 10 Topics (Based on Frequency Analysis):

1. **QUICK Scheme 3rd Order Proof** (Ch5) - Appears almost every year
2. **Crank-Nicolson Discretization** (Ch8) - Very high frequency
3. **Wall Function Source Terms** (Ch3) - 3/7 exams
4. **Interpolation Factor Derivation** (Ch4) - High frequency
5. **Rhie-Chow Interpolation** (Ch11) - Recent years trend
6. **Fully Developed Outlet BC** (Ch9) - 3/7 recent exams
7. **Van Leer = 2nd Order Upwind** (Ch5) - Multiple appearances
8. **Implicit Under-Relaxation** (Ch6) - Medium-high
9. **Global Heat Balance** (Ch4) - Medium
10. **TDMA Matrix Structure** (Ch7) - Medium

---

## 📈 Trends & Predictions

### Increasing Topics (2023-2025):
- Rhie-Chow interpolation details
- Fully developed outlet implementations
- Coefficient matrix analysis in solvers
- Boundary-specific implementations

### Stable Core Topics (Every Year):
- Discretization scheme derivations
- Time stepping methods
- Boundary condition implementations
- Conservation principles

---

## 🎯 How to Use These Notes

### For Quick Review:
1. Start with **MASTER_EXAM_SUMMARY.md**
2. Focus on ⭐⭐⭐ priority chapters (4, 5, 8, 11)
3. Practice the "Most Frequently Asked Questions"

### For Deep Understanding:
1. Read individual chapter notes in order
2. Work through derivations step-by-step
3. Practice proofs (Taylor expansions, algebraic manipulations)
4. Verify with exam papers

### One Week Before Exam:
- [ ] Review all key formulas in each chapter
- [ ] Practice: QUICK proof, CN discretization, Rhie-Chow
- [ ] Memorize: Standard coefficients, boundary conditions
- [ ] Understand: Physical meanings, why methods work

---

## 📖 Key Formulas Reference

### Chapter 5 (Convection):
```
QUICK: φ_e = 6/8·φ_P + 3/8·φ_E - 1/8·φ_W
2nd Upwind: φ_e = 3/2·φ_P - 1/2·φ_W
Peclet: Pe = ρuδx/Γ
```

### Chapter 8 (Unsteady):
```
Time term: a_P^0 = ρV/Δt
Crank-Nicolson: a_E = Δt/2·D_e, a_P = a_P^0 + a_E + a_W
```

### Chapter 11 (Pressure):
```
Rhie-Chow: u_e = (u_E+u_P)/2 + d_e[(∇p)_avg - (∇p)_direct]
Velocity correction: d_u = A/a_P^u
```

---

## 📁 Repository Structure

```
MTF073/
├── Notes/
│   ├── Ch2_Fundamentals_Notes.md
│   ├── Ch3_Turbulence_Notes.md
│   ├── Ch4_BoundaryConditions_Notes.md
│   ├── Ch5_ConvectionSchemes_Notes.md
│   ├── Ch6_PressureVelocity_Notes.md
│   ├── Ch7_SolutionAlgorithms_Notes.md
│   ├── Ch8_UnsteadyFlows_Notes.md
│   ├── Ch9_BoundaryConditionsNS_Notes.md
│   ├── Ch11_AdvancedPressure_Notes.md
│   └── MASTER_EXAM_SUMMARY.md
├── Question Papers/ (PDFs)
└── README.md
```

---

## 🤝 Contributing

These notes are based on exam paper analysis. If you find errors or have suggestions:
- Create an issue
- Submit a pull request
- Share additional insights

---

## 📜 Disclaimer

These notes are study aids based on past exam analysis. Always refer to:
- Official course materials
- Lecture notes
- Textbooks
- Course website

---

## 📅 Last Updated

January 2026 - Based on exams through 250117

---

## 🎓 Good Luck!

**Tips for Success:**
- Understand derivations, don't just memorize
- Practice writing out full solutions
- Check your work (dimensional analysis, limit cases)
- State assumptions clearly
- Show all steps for partial credit

---

*Star ⭐ this repo if it helps you!*
