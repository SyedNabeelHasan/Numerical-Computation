🔬 Numerical Solvers for Heat Transfer and Incompressible Flow
This repository provides robust, Python-based numerical solvers for steady-state and 2D transient heat conduction partial differential equations (PDEs), along with a newly added module for incompressible fluid flow simulation. These solvers are tailored for students, researchers, and engineers engaged in heat transfer, fluid mechanics, and scientific computation.

🧠 Core Solvers
⚡ T2dHC Solver (2D Transient Heat Conduction)
Built using the finite difference method (FDM).

Supports multi-point independent boundary conditions and domain inversion.

Available in both:

Explicit scheme (2D-FDM explicit.py)

Implicit scheme (2D-FDM implicit solver.py)
— utilizes a custom Gauss-Seidel iterative solver for handling sparse and dense systems efficiently.

🧮 Gauss-Seidel Linear System Solver
Designed to bypass limitations of built-in libraries like NumPy/SymPy when dealing with sparse matrices.

Capable of solving systems where det(A) ≈ 0, ensuring robustness.

Offers adjustable tolerance settings (default 1e-15) for fine-grained accuracy control.

🌊 Incompressible Flow Solver (Added: March 9, 2025)
Solves pressure Poisson equation and velocity fields via pressure correction techniques.

Now includes Model 3, our fastest and most optimized version, built over the foundations of Model 1 and Model 2.

Developed to enhance the repository’s coverage from thermal problems to core CFD simulations.

🛠 How to Use
Each script is documented with detailed in-line comments. Users can modify parameters such as domain size, time steps, and boundary conditions easily by following the provided guidelines.

If you face difficulties using any solver, you're welcome to contact me at nabeelhasan661@gmail.com.

📊 Sample Results
T2dHC Solver:
Simulations over a unit-sized square domain with varied boundary conditions.
📎 View Sample Output

Pressure Correction Poisson Solver (Model 3):
Validated against benchmark data (e.g., Ghia et al.).
✅ Fastest solver among Model 1, 2, and 3.

🚀 Coming Soon
A user-friendly interface for editing boundary conditions and initial conditions interactively.

Expanded solver suite with visual outputs, improved post-processing, and integration with Matplotlib or ParaView.

Performance enhancements for large-scale simulations and real-time debugging support.
