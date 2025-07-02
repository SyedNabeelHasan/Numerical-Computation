import sympy as sp
from colorama import Fore, Style, init
import matplotlib.pyplot as plt

print("")
print(Fore.BLUE + "🐚 Welcome to 2D eigenvalue & eigenvector solver" + Style.RESET_ALL)
print("")

# Define symbol
lam = sp.symbols('lam')

# Define matrices using sympy.Matrix from the beginning
A = sp.Matrix([
    [1, 1],
    [4, 1]
])

I = sp.eye(2)  # Identity matrix

# Now subtract symbolically
coff = A - lam * I

# Get determinant (characteristic polynomial)
char_poly = coff.det()

# Solve for eigenvalues
eigenvals = sp.solve(char_poly, lam)

print(Fore.YELLOW + "Characteristic Polynomial:" + Style.RESET_ALL)
print(char_poly)
print("")
print(Fore.YELLOW + "Eigenvalues:" + Style.RESET_ALL +  str(eigenvals))
print("")

v1 = sp.symbols('v1')
v2 = sp.symbols('v2')
v = sp.Matrix([v1, v2])

eq1 = []
eq2 = []
for i in range(0,len(eigenvals),1):
    sys = A - (eigenvals[i] * I) 
    system = sys*v 
    if (i==0):
        eq1.append(system[0])
        eq1.append(system[1]) 
        print(Fore.YELLOW + "system of linear eqn - 1:"+ Style.RESET_ALL)
        sp.pretty_print(eq1[0])
        sp.pretty_print(eq1[1])
        
    else:
        eq2.append(system[0])
        eq2.append(system[1])
        print(Fore.YELLOW + "system of linear eqn - 2:"+ Style.RESET_ALL)
        sp.pretty_print(eq2[0])
        sp.pretty_print(eq2[1])

# Eigen vector evaluation ....
eqn1 = sp.Eq(eq1[0],0)                  # equation formation by equating to zero from set-1
v2_expr_1 = sp.solve(eqn1, v2)[0]       # forming expression for evaluation of v2 from set-1

eqn2 = sp.Eq(eq2[0],0)                  # equation formation by equating to zero from set-2
v2_expr_2 = sp.solve(eqn2, v2)[0]       # forming expression for evaluation of v2 from set-2


# Data-points evaluation and plottting 

x1 = []
y1 = []

x2 = []
y2 = []
for val in range(-5,6,1):
    v_2_a = v2_expr_1.subs(v1, val)
    y1.append(v_2_a)
    x1.append(val)
    v_2_b = v2_expr_2.subs(v1, val)
    y2.append(v_2_b)
    x2.append(val)

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()