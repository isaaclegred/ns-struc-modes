# Generate the equations for the K2  and h2 example (for verification)

from sympy import *
import inspect

r = symbols("r")
k2 = Symbol("K2")
h2 = Symbol("h2")
M = symbols("M")
p = symbols("p")
lam = symbols("lambda")
m2 = symbols("m2")
rho = symbols("mu")
f = symbols("f")
beta = symbols("beta")
omega1 = symbols("omega1")
nu = symbols("nu")

dh2dr_inhomo =  (3-4 * pi* (rho + p) * r**2)/r * f * h2 + 2/r*f * k2 + (1+8* pi*p * r**2/r**2) * f**2 * m2 + exp(-nu) * r/12 * omega1**2 * beta**2  - (4*pi * (p+rho)* r**4*omega1**2)/(3*r) * exp(-nu)/f

dk2dr = 1/(1 - (r - M + 4*pi*p *r**3)/r * 1/f) * (
    -dh2dr_inhomo + (r - 3 * M - 4 * pi * p * r**3)/r**2 * 1/f * h2 + (r-M + 4*pi*p*r**3/r**3)*1/f**2 * m2
)

dk2dr = simplify(dk2dr)


dh2dr = simplify((r - M + 4 * pi * p * r**r)/(r*f)*dk2dr +  dh2dr_inhomo)

dk2dr_func = lambdify((r, k2, h2, M, p, lam, m2, rho, f, beta, omega1, nu), dk2dr, cse=True)

dh2dr_func = lambdify((r, k2, h2, M, p, lam, m2, rho, f, beta, omega1, nu), dh2dr, cse=True)



print(inspect.getsource(dh2dr_func))

print(inspect.getsource(dK2dr_func))
