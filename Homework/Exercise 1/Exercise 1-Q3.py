import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# parameters
# -----------------------------
u0 = lambda x: np.sin(2*np.pi*x)
a = 1.0

T = 5
xl = 0
xr = 1

NN = [100,200,400,800,1600]

Errnorm = np.zeros(len(NN))
Errnorm2 = np.zeros(len(NN))

plt.figure()

# -------------------------------------------------
# upwind scheme
# -------------------------------------------------
def upwind_scheme(x, CFL, T, a, c0):

    dx = x[1]-x[0]
    dt = CFL*dx
    Nt = int(np.ceil(T/dt))

    t = np.linspace(0,T,Nt+1)
    dt = t[1]-t[0]

    lam = dt/dx

    c = np.zeros((len(x),len(t)))
    c[:,0] = c0

    for j in range(len(t)-1):

        for i in range(1,len(x)-1):

            if a >= 0:
                c[i,j+1] = c[i,j] - lam*a*(c[i,j]-c[i-1,j])
            else:
                c[i,j+1] = c[i,j] - lam*a*(c[i+1,j]-c[i,j])

        # periodic BC
        c[0,j+1] = c[-2,j+1]
        c[-1,j+1] = c[1,j+1]

    return c[:,-1]

# -------------------------------------------------
# main loop
# -------------------------------------------------
for nn,N in enumerate(NN):

    dx = (xr-xl)/N

    # grid including ghost cells
    x = np.arange(-dx/2 + xl, xr + dx/2 + dx, dx)

    c0 = np.zeros(len(x))

    # initial condition
    for i in range(1,len(x)-1):
        c0[i] = u0(x[i])

    # periodic BC
    c0[0] = c0[-2]
    c0[-1] = c0[1]

    CFL = 0.9

    upwind = upwind_scheme(x, CFL, T, a, c0)

    # exact solution
    exact = np.zeros(len(x))
    for i in range(1,len(x)-1):
        exact[i] = u0(x[i]-a*T)

    exact[0] = exact[-2]
    exact[-1] = exact[1]

    # error norms
    Errnorm[nn] = np.linalg.norm(upwind-exact,1)*dx
    Errnorm2[nn] = np.linalg.norm(upwind-exact,2)*np.sqrt(dx)

    plt.plot(x, upwind, linewidth=1, label=f"dx=1/{N}")

# exact solution
plt.plot(x, exact, 'k', linewidth=2, label="Exact")

plt.xlim([0,1])
plt.ylim([-1,1])
plt.grid()
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.show()

# -------------------------------------------------
# convergence order
# -------------------------------------------------
Order1 = []
Order2 = []

for k in range(len(NN)-1):

    Order1.append(np.log2(Errnorm[k]/Errnorm[k+1]))
    Order2.append(np.log2(Errnorm2[k]/Errnorm2[k+1]))

print("L1 errors =",Errnorm)
print("L2 errors =",Errnorm2)
print("Order L1 =",Order1)
print("Order L2 =",Order2)