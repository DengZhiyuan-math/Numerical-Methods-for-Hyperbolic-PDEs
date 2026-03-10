import numpy as np
import math, sys
import matplotlib.pyplot as plt

"Function: solve u_t+2*u_x=0"

def init(x):
    u = np.zeros(len(x))
    for index in range(len(x)):
        if x[index] < 0:
            u[index] = -1
        else:
            u[index] = 1
    return u
    
def exactu(t, x):
    u = np.zeros(len(x))
    for index in range(len(x)):
        if x[index] < 2*t:
            u[index] = -1
        else:
            u[index] = 1
    return u

def bc(N, uc):
    # Zero order extrapolation
    uc[-1] = uc[-2]
    uc[0] = uc[1]
    return uc
        
def getflux(ischeme, lambda_, uc):
    flux = lambda u: 2*u
    fluxp = lambda u: 2
    f = flux(uc)*np.ones(len(uc))
    fp = fluxp(uc)*np.ones(len(uc))
    
    if ischeme == "upwind":
        fh = upwind(uc, f, fp)
    elif ischeme == "lax-wendroff":
        fh = lax_wendroff(lambda_, uc, f, fp)
    elif ischeme == "central":
        fh = central(f)
    else:
        sys.exit("ERROR: this is not defined here.")
    rhs = fh[1:]-fh[0:-1]
    return rhs

def upwind(u, f, fp):
    ah = np.zeros(len(u)-1)
    for i in range(len(u)-1):
        if u[i] == u[i+1]:
            ah[i] = fp[i]
        else:
            ah[i] = (f[i+1]-f[i])/(u[i+1]-u[i])
    fh = np.zeros(len(u)-1)
    fh = 0.5*(f[0:-1]+f[1:])-0.5*abs(ah)*(u[1:]-u[0:-1])
    return fh

def lax_wendroff(lambda_, u, f, fp):
    ah = np.zeros(len(u)-1)
    for i in range(len(u)-1):
        if u[i] == u[i+1]:
            ah[i] = fp[i]
        else:
            ah[i] = (f[i+1]-f[i])/(u[i+1]-u[i])
    fh = np.zeros(len(u)-1)
    fh = 0.5*(f[0:-1]+f[1:])-0.5*lambda_*ah**2*(u[1:]-u[0:-1])
    return fh
        
def central(f):
    fh = np.zeros(len(f)-1)
    fh = 0.5*(f[0:-1]+f[1:])
    return fh
    
def L1err(dx, uc, uex):
    u_diff = abs(uc-uex)
    err = sum(u_diff)*dx
    return err

def L2err(dx, uc, uex):
    u_diff = abs(uc-uex)
    err = sum(np.square(u_diff))*dx
    err = math.sqrt(err)
    return err

def L8err(uc, uex):
    u_diff = abs(uc-uex)
    err = np.max(u_diff)
    return err

def ErrorOrder(err, NN):
    ord = np.log(err[0:-1]/err[1:])/np.log(NN[1:]/NN[0:-1])
    return ord
    
if __name__ == "__main__":
    # setup
    cfl = 0.4
    tend = 2

    xleft = -10
    xright = 10
    N = 100
    
    # uniform grid
    x = np.linspace(xleft, xright, N+1)
    dx = (xright-xleft)/N
    dt = cfl*dx/2
    lambda_ = dt/dx
    
    ischeme = "lax-wendroff"
    # "lax-wendroff" "upwind"
    
    # initialize
    uc = np.zeros(N+3) #there are 2 ghost points for the boundary, one on the left and another on the right.
    uc[1:-1] = init(x)
    rhs = np.zeros(N+1)
    
    # exact solution
    uex = np.zeros(N+1)
    uex = exactu(tend, x)

    # time evolution
    time = 0
    kt = 0
    kmax = 1000
    #while (time < tend)&(kt<kmax):
    while time < tend:
        kt = kt+1
        if time+dt >= tend:
            dt = tend-time
            lambda_ = dt/dx
        time = time+dt
        
        uc = bc(N, uc)
        rhs = getflux(ischeme, lambda_, uc)
        
        # Euler forward: update point values from 1 to N+1.
        uc[1:-1] = uc[1:-1]-dt/dx*rhs
    uc = bc(N, uc)
    
    with open ('numerical_sol_{}.txt'.format(ischeme), 'w', encoding='utf-8') as file:
        for index in range(len(x)):
            file.write("%f %f %f \n"%(x[index], uex[index], uc[index+1]))
    file.close()
        
    plt.scatter(x, uc[1:-1], label = '%6s'%(ischeme))
    plt.plot(x, uc[1:-1])
    plt.legend()
    plt.xlabel('x');plt.ylabel('u')
    plt.plot(x, uex, 'k', label="exact")
    plt.show()
    



    
        
    
    


