import numpy as np
import matplotlib.pyplot as plt

# initial coordinate
x0 = np.linspace(-1,1,400)

# initial condition
u0 = np.sin(np.pi*x0)

# times
times = [0,0.15,0.3]

plt.figure(figsize=(10,5))

# plot deformed waves
for t in times:
    x = x0 + u0*t
    plt.plot(x,u0,label=f"t={t}")

# initial wave
plt.plot(x0,u0,'k',linewidth=2,label="initial")

plt.xlabel("x")
plt.ylabel("U")
plt.title("Deformation of Burgers solution")
plt.legend()
plt.grid()

plt.show()


plt.figure(figsize=(7,6))

t = np.linspace(0,0.35,200)

# background characteristics
for x0 in np.linspace(-1,1,40):
    u0 = np.sin(np.pi*x0)
    x = x0 + u0*t
    plt.plot(x,t,color="lightgray",linewidth=1)

# highlighted characteristics
for x0 in [-0.8,-0.4,0,0.4,0.8]:
    u0 = np.sin(np.pi*x0)
    x = x0 + u0*t
    plt.plot(x,t,color="blue",linewidth=2)

plt.xlabel("x")
plt.ylabel("t")
plt.title("Characteristic curves for Burgers equation")
plt.grid(alpha=0.3)

plt.show()