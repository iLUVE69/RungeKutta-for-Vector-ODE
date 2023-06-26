import numpy as np
import matplotlib.pyplot as plt

def dxdt(t, x, y, z, a,b): ################ 1st Differential Equation##################
    I = np.eye(3)
    x_transpose = np.transpose(x)
    dx = -a * np.dot((I - np.outer(x, x_transpose)), np.dot(x,y)) - b * np.dot((I - np.outer(x, x_transpose)), np.dot(x,z))
    return dx

def dydt(t, y, v, c):     ############## 2nd Differential Equation ##############
    v_norm = np.linalg.norm(v)
    dy = -c * y + (((np.outer(v, v)) /(1 + v_norm**2)))
    return dy

def dzdt(t, z, v, c, f): ##########3rd Differential Equation###################3
    n = np.cross(f, v)
    n_norm = np.linalg.norm(n)
    dz = -c * z + ((np.outer(n, n)) / (1 + n_norm**2))
    return dz

def runge_kutta4(t, x, y, z, a, b,v, c, f, h):
   
    k1 = h * dxdt(t, x, y, z, a, b)
    k2 = h * dxdt(t + 0.5 * h, x + 0.5 * k1, y, z, a, b)
    k3 = h * dxdt(t + 0.5 * h, x + 0.5 * k2, y, z, a, b)
    k4 = h * dxdt(t + h, x + k3, y, z, a, b)
    x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    d1 = h * dydt(t, y, v, c)
    d2 = h * dydt(t + 0.5 * h, y + 0.5 * d1, v, c)
    d3 = h * dydt(t + 0.5 * h, y + 0.5 * d2, v, c)
    d4 = h * dydt(t + h, y + d3, v, c)
    y_new = y + (d1 + 2 * d2 + 2 * d3 + d4) / 6.0

    e1 = h * dzdt(t, z, v, c, f)
    e2 = h * dzdt(t + 0.5 * h, z + 0.5 * e1, v, c, f)
    e3 = h * dzdt(t + 0.5 * h, z + 0.5 * e2, v, c, f)
    e4 = h * dzdt(t + h, z + e3, v, c, f)
    z_new = z + (e1 + 2 * e2 + 2 * e3 + e4) / 6.0

    return x_new, y_new, z_new

def solve_coupled_equations(x0, y0, z0,  a, b,v, c, f, h, num_points):
    t = np.zeros(num_points)
    x = np.zeros((num_points, 3))
    y = np.zeros((num_points, 3, 3))
    z = np.zeros((num_points, 3, 3))
    x[0] = x0
    y[0] = y0
    z[0] = z0

    for i in range(1, num_points):
        t[i] = t[i-1] + h
        x[i], y[i], z[i] = runge_kutta4(t[i-1], x[i-1], y[i-1], z[i-1], a, b,v, c, f, h)
   
    return t, x, y, z
    

# Initial conditions
mean = 0.1
std =0.005
#x0 = np.array([0.57706, 0.57706, 0.57706])

x0 = np.array([0.42426407, 0.56568542, 0.70710678])
y0=np.zeros((3,3))
z0=np.zeros((3,3))

#y0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
#z0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
v = np.array([1.0,1.0,0.0])
f = np.array([2.0,2.0,-1.0])
#f += np.random.normal(mean, std, size=f.shape)
#v += np.random.normal(mean, std, size=v.shape)

a = 10
b = 15
c = 5

# Step size and number of points
h = 0.1
num_points = 1001

# Solve the equations
t, x, y, z = solve_coupled_equations(x0, y0, z0, a, b,v, c, f, h, num_points)

print(x) 
#print(np.dot(x,v)) ## final normal concludes and agrees with the given velocity constraint
#print(np.dot(x,np.cross(f,v)))

plt.figure(figsize=(10, 6))

# Plotting x components
plt.subplot(2, 2, 1)
plt.plot(t, x[:, 0], label='x1')
plt.plot(t, x[:, 1], label='x2')
plt.plot(t, x[:, 2], label='x3')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Solution for x')
plt.legend()

# Plotting y components
plt.subplot(2, 2, 2)
plt.plot(t, y[:, 0, 0], label='y11')
plt.plot(t, y[:, 0, 1], label='y12')
plt.plot(t, y[:, 0, 2], label='y13')
plt.plot(t, y[:, 1, 0], label='y21')
plt.plot(t, y[:, 1, 1], label='y22')
plt.plot(t, y[:, 1, 2], label='y23')
plt.plot(t, y[:, 2, 0], label='y31')
plt.plot(t, y[:, 2, 1], label='y32')
plt.plot(t, y[:, 2, 2], label='y33')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Solution for y')
plt.legend()

# Plotting z components
plt.subplot(2, 2, 3)
plt.plot(t, z[:, 0, 0], label='z11')
plt.plot(t, z[:, 0, 1], label='z12')
plt.plot(t, z[:, 0, 2], label='z13')
plt.plot(t, z[:, 1, 0], label='z21')
plt.plot(t, z[:, 1, 1], label='z22')
plt.plot(t, z[:, 1, 2], label='z23')
plt.plot(t, z[:, 2, 0], label='z31')
plt.plot(t, z[:, 2, 1], label='z32')
plt.plot(t, z[:, 2, 2], label='z33')
plt.xlabel('Time')
plt.ylabel('z')
plt.title('Solution for z')
plt.legend()

plt.tight_layout()
plt.show()
