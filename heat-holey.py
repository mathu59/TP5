#!/usr/bin/python3

#--- EDP TP MSIAM 1
#--- TP Heat equation
#--- Maelle Nodet, 2016
# Acknowledgement for the animation: author: Jake Vanderplas
# email: vanderplas@astro.washington.edu website: http://jakevdp.github.com

import numpy as np
import scipy
import math
from matplotlib import pyplot as plt
from matplotlib import animation

# do the checkups or not
verif = 1
question2 = "c"

#--- Initial set up

# independant parameters
n = 20 # grid point in space
dx = 1 / n # step in space
x = np.arange(0, 1+dx, dx)# [0:dx:1]
sx = x.size
m = 500 # nb of time steps
T = 1
dt = 1 / m
t = np.arange(0, T+dt, dt)# 0:dt:T
st = t.size
C = dt/(dx**2)

# initial condition
f = lambda z: np.sin(math.pi * z)
u0 = f(x)

g = lambda z: np.sin(math.pi * (z+t/(math.pi ** 2)))
# plt.figure()
# plt.plot(x, f(x),label="f")
# plt.legend()
# plt.show()

#--- PDE numerical resolution : first definitions

# Define A, Laplacian matrix
A = np.zeros((n + 1, n + 1))
for k in range(1, n):
    A[k, k-1] = 1
    A[k, k] = -2
    A[k, k+1] = 1

# boundary conditions
A[0, 0] = 1
A[n, n] = 1

# right hand side
F1 = 0
F2 = 1

# constant parameter in the pde
D = C/(math.pi ** 2)

fig = plt.figure(2)
#--- Implementation (QUESTION 2a)
if question2 == "a":
	# Euler in time, explicit 2nd order in space
	AA = D*A + np.eye(n+1) # FILL HERE
	AA[0,0] = 1  # not used, just to ensure the matrix
	AA[n,n] = 1  # is inversible if necessary

	u = np.zeros((st, sx))
	u[0,:] = u0
	for i in range(0, st-1):
		u[i+1,:] = AA.dot(u[i,:])
		u[i+1, 0] = F1
		u[i+1, n] = F2

#--- Implementation (QUESTION 2b)
if question2 == "b":
	# Euler in time, explicit 2nd order in space
	AA = D*A 
	AA[0,0] -= 1  
	AA[n,n] -= 1

	u = np.zeros((st, sx))
	u[0,:] = u0
	u[1,:] = AA.dot(u[0,:])
	u[1, 0] = F1
	u[1, n] = F2
	for i in range(1, st-1):
		u[i+1,:] = AA.dot(u[i,:]) + u[i-1,:]
		u[i+1, 0] = F1
		u[i+1, n] = F2

#--- Implementation (QUESTION 2b)
if question2 == "c":
	# Euler in time, explicit 2nd order in space
	AA =  -D*A + np.eye(n+1) # FILL HERE
	AA[0,0] = 1
	AA[n,n] = 1

	u = np.zeros((st, sx))
	u[0,:] = u0
	for i in range(0, st-1):
		u[i+1,:] = (np.linalg.inv(AA)).dot(u[i,:])
		u[i+1, 0] = F1
		u[i+1, n] = F2


# animation for u
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(2)
ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
line, = ax.plot([], [])

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    line.set_data(x, u[i,:])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
# careful, on a Mac blit=True is a known bug => set it to False!
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=st, interval=50, blit=False)
plt.show()

# COMPLETE FOR THE REST OF THE PRACTICAL SESSION QUESTIONS

# IF YOU CHOOSE THIS SESSION FOR YOUR FINAL EXAM:
# REACHABLE GRADES:
# If you stop at question 5:
#		- acceptable: if report is reasonably good, and most questions are ok
#		- exceeds expectations: if report is great and all questions are correct
#
# If you stop at question 9:
#		- acceptable: if report is acceptable and some questions are ok
#		- exceeds expectations: if report is good and most questions are correct
#		- outstanding: great report, everything is correct
#
# If you also do the 2D questions (q 10 and q 11):
# 		- exceeds expectations: if report is acceptable and most questions 1-9 are correct, but questions 10-11 do not work well
#		- outstanding: good report, most questions are correct, including questions 10-11
