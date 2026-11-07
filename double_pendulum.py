#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:30:45 2026

@author: johnathan

The following script uses the RK4-method to numerically solve 
a system of equations describing the motion of a double pendulum system. 

The pendulum angles and momenta are then plotted as a function of time and 
the positions animated.

"""

# =============================================================================
# References for Animation 
#
# 1) https://matplotlib.org/cheatsheets/
#
# 2) https://stackoverflow.com/questions/76583868/overlay-two-animations-within-the-same-window-with-funcanimation
#
# 3) https://matplotlib.org/stable/gallery/animation/double_pendulum.html
# =============================================================================

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

# =============================================================================
# 4th-order Runge-Kutta ODE solver 
#
# RK4 method takes in the arguments Nt= number of times steps, dt= size of each
# time step, y0= intial values of the system, t0= intial time, dyByDt= the 
# equation being solved and dyByDtArgs= additional parameters relevant in the 
# differential equation being solved 
# =============================================================================

def solveODEsWithRK4(Nt, dt, y0, t0, dyByDt, dyByDtArgs=[]):
    y = np.empty((Nt+1, y0.size)) # creates an array to hold solutions
    t = np.empty(Nt+1) # creates an array of time steps
    y[0, :] = y0 # assigns initial conditions to array created above
    t[0] = t0 # assigns initial time to array created above

# compute different approximations for the gradient of the function
    for n in range(Nt):
        F1 = dyByDt(y[n, :],            t[n],          *dyByDtArgs)
        F2 = dyByDt(y[n, :]+0.5*dt*F1,  t[n]+0.5*dt,   *dyByDtArgs)
        F3 = dyByDt(y[n, :]+0.5*dt*F2,  t[n]+0.5*dt,   *dyByDtArgs)
        F4 = dyByDt(y[n, :]+dt*F3,      t[n]+dt,       *dyByDtArgs)
        
# computes a weighted average of the gradients above and uses this to approximate
# the function value at the next time step
        y[n+1, :] = y[n] + (dt/6) * (F1 + 2*F2 + 2*F3 + F4)
        t[n+1] = t[n] + dt
    return y, t 

# =============================================================================
#  The following functions are defined to simplyfy the equations of motion of 
#  the double pendulum system in order to make calculations easier to implement 
#  and to reduce compuational cost of the caculations
# =============================================================================

def delta (T1, T2):
    return ( T1 - T2 )
def mu (m1, m2):    
    return ( m1 + m2 )
def eta (l2, P1):    
    return ( l2 * P1)
def nu (l1, P2):  
    return ( l1 * P2 )
def alpha (l1, l2, m1, m2, delta):   
    return ( l1 * l2 * ( m1 + (m2 * ( np.sin(delta)**2 ))))**-1
def epsilon (alpha, P1, P2, delta):    
    return ( alpha * P1 * P2 * np.sin(delta) )
def gamma (alpha, delta, m2, eta, nu):
    return ( alpha**2 * np.sin(delta) * np.cos(delta) * ( (m2 * eta**2) + (mu(m1,m2) * nu**2) - (m2 * eta * nu * np.cos(delta))))

# =============================================================================
# doublePendulum function beneath defines the physics which determines the 
# evolution of the double pendulum system
# =============================================================================

def doublePendulum1(y, t):
    T1, T2, P1, P2 = y # T1,T2 and P1,P2 are the angles and momenta of the 
                       # closer and farther masses respectively 

    d   = delta(T1, T2)
    a   = alpha(l1, l2, m1, m2, d)
    e   = eta(l2, P1)
    n   = nu(l1, P2)
    ep  = epsilon(a, P1, P2, d)
    gam = gamma(a, d, m2, e, n)

# calculates the changes in angles and changes in momenta w.r.t time
    dT1 = ( a / l1 ) * ( e - (n * np.cos(d)))
    dT2 = ( a / l2 ) * ( mu(m1,m2) * n - (m2 * e * np.cos(d)))
    dP1 = gam - ep - ( mu(m1,m2) * l1 * g * np.sin(T1))
    dP2 = ep - gam - ( m2 * l2 * g * np.sin(T2))
    return ( np.array( [dT1, dT2, dP1, dP2]) ) 

def doublePendulum2(y, t): # defines an identical second pendulum with slightly  
                           # perturbed initial conditions to show choas
    T3, T4, P3, P4 = y

    d   = delta(T3, T4)
    a   = alpha(l3, l4, m3, m4, d)
    e   = eta(l4, P3)
    n   = nu(l3, P4)
    ep  = epsilon(a, P3, P4, d)
    gam = gamma(a, d, m4, e, n)

    dT3 = ( a / l3 ) * ( e - (n * np.cos(d)))
    dT4 = ( a / l4 ) * ( mu(m3,m4) * n - (m4 * e * np.cos(d)))
    dP3 = gam - ep - ( mu(m3,m4) * l3 * g * np.sin(T3))
    dP4 = ep - gam - ( m4 * l4 * g * np.sin(T4))
    return ( np.array( [dT3, dT4, dP3, dP4]) ) 

Nt = 2500   # no. of time steps
dt = 0.0125 # size of each time step in seconds
g = 9.81    # acceleration due to gravity
m1 = 1.     # mass closer to pivot on pendulum 1
m2 = 1.     # mass farther from pivot on pendulum 1
m3 = m1     # mass closer to pivot on pendulum 2
m4 = m2     # mass farther from pivot on pendulum 2
l1 = 1.     # length of rod connecting closer mass to main pivot
l2 = 0.5    # length of rod connecting farther mass to closer mass
l3 = l1     #...
l4 = l2     # second pendulum uses same logic as first
T10 = 360 * np.pi / 361 # initial angles of both masses set to be almost vertical
T20 = 360 * np.pi / 361
T30 = T10 * (1.000001) # initial angle of second pendulum slighty pertured from first
T40 = T20 * (1.000001)
P10 = 0.0 # initial momenta set to zero for both pendulums
P20 = 0.0
P30 = 0.0
P40 = 0.0

y10 = np.array ([T10, T20, P10, P20]) # defines state vectors using an array 
y20 = np.array ([T30, T40, P30, P40]) # containing the intial conditions
t0 = 0

# =============================================================================
#  Passes the double pendulum ODEs and initial conditions into the RK4 ODE
#  solver and unpacks solutions giving the values of the pendulum angles and momenta
# =============================================================================
sol1, t = solveODEsWithRK4(Nt, dt, y10, t0, doublePendulum1)
T1, T2, P1, P2 = sol1[:,0], sol1[:,1], sol1[:,2], sol1[:,3]

sol2, t = solveODEsWithRK4(Nt, dt, y20, t0, doublePendulum2)
T3, T4, P3, P4 = sol2[:,0], sol2[:,1], sol2[:,2], sol2[:,3]

# =============================================================================
# Changes the angle values from polar to cartesian coordinates for ease later in 
# animating the solutions of the pendulum positions
# =============================================================================

x_1 = l1 * np.sin(T1)
y_1 = -l1 * np.cos(T1)
x_2 = x_1 + l2 * np.sin(T2)
y_2 = y_1 - l2 * np.cos(T2)

x_3 = l3 * np.sin(T3)
y_3 = -l3 * np.cos(T3)
x_4 = x_3 + l4 * np.sin(T4)
y_4 = y_3 - l4 * np.cos(T4)


# =============================================================================
#  Plots two static graphs, one containing the angles and another the momenta of
#  the pendulums, both as a function of time, leaves a panel blank on the right 
#  side to hold the pendulum animation
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))


axes[0].plot(t, T1, label='Pend. 1: Angle 1', color = '#43AA8B')
axes[0].plot(t, T2, label='Pend. 1: Angle 2', color = 'steelblue')
axes[0].plot(t, T3, label='Pend. 2: Angle 1', color = '#F4A261')
axes[0].plot(t, T4, label='Pend. 2: Angle 2', color = '#F94144')
axes[0].set_xlabel(' Time (s)')
axes[0].set_ylabel('Angle (Rad)')
axes[0].set_title('Pendulum Angles')
axes[0].legend()
axes[0].grid(linestyle='--', linewidth=0.5)

axes[1].plot(t, P1, label='Pend. 1: Momentum 1', color = '#43AA8B')
axes[1].plot(t, P2, label='Pend. 1: Momentum 2', color = 'steelblue')
axes[1].plot(t, P3, label='Pend. 2: Momentum 1', color = '#F4A261')
axes[1].plot(t, P4, label='Pend. 2: Momentum 2', color = '#F94144')
axes[1].set_xlabel(' Time (s)')
axes[1].set_ylabel('Momentum (kg*m/s)')
axes[1].set_title('Pendulum Momenta')
axes[1].legend()
axes[1].grid(linestyle='--', linewidth=0.5)

# defines the x and y boundaries of the panel which holds the animation by 
# calculating values 10% greater than the overal length of the pendulums
lim = ( l1 + l2 ) * 1.1
axes[2].set_xlim(-lim, lim)
axes[2].set_ylim(-lim, lim)
axes[2].set_aspect('equal')
axes[2].set_title('Double Pendulums')
axes[2].grid(linestyle='--', linewidth=0.5)

# defines the pendulum rods and each of the trails from the lower
# masses to be used in the animation panel
TRAIL = 500 # defines the number of past frames for which the trail is present
trail1, = axes[2].plot([], [], 'c-', lw=1, alpha=0.5) 
trail2, = axes[2].plot([], [], 'r-', lw=1, alpha=0.5)
rod1,   = axes[2].plot([], [], 'o-', color='steelblue', lw=2, markersize=8)
rod2,   = axes[2].plot([], [], 'o-', color='#F94144', lw=2, markersize=8)

# initializes the pendulums rods and trails at the beginning of each animation
def init():
    trail1.set_data([], [])
    trail2.set_data([], [])
    rod1.set_data([], [])
    rod2.set_data([], [])
    return trail1, trail2, rod1, rod2

# Updates and redraws the rod positions and the trail once per frame
def update(i):
    start = max(0, i - TRAIL)
    trail1.set_data(x_2[start:i+1], y_2[start:i+1])
    trail2.set_data(x_4[start:i+1], y_4[start:i+1])
    rod1.set_data([0, x_1[i], x_2[i]], [0, y_1[i], y_2[i]])
    rod2.set_data([0, x_3[i], x_4[i]], [0, y_3[i], y_4[i]])
    return trail1, trail2, rod1, rod2

# =============================================================================
# Plots the static graphs and runs the animations using the function FuncAnimation, 
# total number of frames = Nt+1, interval=10 means the delay between each frame is 
# set to 10ms and blit=True tells the function only to plot what has changed since 
# the frame has been updated last, this allows the animation to run smoothly by 
# reducing the computational load
# =============================================================================

ani = animation.FuncAnimation(fig, update, frames=Nt+1, init_func=init, interval=10, blit=True)

plt.tight_layout() # formats the plot panels to have smaller spacing between plots
plt.show()