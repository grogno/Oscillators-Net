
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

def get_spectrum(N, wmin, wmax, total_time):
    
    #N must be odd in order to locate the central oscillator
    if N%2 == 0:
        raise ValueError('N must be odd')
    else:
        N = N+2
        centre = int((N-1)/2)

    t = 0
    dt = 0.001 #Somewhat optimal value
    w = wmin
    dw = (wmax-wmin)/total_time*dt
    u = 0.01
    
    num_steps = round(total_time/dt)
    
    #Empty lists to collect data
    frequencies = []
    displacements = []
    
    z = np.zeros((N,N))
    v = np.zeros((N,N))
        
    for step in tqdm(range(num_steps)):
        #Increase w and t, set position of the central oscillator
        w += dw
        t += dt
        z[centre, centre] = np.sin(w*t)
        
        #Find estimates of next v and z
        a = np.roll(z,1,1) + np.roll(z,-1,1) + np.roll(z,1,0) + np.roll(z,-1,0) - 4*z - u*v
        vE_next = v + a*dt
        zE_next = z + v*dt
        
        #Clean up boundaries and correct position of the central oscillator
        zE_next[0]=zE_next[1]
        zE_next[-1]=zE_next[-2]
        zE_next[:,0]=zE_next[:,1]
        zE_next[:,-1]=zE_next[:,-2]
        zE_next[centre, centre] = np.sin(w*(t+dt))
        
        #Find next v and z using Heun method        
        aE_next = (np.roll(zE_next,1,1) + np.roll(zE_next,-1,1) + np.roll(zE_next,1,0) + np.roll(zE_next,-1,0)
                   - 4*zE_next - u*vE_next)
        vH_next = v + 0.5*(a+aE_next)*dt
        zH_next = z + 0.5*(v+vE_next)*dt
        
        #Set v and z for the next step
        v = vH_next
        z = zH_next
        
        #Clean up boundaries
        z[0]=z[1]
        z[-1]=z[-2]
        z[:,0]=z[:,1]
        z[:,-1]=z[:,-2]
        
        #Calculate and write zero plane displacement
        displacement = np.linalg.norm(z)
        displacements.append(displacement)
        frequencies.append(w)
        
    return frequencies, displacements

N = 3
wmin = 0
wmax = 2
total_time = 10000
spectrum = get_spectrum(N, wmin, wmax, total_time)
plt.plot(spectrum[0], spectrum[1])
plt.xlim(xmin,xmax)
plt.xlabel('Frequency')
plt.ylabel('Zero Plane Displacement')
plt.title(str(N)+'x'+str(N)' Net Spectrum')
plt.show()
