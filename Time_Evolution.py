import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

def get_time_evolution(N, wmin, wmax, total_time):
    
    #N must be odd in order to locate the central oscillator
    if N%2 == 0:
        raise ValueError('N must be odd')
    else:
        N = int(N/2)+3
        centre = -2 

    t = 0
    dt = 0.01 #Somewhat optimal value
    w = wmin
    dw = (wmax-wmin)/total_time*dt
    u = 0.01
    
    num_steps = round(total_time/dt)
    
    #Empty lists to collect data
    frequencies = []
    shifts = []
    
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
        zE_next[-1]=zE_next[:,-3]
        zE_next[:,0]=zE_next[:,1]
        zE_next[:,-1]=zE_next[-3]
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
        z[-1]=z[:,-3]
        z[:,0]=z[:,1]
        z[:,-1]=z[-3]
        
        #Calculate and write zero plane displacement
        shift = np.mean(z[1:-1,1:-1]**2)/(N-2)**2
        shifts.append(shift)
        frequencies.append(w)
        
    return frequencies, shifts

#Input Values
N = 3
wmin = 0
wmax = 2
total_time = 10

time_evolution = get_time_evolution(N, wmin, wmax, total_time)
plt.plot(time_evolution[0], time_evolution[1])
plt.xlim(wmin,wmax)
plt.xlabel('Frequency')
plt.ylabel('Mean Squared Shift')
plt.title('{0}x{0} Plate Time Evolution\nw: from {1} to {2}'.format(N, wmin, wmax))
plt.show()
