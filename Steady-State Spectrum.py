import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

def get_steady_state_max_zpd(N, w):
    N = int(N/2)+3
    centre = -2 

    u = 0.01 
    t = 0
    dt = 0.01 #Somewhat optimal value
    step = 0
       
    revolution_time = 2*np.pi/w
    revolution_steps = int(revolution_time/dt)
    
    #Variables to collect data
    last_revolution_shift = []
    estimates = [0, 0, 0]
    reached_steady_state = False
    
    z = np.zeros((N,N))
    v = np.zeros((N,N))
            
    while not reached_steady_state:
        #Increase w and t, set position of the central oscillator
        step += 1
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
        last_revolution_shift.append(shift)
                  
        if step%revolution_steps == 0:
            estimates = [estimates[1], estimates[2], max(last_revolution_shift)]
            last_revolution_shift = []   
            if np.std(estimates)/np.mean(estimates) < 0.001:
                reached_steady_state = True
                
    return np.mean(estimates)
    
def get_precise_spectrum_in_parallel(N, wmin, wmax, dw):
    from ipyparallel import Client
    rc = Client()
    dview = rc[:]
    push_get_spectrum=dict(get_steady_state_max_zpd=get_steady_state_max_zpd)
    dview.push(push_get_spectrum)
    dview.execute('import numpy as np')
    
    if wmin == 0:
        wmin += dw
    
    points = np.arange(wmin, wmax+dw, dw)
    func = lambda w: get_steady_state_max_zpd(N, w)
    spectrum = dview.map(func, [float(w) for w in points])
    
    return points, spectrum.get()
    
def get_precise_spectrum(N, wmin, wmax, dw):
    if wmin == 0:
        wmin += dw
        
    func = lambda w: get_steady_state_max_zpd(N, w)
    points = np.arange(wmin, wmax+dw, dw)
    spectrum = [func(w) for w in tqdm(points)]
    
    return points, spectrum

#INPUT VALUES    
N = 3
wmin = 0
wmax = 4
dw = 0.1

spectrum = get_precise_spectrum(N, wmin, wmax, dw)
#spectrum = get_precise_spectrum_in_parallel(N, wmin, wmax, dw) #Use that line instead of the previous for parallel computing

plt.plot(spectrum[0], spectrum[1])
plt.title('{}x{} Steady-State Spectrum'.format(N,N))
plt.xlabel('Frequency')
plt.ylabel('Steady-State Max Shift')
plt.show()
