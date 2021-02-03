# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 09:20:16 2020

@author: janousu
"""

import numpy as np
import matplotlib.pyplot as plt


gamma = 2/3     # [-]
Scmax = 3       # [mm]
kcan = 0.50     # [d-1]
Sfc = 25        # [mm]
Suzmax = 100    # [mm]   
alpha = 0.25    # [-]
ksat = 500      # [mm-1]
beta = 3        # [-]
ksz = 0.10      # [d-1]


""" DIAGNOSTIC VARIABLES USED IN THE FLUX PARAMETERIZATIONS """

# wetted fraction of the canopy (12)
def canopywetfrac(Sc, Scmax, gamma):
    if Sc < Scmax:
        Fwet = (Sc/Scmax)**gamma
    else:
        Fwet = 1
    return Fwet

# saturated fraction of the unsaturated zone (16)
def unsaturatedfrac(Suz, Suzmax, alpha):
    if Suz < Suzmax:
        Fsat = 1 - (1 - Suz/Suzmax)**alpha
    else:
        Fsat = 1
    return Fsat

# plotting Fwet and Fsat
Si_values  = np.arange(0,3.01,0.01)
Suz_values = np.arange(0,101,1)

check_fwet = []
check_fsat = []

for Si_value in Si_values:
    check_fwet.append(canopywetfrac(Si_value,  Scmax,  gamma))

for Suz_value in Suz_values:
    check_fsat.append(unsaturatedfrac(Suz_value, Suzmax, alpha))

# plot both
fig, ax = plt.subplots(1,2,figsize=(10,5))

# fwet
ax[0].plot(Si_values,check_fwet);
ax[0].set_xlabel('Canopy storage [mm]');
ax[0].set_ylabel('Wetted fraction [-]');

# fsat
ax[1].plot(Suz_values,check_fsat);
ax[1].set_xlabel('Unsaturated zone storage [mm]');
ax[1].set_ylabel('Saturated fraction [-]');

""" MODEL FLUXES """

# Evaporation from canopy (8)
def canopyevap(Ep, Fwet):
    Ec = Ep*Fwet
    return Ec

# Throughfall from canopy (9)
def canopythroughfall(P, Sc, Scmax):
    if Sc < Scmax:
        Qt = P*(Sc/Scmax)
    else:
        Qt = P
    return Qt

# Drainage from canopy (10)
def canopydrain(Sc, Scmax, kcan):
    if Sc < Scmax:
        Qc = 0
    else:
        Qc = kcan * (Sc - Scmax)
    return Qc

# Percolation from canopy (11)
def canopyperc(Qt, Qc):
    Pe = Qt + Qc
    return Pe


# Evaporation from unsat zone (13)
def unsatevap(Suz, Sfc, Ep, Fwet):
    if Suz < Sfc:
        Euz = Ep*(Suz/Sfc)*(1-Fwet)
    else:
        Euz = Ep*(1-Fwet)
    return Euz

# Overland flow from unsat zone (14)
def unsatrunoff(Pe, Fsat):
    Qo = Pe * Fsat
    return Qo

# Percolation from unsat zone (15)
def unsatperc(Suz, Sfc, ksat, Suzmax, beta):
    if Suz < Sfc:
        Qp = 0
    else:
        Qp = ksat * ((Suz-Sfc)/(Suzmax-Sfc))**beta
    return Qp

# Baseflow from sat zone (17)
def satbaseflow(ksz, Ssz):
    Qb = ksz*Ssz
    return Qb

# plotting the fluxes as a function of the state variables

# forcing
P  = 1 # [mm d-1]
Ep = 1 # [mm d-1]

# states
Si_values  = np.arange(0,4.51,0.01)
Suz_values = np.arange(0,141,1)
Ssz_values = np.arange(0,141,1)

# empties
check_ei  = []
check_qt  = []
check_qi  = []
check_euz = []
check_qo  = []
check_qp  = []
check_qb  = []

# generate canopy fluxes
for Si_value in Si_values:
    check_ei.append(canopyevap(Ep, canopywetfrac(Si_value,  Scmax,  gamma)) ) # Eq. 8
    check_qt.append(canopythroughfall(P, Si_value, Scmax) ) # Eq. 9
    check_qi.append(canopydrain(Si_value, Scmax, kcan) ) # Eq. 10
    
# generate unsaturated zone fluxes
for Suz_value in Suz_values:
    # unsat zone
    check_euz.append(unsatevap(Suz_value, Sfc, Ep, 0)) # Eq. 13; Fwet = 0
    check_qo.append(unsatrunoff(P, unsaturatedfrac(Suz_value, Suzmax, alpha)) ) # Eq. 14; Pe = P; 
    check_qp.append(unsatperc(Suz_value, Sfc, ksat, Suzmax, beta) ) # Eq. 15
        
# generate saturated zone fluxes
for Ssz_value in Ssz_values:
    check_qb.append(satbaseflow(ksz, Ssz_value) )# Eq. 17


# plot all
fig,ax = plt.subplots(3,3,figsize=(15,15))


# canopy
ax[0,0].plot(Si_values,check_ei);
ax[0,0].set_xlabel('Canopy storage [mm]');
ax[0,0].set_ylabel('Canopy evaporation [mm d-1]');

ax[0,1].plot(Si_values,check_qt);
ax[0,1].set_xlabel('Canopy storage [mm]');
ax[0,1].set_ylabel('Canopy throughfall [mm d-1]');

ax[0,2].plot(Si_values,check_qi);
ax[0,2].set_xlabel('Canopy storage [mm]');
ax[0,2].set_ylabel('Canopy drainage [mm d-1]');

# unsaturated zone
ax[1,0].plot(Suz_values,check_euz);
ax[1,0].set_xlabel('Unsaturated storage [mm]');
ax[1,0].set_ylabel('Soil transpiration [mm d-1]');

ax[1,1].plot(Suz_values,check_qo);
ax[1,1].set_xlabel('Unsaturated storage [mm]');
ax[1,1].set_ylabel('Soil surface runoff [mm d-1]');

ax[1,2].plot(Suz_values,check_qp);
ax[1,2].set_xlabel('Unsaturated storage [mm]');
ax[1,2].set_ylabel('Soil percolation [mm d-1]');

# saturated zone
ax[2,0].plot(Ssz_values,check_qb);
ax[2,0].set_xlabel('Saturated storage [mm]');
ax[2,0].set_ylabel('Soil baseflow [mm d-1]');

ax[2,1].set_visible(False)
ax[2,2].set_visible(False)

fig.tight_layout()



'''
SOLVE ODES
'''
# Calculate RHS of ODEs (Eq. 1-3)
def calc_fluxes(P, Ep, S, par, fix_uz):
    
    # unpack storages
    Si_value  = S[0]
    Suz_value = S[1]
    Ssz_value = S[2]
    
    # unpack parameters
    gamma  = par[0]
    Scmax  = par[1]
    kcan   = par[2]
    Sfc    = par[3]
    Suzmax = par[4]
    alpha  = par[5]
    ksat   = par[6]
    beta   = par[7]
    ksz    = par[8]
      
    # aux. functions
    Fwet = canopywetfrac(Si_value, Scmax,  gamma)
    Fsat = unsaturatedfrac(Suz_value, Suzmax, alpha)
    
    # canopy
    Ec = canopyevap(Ep, Fwet)
    Qt = canopythroughfall(P, Si_value, Scmax)
    Qc = canopydrain(Si_value, Scmax, kcan)

    # unsaturated zone
    Pe  = canopyperc(Qt, Qc)
    
    # build in a testing statement
    if fix_uz:
        Pe = 100
        Fwet = 0
    
    Euz = unsatevap(Suz_value, Sfc, Ep, Fwet)
    Qo  = unsatrunoff(Pe, Fsat)
    Qp  = unsatperc(Suz_value, Sfc, ksat, Suzmax, beta)
    
    # saturated zone
    Qb = satbaseflow(ksz, Ssz_value)
    
    # package into RHS
    rhs = np.array([P  - Ec   - Qt - Qc, # Eq. 1
                    Pe - Euz  - Qo - Qp, # Eq. 2
                    Qp - Qb]) # Eq. 3 
    
    # total flow
    q = Qo + Qb
    
    return rhs,q
    
# Solve the ODEs
def run_timestepping(P, Ep, par, initialStates, dt=1, fix_uz=0):
    
    # Initialize storages vectors
    Si  = [initialStates[0]]
    Suz = [initialStates[1]]
    Ssz = [initialStates[2]]
    Q   = [0]
    
    # Check if we're fudging UZ values
    if fix_uz:
        print('running with fixed Pe and Fwet values')
    
    # Find the number of time steps
    nTime = len(P) # assume length of Ep is the same
    
    # Initiate the timestepping
    for t in range(0,nTime):
        
        # Old states
        if t == 0: # very first time step
            states_old = initialStates
        else:
            states_old = states_new
                                
        # compute RHS
        rhs,q = calc_fluxes(P[t],Ep[t],states_old,par,fix_uz)
            
        # Explicit Euler
        states_new = states_old + rhs*dt
            
        # Perform some non-negativety checks
        states_new[states_new < 0] = 0
            
        # Calculate the average flux over the time step
        Q.append(q) 
            
        # save the states
        Si.append(states_new[0])
        Suz.append(states_new[1])
        Ssz.append(states_new[2])
        
    # output
    return Si,Suz,Ssz,Q


''' 
CANOPY
 '''

# Inputs 
P             = 100 # [mm d-1]
Ep            = 5 # [mm d-1]
dts           = [1,1/24,1/(12*24)] # [d]
data_length   = 10 # [d]
initialStates = np.array([0,0,0]) # [mm] 
par           = [gamma,
                 Scmax,
                 kcan,
                 Sfc,
                 Suzmax,
                 alpha,
                 ksat,
                 beta,
                 ksz]


fig,ax = plt.subplots(1,3,figsize=(15,5)) # open figure
fig.tight_layout()
i = 0

# loop
for dt in dts:
    
    print('Running with dt = ' + str(dt) + ' [d].')
    
    # make the forcing
    data_p = np.ones(int(data_length / dt))*P
    data_e = np.ones(int(data_length / dt))*Ep
    
    # model run
    Si,_,_,_ = run_timestepping(data_p,data_e,par,initialStates,dt)
    
    # plot
    t = np.arange(0,data_length,dt)
    t = np.append(t,t[-1]+dt)   
    ax[i].plot(t,Si)
    ax[i].set_title('dt = ' + str(round(dt,4)) + ' [d]')
    ax[i].set_ylabel('Canopy storage [mm]');
    i += 1
    
# freshen up the plot
plt.xlabel('time [d]');

plt.savefig('steadyState_can.png')


'''
UNSATURATED ZONE
'''

# Inputs 
P             = 100 # [mm d-1]
Ep            = 5 # [mm d-1]
dts           = [1,1/24,1/(12*24)] # [d]
data_length   = 10 # [d]
initialStates = np.array([0,0,0]) # [mm]
par           = [gamma,
                 Scmax,
                 kcan,
                 Sfc,
                 Suzmax,
                 alpha,
                 ksat,
                 beta,
                 ksz]
fix_uz        = 1 # toggle fixing of Pe and Fwet on

fig,ax = plt.subplots(1,3,figsize=(15,5)) # open figure
fig.tight_layout()
i = 0

# loop
for dt in dts:
    
    print('Running with dt = ' + str(dt) + ' [d].')
    
    # make the forcing
    data_p = np.ones(int(data_length / dt))*P
    data_e = np.ones(int(data_length / dt))*Ep
    
    # model run
    _,Suz,_,_ = run_timestepping(data_p,data_e,par,initialStates,dt,fix_uz)
    
    # plot
    t = np.arange(0,data_length,dt)
    t = np.append(t,t[-1]+dt)
    ax[i].plot(t,Suz)
    ax[i].set_title('dt = ' + str(round(dt,4)) + ' [d]')
    ax[i].set_ylabel('Unsaturated zone storage [mm]');
    i += 1
    
# freshen up the plot
plt.xlabel('time [d]');

plt.savefig('steadyState_uz.png')


'''
UNSATURATED ZONE WITH ADJUSTED PARAMETERS
'''

# Inputs 
P             = 100 # [mm d-1]
Ep            = 5 # [mm d-1]
dts           = [1,1/24,1/(12*24)] # [d]
data_length   = 10 # [d]
initialStates = np.array([0,0,0]) # [mm]
par           = [gamma,
                 Scmax,
                 kcan,
                 2.5,
                 10,
                 alpha,
                 ksat,
                 beta,
                 ksz]
fix_uz        = 1 # toggle fixing of Pe and Fwet on

fig,ax = plt.subplots(1,3,figsize=(15,5)) # open figure
fig.tight_layout()
i = 0

# loop
for dt in dts:
    
    print('Running with dt = ' + str(dt) + ' [d].')
    
    # make the forcing
    data_p = np.ones(int(data_length / dt))*P
    data_e = np.ones(int(data_length / dt))*Ep
    
    # model run
    _,Suz,_,_ = run_timestepping(data_p,data_e,par,initialStates,dt,fix_uz)
    
    # plot
    t = np.arange(0,data_length,dt)
    t = np.append(t,t[-1]+dt)
    ax[i].plot(t,Suz)
    ax[i].set_title('dt = ' + str(round(dt,4)) + ' [d]')
    ax[i].set_ylabel('Unsaturated zone storage [mm]');
    i += 1
    
# freshen up the plot
plt.xlabel('time [d]');

plt.savefig('steadyState_uz10.png')


''' 
RUN THE MODEL
'''

# Synthetic P 
Pmax     = 100 # [mm d-1]
t_bar    = 2 # [d]
sigma_t  = 0.24 # [-]
t_dur    = 10 # [d]
dt       = 1 / (12*24) # [d]
t        = np.arange(0,t_dur,dt)
data_p   = Pmax * np.exp(-1 * ((t_bar-t)/sigma_t)**2) # [mm d-1]
Ep       = 5 # [mm d-1]
data_e   = np.ones(int(data_length / dt))*Ep # [mm d-]1

plt.plot(t,data_p);
plt.xlabel('time [d]');
plt.ylabel('P [mm d-1]');


# Inputs
initialStates = np.array([0,10,1]) # [mm]
par           = [gamma,
                 Scmax,
                 kcan,
                 Sfc,
                 Suzmax,
                 alpha,
                 ksat,
                 beta,
                 ksz]


Si,Suz,Ssz,Q = run_timestepping(data_p,data_e,par,initialStates,dt)

fig,ax = plt.subplots(5,1,figsize=(15,15))

ax[0].plot(t,data_p);
ax[0].set_ylabel('P [mm d-1]');

tp = np.append(t,t[-1]+dt)
ax[1].plot(tp,Si)
ax[1].set_ylabel('Canopy storage [mm]');

ax[2].plot(tp,Suz)
ax[2].set_ylabel('Unsaturated storage [mm]');

ax[3].plot(tp,Ssz)
ax[3].set_ylabel('Saturated storage [mm]');

ax[4].plot(tp,Q)
ax[4].set_ylabel('Streamflow [mm d-1]');
ax[4].set_xlabel('Time [d]');

plt.savefig('modelSimulations.png')

#