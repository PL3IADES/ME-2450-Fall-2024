
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def effectT(temp):
    # Determine effect of temperature
    tempE = -0.35968+0.10789*temp-0.00214*np.square(temp)
    return tempE

def effectAir(temp):
    # Determine effect of air on infection rate
    if (temp>0) and (temp<35):
        tempBeta = 0.000241*np.power(temp,2.06737)*np.power(35-temp,0.72859)
    else:
        tempBeta = 0
    return tempBeta

def SLIRP(Si, Li, Ii, Ri, Pi, Bi, params):
    # Parse input params
    [beta,muL,muI,e,Ap,T,tDay] = params
    
    # Calculate derivatives
    dPbdt = (0.1724*Bi-0.0000212*np.square(Bi))*effectT(T)
    dPldt = 1.33*(tDay+30)*effectT(T)
    dPdt = dPbdt+dPldt
    dSdt = -1.0*beta*Si*Ii+dPdt/Ap
    dLdt = Si*Ii-np.power(muL,-1.0)*Li+e
    dIdt = np.power(muL,-1.0)*Li-np.power(muI,-1.0)*Ii
    dRdt = np.power(muI,-1.0)*Ii
    
    # Return derivatives
    return dSdt, dLdt, dIdt, dRdt, dPdt, dPbdt

def plot(tSpan, S, L, I, R, B, params):
    # Plot data
    plt.figure(figsize=(8,8))
    plt.plot(tSpan,S+L+I+R,label="Total Population")
    plt.plot(tSpan,B,label="Berry Population")
    plt.plot(tSpan,S,label="Susceptible")
    plt.plot(tSpan,L,label="Latent")
    plt.plot(tSpan,I,label="Infected")
    plt.plot(tSpan,R,label="Removed")
    plt.xlabel('Time [Days]',fontsize=14)
    plt.ylabel('Population (Fraction of Initial)',fontsize=14)
    titleStr = "SLIR Model System for beta,max="+str(params[0])+", muL,min="+str(params[1])
    plt.title(titleStr,fontsize=14)
    plt.grid()
    plt.legend()
    plt.show()
    
def rungeKutta(ode, tSpan, S, L, I, R, P, B, params, temp):
    # Implement runge-kutta
    h = tSpan[1]-tSpan[0]
    [betaMax,muLMin,muI,e,Ap] = params
    for idx, t in enumerate(tSpan):
        if not idx == 0:
            # Define parameters
            beta = betaMax*effectAir(temp[idx])
            if len(temp)-idx>=muLMin+1:
                effectSum = 0
                for effectIdx in range(idx,len(temp)):
                    effectSum += effectAir(temp[effectIdx+int(muLMin)])
                    if effectSum >= np.power(muLMin,-1.0):
                        break
                muL = np.power(effectSum,-1.0)+muLMin
            else:
                muL = muLMin
            newParams = [beta,muL,muI,e,Ap,temp[idx],t]
            newParams2 = [beta,muL,muI,e,Ap,temp[idx],t+0.5*h]
            
            # First pass
            kS1, kL1, kI1, kR1, kP1, kB1 = ode(S[idx-1], L[idx-1], I[idx-1], R[idx-1], P[idx-1], B[idx-1], newParams)
            S1 = S[idx-1]+0.5*kS1*h
            L1 = L[idx-1]+0.5*kL1*h
            I1 = I[idx-1]+0.5*kI1*h
            R1 = R[idx-1]+0.5*kR1*h
            P1 = P[idx-1]+0.5*kP1*h
            B1 = B[idx-1]+0.5*kB1*h
            
            # Second pass
            kS2, kL2, kI2, kR2, kP2, kB2 = ode(S1, L1, I1, R1, P1, B1, newParams2)
            S2 = S[idx-1]+0.5*kS2*h
            L2 = L[idx-1]+0.5*kL2*h
            I2 = I[idx-1]+0.5*kI2*h
            R2 = R[idx-1]+0.5*kR2*h
            P2 = P[idx-1]+0.5*kP2*h
            B2 = B[idx-1]+0.5*kB2*h
            
            # Third pass
            kS3, kL3, kI3, kR3, kP3, kB3 = ode(S2, L2, I2, R2, P2, B2, newParams2)
            S3 = S[idx-1]+kS3*h
            L3 = L[idx-1]+kL3*h
            I3 = I[idx-1]+kI3*h
            R3 = R[idx-1]+kR3*h
            P3 = P[idx-1]+kP3*h
            B3 = B[idx-1]+kB3*h
            
            # Fourth pass
            kS4, kL4, kI4, kR4, kP4, kB4 = ode(S3, L3, I3, R3, P3, B3, newParams)
            
            # Update SLIRP
            S[idx] = S[idx-1]+(kS1+2*kS2+2*kS3+kS4)*h/6
            L[idx] = L[idx-1]+(kL1+2*kL2+2*kL3+kL4)*h/6
            I[idx] = I[idx-1]+(kI1+2*kI2+2*kI3+kI4)*h/6
            R[idx] = R[idx-1]+(kR1+2*kR2+2*kR3+kR4)*h/6
            P[idx] = P[idx-1]+(kP1+2*kP2+2*kP3+kP4)*h/6
            B[idx] = B[idx-1]+(kB1+2*kB2+2*kB3+kB4)*h/6
    
    return S, L, I, R, P, B


# Init system parameters
muI = 10.0
e = 0.001
Ap = 5000.0
Pi = 1.33*np.power(30,2)*effectT(15)

# Init iterating parameters
betaMaxVals = [1.0,0.5,2.0]
muLMinVals = [6.0,8.0,10.0]

# Import data
data = sio.loadmat("EnvironmentalForcing.mat")
temp = np.array(data['T'][0])
tSpan = np.array(data['tspan'][0])

# Iterate and implement runge-kutta
for betaMax in betaMaxVals:
    for muLMin in muLMinVals:
        # Define population arrays
        S = np.zeros(len(tSpan),dtype=float)
        L = np.zeros(len(tSpan),dtype=float)
        I = np.zeros(len(tSpan),dtype=float)
        R = np.zeros(len(tSpan),dtype=float)
        P = np.zeros(len(tSpan),dtype=float)
        B = np.zeros(len(tSpan),dtype=float)
        
        # Set init conditions
        S[0] = Pi/Ap
        L[0] = 0.01*S[0]
        I[0] = 0.0
        R[0] = muI*I[0]
        P[0] = Pi/Ap
        B[0] = 1.0/Ap
        
        # Solve SLIRP model
        params = [betaMax,muLMin,muI,e,Ap]
        S, L, I, R, P, B = rungeKutta(SLIRP, tSpan, S, L, I, R, P, B, params, temp)
        plot(tSpan, S, L, I, R, B, params)