# Research 1

import numpy as np

# Inputs Enter Here
M1 = 2.0 # Mach 1 Entry
x = 1 # x Coordinate Entry
y = 1 # y Coordinate Entry
gamma = 1.4

def position(x,y):
    # Calculates the angle and entropy angle
    a = np.arctan2(y, x)
    a = (a + 2 * np.pi) % np.pi
    a_s = np.arctan(1/(m * (1/np.tan(a))))
    return a, a_s

def calculate_m(M1,gamma):
    # Solve for m using M1
    m = ((gamma + 1) * M1**2)/(2 + ((gamma - 1) * M1**2))
    return m

def calculate_M2(M1,gamma):
    # Solve for M2 using M1
    M2 = np.sqrt((2 + ((gamma -1)*(M1**2)))/((2*gamma*(M1**2)) - (gamma -1)))
    return M2

def calculate_decay_rate(M2,m,a):
    # Solve for Decay Rate, Zeta, and Pressure Wave Angle (ap). Values found if regime is propagative or nonpropagative
    ac = np.arctan(1/(np.sqrt((1 - (M2**2))/(m * M2)))) # Critical Angle
    acp = np.arccos(-M2) # Resulting Angle of Pressure 

    if a > 0 and a < ac:
        # Propagative Regime
        decay_rate = 0 
        ap = np.arctan(1/((1/np.tan(acp)) * ((np.tan(ac)/np.tan(a)) - (1/M2**2) * np.sqrt((np.tan(ac)/np.tan(a))**2 - 1) )))
        Zeta = 1

    elif a > ac and a < np.pi/2:
        # NonPropagative Regime both slow and fast wave
        ap = np.arctan(1/(((1/np.tan(acp)) * (1/np.tan(a)))/(1/np.tan(ac))))
        decay_rate = (abs((1/np.tan(acp))*np.sin(ap))/M2)*np.sqrt(1-(np.tan(ac)/np.tan(a))**2)
        Zeta = np.sqrt(1 - decay_rate**2 + 2*1J*decay_rate*np.cos(ap))

    return ap, decay_rate, Zeta

def calculate_decay_rate_prime(M1,M2,m,a,aPrime):
    # Solve for Decay Rate, Zeta, and Pressure Wave Angle (ap). Values found if regime is propagative or nonpropagative
    ac = np.arctan(1/(np.sqrt((1 - (M2**2))/(m * M2)))) # Critical Angle
    acp = np.arccos(-M2) # Resulting Angle of Pressure 

    # Upper and Lower angles of critical region (Acoustic Waves)
    acl = bisection_solve(equation_to_solve(a,-1, M1, M2, m), 0.000000001, np.pi/2, -1, 1e-6, M1, M2, m) # Find acl
    acu = bisection_solve(equation_to_solve(a,1, M1, M2, m), 0.000000001, np.pi/2, 1, 1e-6, M1, M2, m) # Find acu

    if a > 0 and a < ac:
        # Propagative Regime
        if a > acu and a < np.pi:
            # Slow wave Propagative regime. No decay rate. Solve for angle of pressure
            decay_rate = 0
            ap = np.arctan(1/((1/np.tan(acp)) * ((np.tan(ac)/np.tan(aPrime)) + (1/M2**2) * np.sqrt((np.tan(ac)/np.tan(aPrime))**2 - 1) )))
            Zeta = 1

        else:
            # Fast wave Propagative regime. No decay rate. Solve for angle of pressure
            decay_rate = 0
            ap = np.arctan(1/((1/np.tan(acp)) * ((np.tan(ac)/np.tan(aPrime)) - (1/M2**2) * np.sqrt((np.tan(ac)/np.tan(aPrime))**2 - 1) )))
            Zeta = 1

    elif a > ac and a < np.pi/2:
        # NonPropagative Regime both slow and fast wave
        ap = np.arctan(1/(((1/np.tan(acp)) * (1/np.tan(aPrime)))/(1/np.tan(ac))))
        decay_rate = (abs((1/np.tan(acp))*np.sin(ap))/M2)*np.sqrt(1-(np.tan(ac)/np.tan(aPrime))**2)
        Zeta = np.sqrt(1 - decay_rate**2 + 2*1J*decay_rate*np.cos(ap))

    return ap, decay_rate, Zeta

def calculate_aPrime(M1,a):
    # Find alpha prime. A new angle that is related to alpha
    aPrime = np.arctan(1/((1/np.tan(a)) + (1/(M1 * np.sin(a)))))
    return aPrime

def equation_to_solve(a, sign, M1, M2, m):
    # Equation that is the relationship between alpha and acl and acu
    ans = (1/np.tan(a)) + 1 / (M1 * np.sin(a)) - sign * np.sqrt(1 - M2**2) / (m * M2)
    return ans

def bisection_solve(equation, a_min, a_max, sign, tol, M1, M2, m):
    # Method of Solving for acu and acl becuase there is no exact solultion.
    while abs(a_max - a_min) > tol:
        a_mid = (a_min + a_max) / 2
        if equation_to_solve(a_min, sign, M1, M2, m) * equation_to_solve(a_mid, sign, M1, M2, m) < 0:
            a_max = a_mid
        else:
            a_min = a_mid
    return a_min

def make_A(a,a_s,ap,decay_rate,Zeta,M1,M2,m,gamma):
    # Used to build A matrix. This is used to solve for all following values.
    A = np.array([[np.sin(a_s**2), -1, (1/gamma) + ((np.cos(ap) + decay_rate*1J)/(gamma * M2 * Zeta)), 1J*(m - 1) * np.cos(a)],
              [2 * np.sin(a_s**2), -1, ((M2**2 + 1)/(gamma * M2**2)) + ((2 * np.cos(ap) + 1J * decay_rate)/(gamma * M2 * Zeta)), 0],
              [-np.cos(a_s**2), 0, (np.sin(ap))/(gamma * M2 * Zeta), 1J * (1 - m) * np.sin(a)],
              [np.sin(a_s**2), (1/((gamma - 1) * M2**2)), (1 / (gamma * M2**2)) + ((np.cos(ap) + 1J * decay_rate)/(gamma * M2 * Zeta)), 1J * m * (1 - m)*np.cos(a)]])
    return A

m = calculate_m(M1,gamma) # Solve m
M2 = calculate_M2(M1,gamma) # Solve M2
a, a_s = position(x,y) # Solve for angles
ap, decay_rate, Zeta = calculate_decay_rate(M2,m,a) #Solve Other flow properties 

A = make_A(a,a_s,ap,decay_rate,Zeta,M1,M2,m,gamma) # Build A matrix

b = np.array([-1, -m, 0, m**2 / ((gamma - 1) * M1**2)]).reshape((-1, 1)) # b matrix of equation A18

[Zsv, Zss, Zsp, Zsx] = np.linalg.solve(A, b) # Solve for Z vaules of equation A18

print("Zsv:", Zsv) # Print Zsv
print("Zss:", Zss) # Print Zss
print("Zsp:", Zsp) # Print Zsp
print("Zsx:", Zsx) # Print Zsx

# Appendix A2
 # b matrix of equation A20
c = np.array([np.sin(a), 2 * m * np.sin(a), -m * np.cos(a), m**2 * np.sin(a)]).reshape((-1,1))

[Zvv, Zvs, Zvp, Zvx] = np.linalg.solve(A,c) # Solve for Z vaules of equation A20

print("Zsv:", Zvv) # Print Zsv
print("Zss:", Zvs) # Print Zss
print("Zsp:", Zvp) # Print Zsp
print("Zsx:", Zvx) # Print Zsx

# Appendix A3
aPrime = calculate_aPrime(M1,a) # Find Alpha Prime
# Solve for new flow properties using aplha prime instead of alpha
del ap, decay_rate, Zeta
ap, decay_rate, Zeta = calculate_decay_rate_prime(M1,M2,m,a,aPrime)

# Build new A matrix with alpha prime instead of alpha
APrime = make_A(aPrime,a_s,ap,decay_rate,Zeta,M1,M2,m,gamma)

# b matrix of equation A27
d = np.array([
    (1/gamma) + (np.cos(a)/(gamma * M1)),
    (m/gamma) * (((M1**2 + 1)/M1**2) + ((2 * np.cos(a))/M1)),
    (m * np.sin(a))/(gamma * M1),
    (m**2/gamma) * ((1/M1**2) + (np.cos(a)/M1))
])

[Zpv, Zps, Zpp, ZpxSol] = np.linalg.solve(A,d) # Solve for Z vaules of equation A27
Zpx = ZpxSol / (np.sin(a)/np.sin(aPrime)) # Solve for Zpx vaule of equation A27

print("Zpv:", Zpv) # Print Zpv
print("Zps:", Zps) # Print Zps
print("Zpp:", Zpp) # Print Zpp
print("Zpx:", Zpx) # Print Zpx