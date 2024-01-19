import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

deg = 180/np.pi

gamma = 1.4
M1 = 4
beta = 90 / deg

def th_tbm(m,b, gamma): #explicit form of the theta-beta-M relation
    th = np.arctan((2*(m*np.sin(b))**2-2)/(m**2*(gamma+np.cos(2*b))+2)/np.tan(b))
    return abs(th)

def calculate_M2(M1,gamma,beta):
    if beta == 90 * np.pi/180:
        M2 = np.sqrt((M1**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * M1**2 - 1))
        PR = 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)
        m = ((gamma + 1) * M1**2)/(2 + ((gamma - 1) * M1**2))
        TR = ((2 * gamma * M1**2 - (gamma - 1)) * ((gamma - 1) * M1**2 + 2))/((gamma + 1)**2 * M1**2)
        return M2, PR, m, TR
    
    elif beta < 90 * np.pi/180:
        th = th_tbm(M1, beta, gamma)
        Mn = M1 * np.sin(beta)
        M2 = (np.sqrt((Mn**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * Mn**2 - 1)))/np.sin(beta - th)
        PR = 1 + (2 * gamma / (gamma + 1)) * (Mn**2 - 1)
        m = ((gamma + 1) * Mn**2)/(2 + ((gamma - 1) * Mn**2))
        TR = ((2 * gamma * Mn**2 - (gamma - 1)) * ((gamma - 1) * Mn**2 + 2))/((gamma + 1)**2 * Mn**2)
        return M2, PR, m, TR

def alpha_s(a,m):
    # Calculates the angle and entropy angle
    
    a_s = np.arctan(np.tan(a)/m)
    # while a_s < 0:
    #     a_s = a_s + np.pi
    return a_s

def Crit_Angles(M1, gamma,beta):

    M2, PR, m, TR = calculate_M2(M1,gamma,beta)

    aM = np.arccos(-1/M1)
    acu = bisection_solve(0.000000001, np.pi, -1, 1e-6, M1, M2, m) # Find acl
    acl = bisection_solve(0.000000001, np.pi, 1, 1e-6, M1, M2, m) # Find acu
    ac = np.arctan((m * M2)/(np.sqrt(1 - M2**2)))

    return aM, acu, acl, ac

def equation_to_solve(a, sign, M1, M2, m):
    # Equation that is the relationship between alpha and acl and acu
    eq = (1/np.tan(a)) + 1 / (M1 * np.sin(a)) - sign * np.sqrt(1 - M2**2) / (m * M2)
    return eq

def bisection_solve(a_min, a_max, sign, tol, M1, M2, m):
    # Method of Solving for acu and acl becuase there is no exact solultion.
    while abs(a_max - a_min) > tol:
        a_mid = (a_min + a_max) / 2
        if equation_to_solve(a_min, sign, M1, M2, m) * equation_to_solve(a_mid, sign, M1, M2, m) < 0:
            a_max = a_mid
        else:
            a_min = a_mid
    return a_min

def calculate_decay_rate_prime(M1,M2,m,a,aPrime):
    deg = 180 / np.pi
    # Solve for Decay Rate, Zeta, and Pressure Wave Angle (ap). Values found if regime is propagative or nonpropagative
    ac = np.arctan((m * M2)/(np.sqrt(1 - M2**2))) # Critical Angle
    acp = np.arccos(-M2) # Resulting Angle of Pressure 
    # Upper and Lower angles of critical region (Acoustic Waves)
    aM = np.arccos(-1/M1)
    acu = bisection_solve(0.000000001, np.pi, -1, 1e-6, M1, M2, m) # Find acl
    acl = bisection_solve(0.000000001, np.pi, 1, 1e-6, M1, M2, m) # Find acu

    if  a > 0 and a < acl:
        ap = np.arctan(np.tan(acp)/((np.tan(ac)/np.tan(aPrime)) - ((1/M2) * np.sqrt((np.tan(ac)/np.tan(aPrime))**2 - 1))))
        if ap < 0:
            ap = ap + np.pi
        decay_rate = 0
        Zeta = 1

    elif a > acu and a < np.pi:
        ap = np.arctan(np.tan(acp)/((np.tan(ac)/np.tan(aPrime)) + ((1/M2) * np.sqrt((np.tan(ac)/np.tan(aPrime))**2 - 1))))
        while ap < np.pi:
            ap = ap + np.pi
        decay_rate = 0
        Zeta = 1

    return ap, decay_rate, Zeta

def calculate_aPrime(M1,a):
    # Find alpha prime. A new angle that is related to alpha
    if a == 0:
        return 0
    
    else:
        aPrime = np.arctan(1/((1/np.tan(a)) + (1/(M1 * np.sin(a)))))
        if aPrime < 0:
            aPrime = aPrime + np.pi

        return aPrime
    
def equations_crit(phi):
    # Define your system of equations here
    eq1 = np.sqrt((V/np.tan(phi[0]) + 1/np.sin(phi[0]))**2 + (V-U)**2) - np.sqrt(TR)
    return [eq1]

def moore(phi, M1, gamma, beta):
    M2, PR, m, TR = calculate_M2(M1,gamma,beta)

    V = M1
    U = -V/m + V

    PhiVort = np.arctan(((m/V)*(V/np.tan(phi) + 1/np.sin(phi))))
    #if PhiVort < 0:
    #    PhiVort = PhiVort + np.pi
    """
    Cx, Cy = (V - U), 0 
    r = np.sqrt(TR)
    Px, Py =  V, (V/np.tan(phi) + 1/np.sin(phi))

    dx, dy = Px-Cx, Py-Cy
    dxr, dyr = -dy, dx
    d = np.sqrt(dx**2+dy**2)
    if d >= r :
        rho = r/d
        ad = rho**2
        bd = rho*np.sqrt(1-rho**2)
        T1x = Cx + ad*dx + bd*dxr
        T1y = Cy + ad*dy + bd*dyr
        T2x = Cx + ad*dx - bd*dxr
        T2y = Cy + ad*dy - bd*dyr
    
    if phi <= CritLower:
        PhiRefract = np.arctan((Px - T1x)/(Py - T1y))

    elif phi >= CritUpper:
        PhiRefract = np.arctan((Px - T2x)/(Py - T2y))
        if PhiRefract < 0:
            PhiRefract = PhiRefract + np.pi
    """
    PhiRefract = 1
    return PhiVort, PhiRefract

M1 = 1 + 1e-9
acuL = []
aclL = []
M = []

while M1 < 7:

    M2, PR, m, TR = calculate_M2(M1,gamma,beta)

    V = M1
    U = -V/m + V


    ## Moore Section
    CritLower = sp.fsolve(equations_crit,np.pi/3)
    CritUpper = sp.fsolve(equations_crit, np.pi - 1e-2)
    PhiVortL = np.arctan( 1/(V/m) * 1/(np.tan(CritLower)/V))
    PhiVortU = np.arctan( 1/(V/m) * 1/(np.tan(CritUpper)/V))

    acuL.append(PhiVortU * deg)    
    aclL.append(PhiVortL * deg)
    M.append(M1)

    M1 = M1 + 0.1

# angle = []
# FabreS = []
# MooreS = []
# psi = 0.001

# while psi < CritLower:
#     FabreS.append(alpha_s(psi,m))
#     a,b = moore(psi, M1, gamma, beta)
#     MooreS.append(a)
#     angle.append(psi)
#     psi = psi + 0.01


psi_range = np.linspace(0 + 1e-6,CritLower,20)
f_v = alpha_s(psi_range,m)
m_v = moore(psi_range,M1,1.4,np.pi/2)[0]



# plt.plot(psi_range*deg,f_v*deg)
# plt.plot(psi_range*deg,m_v*deg)
plt.plot(M,acuL)
plt.plot(M,aclL)
plt.xlabel('M1')
plt.ylabel("Critical Angle (Deg)")
plt.legend(['Upper Critical Vorticity Angle','Lower Critical Vorticity Angle'])
plt.grid()
plt.savefig('/workspaces/Research/Angle.png')
plt.show