
import numpy as np

def th_tbm(m,b, gamma): #explicit form of the theta-beta-M relation
    th = np.arctan((2*(m*np.sin(b))**2-2)/(m**2*(gamma+np.cos(2*b))+2)/np.tan(b))
    return abs(th)

def calculate_M2(M1,gamma,beta):
    if beta == 90 * np.pi/180:
        M2 = np.sqrt((M1**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * M1**2 - 1))
        PR = 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)
        m = ((gamma + 1) * M1**2)/(2 + ((gamma - 1) * M1**2))
        return M2, PR, m
    
    elif beta < 90 * np.pi/180:
        th = th_tbm(M1, beta, gamma)
        Mn = M1 * np.sin(beta)
        M2 = (np.sqrt((Mn**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * Mn**2 - 1)))/np.sin(beta - th)
        PR = 1 + (2 * gamma / (gamma + 1)) * (Mn**2 - 1)
        m = ((gamma + 1) * Mn**2)/(2 + ((gamma - 1) * Mn**2))
        return M2, PR, m


M1 = 8
gamma = 5/3
beta = 19 * np.pi/180

M2, PR, m = calculate_M2(M1,gamma,beta)
print(M2, PR, m)