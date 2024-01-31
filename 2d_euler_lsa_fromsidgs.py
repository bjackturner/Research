import numpy as np
from math import *
import scipy
# from scipy.sparse import coo_matrix, eye
# import scipy.sparse.linalg as lg
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

#Geometry and inputs
n = 256*2
lx = 0.1*2
cfl = 5

total_time_coeff = 1.5e-1
om = 100 #om_nondim = om_dim*Lx/Uinf
shock_loc = 0.5*lx
sponge_loc = 0.9*lx

minf = 8
rinf = 0.0619; cinf = np.sqrt(1.4*287.05*57.9); Uinf = minf*cinf;  tinf = 57.9
#________________________________
alpha = 50*np.pi/180
#--------------------------------

xf = np.linspace(0,lx,n-1)
xc = np.zeros((n)); xc[1:-1] = 0.5*(xf[:-1] + xf[1:])
dx = xc[2]-xc[1]; dsi = 1/dx
xc[0] = xc[1] - dx; xc[-1] = xc[-2] + dx
ife = np.zeros((xf.shape[0],2),dtype='int32')
ife[:,0] =     np.arange(n-1) 
ife[:,1] = 1 + np.arange(n-1)

#Fluid
mu = 1# Uinf*dx
l2m = 1#4*mu/3
kap = 1#2*mu
cv = 287.05/0.4

#Matrices and vectors
mx,N,inv = np.zeros((3,4,4),dtype='complex128')
A = np.zeros((n,n,4,4),dtype='complex128'); U = np.zeros((n,4),dtype='complex128')

def getprim(U):
    
    r = U[0]; u = U[1]/r; p = 0.4*(U[3]-r*u*u/2); h0 = (U[3] + p)/r ; a = np.sqrt(1.4*p/r); t = p/(287.05*r)
    return np.array([r,u,h0,a,t],dtype='complex128')

def get_facevars(Ui,Uii):
    si  = getprim(Ui); 
    sii = getprim(Uii); 
    sf  = 0.5*(si + sii)
    return sf[0],sf[1],sf[2],sf[3],sf[4],sf[3],0 #r,u,h0,a,t,at,e

def sgmab(a,b,s): #sigmoid ab
    return 0.5*((b + a) + (b - a)*s)
        

def con2T(r,u,t):
    cv = 287.05/0.4
    #N[0,0] = 1; N[1,0] = -u/r; N[1,1] = 1/r; N[2,0] = u**2/(2*r*cv) - t/r ; N[2,1] = -u/(r*cv); N[2,2] = 1/(r*cv)
    N[0,0] = 1; N[1,0] = -u/r; N[1,1] = 1/r; N[2,2] = 1/r; N[3,0] = u**2/(2*r*cv) - t/r ; N[3,1] = -u/(r*cv); N[3,3] = 1/(r*cv)
    return N

def con2p(r,u):
    #N[0,0] = 1; N[1,0] = -u/r; N[1,1] = 1/r; N[2,0] = 0.2*(u*u) ; N[2,1] = -u*0.4; N[2,2] = 0.4
    N[0,0] = 1; N[1,0] = -u/r; N[1,1] = 1/r; N[2,2] = 1/r; N[3,0] = 0.2*(u*u) ; N[3,1] = -u*0.4; N[3,3] = 0.4
    return N



shock_width = dx*1
sponge_width = 5*dx
fsh = 5
viscx = 0.5*np.tanh((xf-shock_loc+fsh*shock_width)/(0.5*fsh*shock_width)) - 0.5*np.tanh((xf-shock_loc-fsh*shock_width)/(0.5*fsh*shock_width))
viscx += 0.5 + 0.5*np.tanh((xf-sponge_loc+sponge_width)/(1.5*sponge_width))
#viscx *= 0 #REMOVE REMOVE REMOVE
viscx  += 1e-3
viscx[viscx>1] = 1
viscx[viscx<0] = 0
plt.plot(viscx); plt.plot(np.tanh((xf-shock_loc)/1e-12)); plt.show()
visc_coeff = 1

#Initialize base flow
sgm = np.tanh((xc-shock_loc)/(shock_width))
# U[:,0] = sgmab(0.0619,0.0619*5.28926794,sgm)
# U[:,1] = sgmab(930.2,930.2/5.28926794,sgm)
# U[:,3] = sgmab(57.9,57.9*8.17598965,sgm)
U[:,0] = sgmab(rinf,rinf*5.56521739,sgm)
U[:,1] = sgmab(Uinf,Uinf/5.56521739,sgm)
U[:,3] = sgmab(tinf,tinf*13.3867187,sgm)


tmp_r = U[:,0].copy(); tmp_u = U[:,1].copy()
U[:,1] *= tmp_r; U[:,3] *= tmp_r*cv; U[:,3] += 0.5*tmp_r*tmp_u**2


for j in range(n-1):
    i  = ife[j,0]; ii = ife[j,1]
    r,u,h0,a,t,at,e = get_facevars(U[i],U[ii])

    # inv = np.outer(np.array([0,1,u]),np.array([0.2*u**2,-0.4*u,0.4])) + \
    #       np.outer(np.array([1,u,h0]),np.array([-u,1,0])) + u*np.eye(3)
    invx = np.outer(np.array([0,1,0,u]),np.array([0.2*u**2,-0.4*u,0,0.4])) + \
           np.outer(np.array([1,u,0,h0]),np.array([-u,1,0,0])) + u*np.eye(4)   
    
    inv = invx 
    #inv = u*np.eye(3)
    A[i,i]   +=  inv; A[i,ii] +=  inv 
    A[ii,ii] += -inv; A[ii,i] += -inv


    mx[1,1] = -r*l2m; mx[2,2] = -r*l2m; mx[3,1] = u*mx[1,1]; mx[3,3] = -r*kap
    N = con2T(r,u,t)
    
    #vis = dsi*np.matmul(mx,N)
    vis = visc_coeff*viscx[j]*dsi*np.eye(4)
    #vis = visc_coeff*viscx[j]*dsi*np.matmul(mx,N)
    A[i,i]   += vis;  A[i,ii] +=  -vis
    A[ii,ii] += vis;  A[ii,i] +=  -vis
    #check this

ky = (dx/lx)*(om*sin(alpha))/(cos(alpha)+cinf/Uinf)
#cell looping
for i in range(n):
    _,u,h0,_,_ = getprim(U[i])
    invy = np.outer(np.array([0,0,1,u]),np.array([0,0,0,0.4])) + \
           np.outer(np.array([1,0,0,h0]),np.array([0,0,1,0]))    
    inv  = -invy*ky*1j
    A[i,i] += inv
    # if(xc[i]<0.05*lx or xc[i]>0.95*lx):
    #     A[i,:] = 0*np.eye(4)
    #     A[i,i] = 1e7*np.eye(4)


# ndim = np.zeros((4,4))
ndim = np.outer(1/(Uinf*np.array([rinf,rinf*Uinf,rinf*Uinf,rinf*Uinf**2])),np.array([rinf,rinf*Uinf,rinf*Uinf,rinf*Uinf**2]))
for i in range(n):
    A[i,:] *= ndim


#Converting to conservative perturbations (not non dimensionalization)
Ninv = np.zeros((4,4))
Ninv[0,0] = 1
Ninv[1,1] = rinf; Ninv[1,0] = Uinf
Ninv[2,2] = rinf; Ninv[2,0] = 0
Ninv[3,3] = 1/0.4; Ninv[3,0] = Uinf*Uinf/2; Ninv[3,1] = rinf*Uinf; 
#Enter perturbation form
bc = np.array([1/cinf**2,cos(alpha)/(rinf*cinf),sin(alpha)/(rinf*cinf),1])
pert = np.dot(Ninv,bc)
pert = np.dot(np.diag([1/rinf,1/(rinf*Uinf),1/(rinf*Uinf),1/(rinf*Uinf**2)]),pert)


def time_stepper(A,xc,pert,om):
    nmat = n*4
    M = np.transpose(A,(0,2,1,3)).reshape(nmat,nmat)
    # if 0:
    #     print(M[3,::3])
    #     eigs,ev = scipy.linalg.eig(-M,right=True)
    #     eigs = np.array(eigs)[::3]
    #     plt.scatter(eigs.real,eigs.imag); plt.show(); exit()

    # if 0:
    #     ieig = np.argwhere(np.array(eigs).real>0)
    #     print(eigs[np.array(eigs).real>0])
    #     print(ieig);
    #     print(ev.shape);
    #     plt.plot(xc,ev[0::3,127]); plt.show(); exit()
    I = np.eye(nmat,dtype='float64')
    # Mtx = scipy.sparse.coo_matrix((vals3, (vals1,vals2)), shape = (nmat,nmat))
    #A = 5e-2*I*1j + Mtx
    # u' = - Au => i om us = -A us => (i om I + A) us = 0 => d us =  AA (us + dus) dt 
    AA = -(M*(lx/dx) + om*I*1j) #Non-dimensional length-scale is dx
    u  = np.zeros((nmat),dtype='complex128')
    du = 0*u
    total_time = total_time_coeff*(2*lx*(om/Uinf)*(Uinf/lx)); dt = (cfl*dx/600)*(Uinf/lx)  ;nt = 100 # int(total_time/dt)
    x = np.repeat(xc, 4)
    #Change BC here
    bc_rhs = x < 0.1*lx; pt_vec = np.tile(pert, int(u[bc_rhs].shape[0]/4))
    tmp_pert = np.ones_like(u[bc_rhs])
    print('in time_stepper:',total_time,nt,dt)
    # impop = lg.splu()
    lu, piv = lu_factor(I - dt*AA)
    store_u = np.zeros((nt,nmat),dtype='complex128')
    for t in range(nt):
        #Explicit
        #du = dt*(A.dot(u) + rhs)
        #du = lg.spsolve(I - dt*A,dt*(A.dot(u) + rhs))

        #BC

        #u[bc_rhs] = tmp_pert 
        u[bc_rhs] = pt_vec + 0*1j #FIX THIS
        #print(np.mean(u[bc_rhs][0::5].real),np.mean(u[0::5].real))
        du = lu_solve((lu,piv),dt*(AA.dot(u)))
        print(t,np.linalg.norm(du.imag))
        u += du
        u[bc_rhs] = pt_vec + 0*1j
        u = (u*0.5*(1-np.tanh((x-sponge_loc)/sponge_width)))
        store_u[t] = u

    #plt.contourf(store_u[:,0::3])
    #plt.show()
    
    #print(u[bc_rhs])
    return store_u


su = time_stepper(A,xc,pert,om)

V = np.zeros((4*n),dtype='complex128')
for i in range(n):
    Urespdim = np.dot(np.diag([rinf,rinf*Uinf,rinf*Uinf,rinf*Uinf**2]),su[-1,i*4:i*4+4])
    #V[3*i:3*i+3] = np.dot(N,Urespdim)
    r,u,h0,a,t = getprim(U[i])
    #temperature
    #N = con2T(r,u,t) 
    #V[3*i:3*i+3] = np.dot(np.diag([1/rinf,1/Uinf,1/(287.05*Uinf**2)]),np.dot(N,Urespdim)) #non-dimensional V
    #pressure
    N = con2p(r,u) 
    V[4*i:4*i+4] = np.dot(np.diag([1/rinf,1/Uinf,1/Uinf,1/(rinf*Uinf**2)]),np.dot(N,Urespdim)) #non-dimensional V

#np.savetxt(str(om)+'.dat',V[3::4].real)

if 1:
    for var in [3]:# range(4):
        plt.plot(V[var::4].real,label=var); 
        z = scipy.signal.hilbert(V[var::4].real)
        #plt.plot(np.abs(z),label='envelope'); 
    plt.legend(); plt.show()
    #plt.plot(V[var::3],label=var); 
# plt.plot(1e-4*np.tanh((xf-shock_loc)/1e-12))

def make_plot2d(u,x):
    
    xx,yy = np.meshgrid(x,x,indexing='ij')
    uu = np.repeat(u[:,None],u.shape[0],axis=1)
    print(xx.shape, uu.shape)
    zz = uu.real*np.cos(ky*yy/dx) - uu.imag*np.sin(ky*yy/dx)

    plt.pcolor(xx,yy,zz)
    plt.clim(-1e-3)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()


make_plot2d(V[3::4],xc)



# if 0: #for conservative vars at different times
#     plt.plot(su[0,var::3],'-o'); plt.plot(su[-1,var::3],'-o'); plt.show()