from __future__ import print_function
import numpy as np
from math import *
from scipy.stats import linregress
import time
from math import erfc

# path containing LAMMPS' dumpfile and param.qeq file
InputPath = "/backup/udoka/using_these/Li3ClO_mp-985585_files/"  

def readlammpsdumpfile(filename):

    f = open(filename,"r")
    r = f.readlines()
    f.close()
    
    nl = len(r)
    nat = int(r[3])
    nf = int(nl/(nat+9))
        
    # Read the unit cell dimensions including the tilt variable in real-space
    lmpcell = np.array([line.split() for line in r[5:8]],dtype=float)
    if len(lmpcell[0]) == 2: xy = xz = yz = 0.0
    else: xy, xz, yz = lmpcell[0][2], lmpcell[1][2], lmpcell[2][2]
    lx = lmpcell[0][1]-lmpcell[0][0]-xy
    ly = lmpcell[1][1]-lmpcell[1][0]
    lz = lmpcell[2][1]-lmpcell[2][0]


    b = sqrt(ly*ly+xy*xy)
    c = sqrt(lz*lz+xz*xz+yz*yz)
    A_angle = acos((xy*xz+ly*yz)/(b*c))
    B_angle = acos(xz/c)
    G_angle = acos(xy/b)

    a1, a2, a3 = [0.0]*3, [0.0]*3, [0.0]*3              # lists of length 3
    a1[0] = lx
    a2[0] = b*cos(G_angle)
    a2[1] = b*sin(G_angle)
    a3[0] = c*cos(B_angle)
    a3[1] = (b*c*cos(A_angle) - a2[0]*a3[0])/a2[1]
    a3[2] = np.sqrt(c*c - a3[0]*a3[0] -a3[1]*a3[1])

    t,x,y,z = [],[],[],[]
    for i in range(nf):
        for j in range(nat):
            s = r[i*(nat+9)+9+j].split()
            t.append(s[1])
            x.append(float(s[2]))
            y.append(float(s[3]))
            z.append(float(s[4]))
    #print(lx, a1[0],"\n",ly,a2[1],"\n",lz,a3[2])
    return nat,t,x,y,z,a1,a2,a3

# Collect QEq parameters and calculate shielding correction
def QEq_gamma_shield_param(natm):
   p = open(InputPath+"param.qeq","r")
   L = p.readlines()
   tp,chi,eta,gamma=[],[],[],[]
   for i in range(len(np.unique(t))):
      s=L[i].split()
      #if (s[0]==s[0].split("#")[0]):
      tp.append(s[0])
      chi.append(float(s[1]))
      eta.append(float(s[2]))
      gamma.append(float(s[3]))
   chi ={tp[i]:chi[i] for i in range(len(tp))}
   eta ={tp[i]:eta[i] for i in range(len(tp))}
   gamma ={tp[i]:gamma[i] for i in range(len(tp))}

   gamma_shd = np.matrix([[0.0]*natm]*natm)
   for i in range(natm):
      for j in range(natm):
         gamma_shd[i,j]=np.power(gamma[t[i]]*gamma[t[j]],-1.5)
   return chi,eta,gamma_shd

nat,t,x,y,z,a1,a2,a3 =readlammpsdumpfile(InputPath+"traj.shd")

def Rij(i,j,X,Y,Z,a,b,c):
    # Compute distance 
    dx = X[i]-X[j]
    dy = Y[i]-Y[j]
    dz = Z[i]-Z[j]
    # Adjust offsets against pbc
    if dx > 0.5*a:
        dx -= a
    if dx < -0.5*a:
        dx += a
    if dy > 0.5*b:
        dy -= b
    if dy < -0.5*b:
        dy += b
    if dz > 0.5*c:
        dz -= c
    if dz < -0.5*c:
        dz += c
    r2 = dx**2+dy**2+dz**2
    return dx,dy,dz,r2

def QEqLR(lmda,zeta):
    
    """ 
    This computes QEqLR according to Ref. []
    zeta denotes the Ewald splitting parameter
    lmda the dielectric constant
    """

    Nmax = 2
    K = 14.4            # Coulomb constant

    chi,eta,gamma_shd = QEq_gamma_shield_param(nat)
 
    """Calculation of the K-space vectors"""
    vf = 2*np.pi/np.dot(a1,np.cross(a2,a3))
    b1 = np.cross(a2,a3)*vf
    b2 = np.cross(a3,a1)*vf
    b3 = np.cross(a1,a2)*vf

    start = time.time()
    # Calculate all interactions
    def JEEQeq(i,j,lmda,zeta):  
        if i==j:
            Astar = 0
            Bstar = 0
            for u in range(-Nmax,Nmax+1):
                for v in range(-Nmax,Nmax+1):
                    for w in range(-Nmax,Nmax+1):
                        if not (u==0 and v==0 and w==0):
                            dx = u*a1[0]+v*a2[0]+w*a3[0]
                            dy = u*a1[1]+v*a2[1]+w*a3[1]
                            dz = u*a1[2]+v*a2[2]+w*a3[2]
                            dh1 = u*b1[0]+v*b2[0]+w*b3[0]   
                            dh2 = u*b1[1]+v*b2[1]+w*b3[1]
                            dh3 = u*b1[2]+v*b2[2]+w*b3[2]
                            dr = np.sqrt(dx*dx+dy*dy+dz*dz)
                            dh = np.sqrt(dh1**2+dh2**2+dh3**2)
                            Astar += erfc(dr*zeta)/dr
                            Bstar += (2*vf/dh**2)*np.exp(-dh*dh*0.25/(zeta*zeta))
            return eta[t[i]] + 0.5*lmda*K*(Astar+Bstar-4*zeta/np.sqrt(np.pi))
        else:
            AlphU = 0.
            Alpha = 0.
            Beta = 0.
            rx = x[j]-x[i]
            ry = y[j]-y[i]
            rz = z[j]-z[i]
            for u in range(-Nmax,Nmax+1):
                for v in range(-Nmax,Nmax+1):
                    for w in range(-Nmax,Nmax+1):
                        if u==0 and v==0 and w==0:
                            dr = np.sqrt(rx*rx + ry*ry + rz*rz)
                            r3g32 = np.power((dr*dr*dr+gamma_shd[i,j]),0.333333333333333)
                            Alpha += erfc(r3g32*zeta)/r3g32
                        else:
                            dx = x[j]-x[i]+u*a1[0]+v*a2[0]+w*a3[0]
                            dy = y[j]-y[i]+u*a1[1]+v*a2[1]+w*a3[1]
                            dz = z[j]-z[i]+u*a1[2]+v*a2[2]+w*a3[2]
                            dr = np.sqrt(dx*dx + dy*dy + dz*dz)
                            Alpha += erfc(dr*zeta)/dr
                        if not (u==0 and v==0 and w==0):
                            dhx = u*b1[0]+v*b2[0]+w*b3[0]
                            dhy = u*b1[1]+v*b2[1]+w*b3[1]
                            dhz = u*b1[2]+v*b2[2]+w*b3[2]
                            dh = np.sqrt(dhx*dhx + dhy*dhy + dhz*dhz)
                            Beta += np.exp(-dh*dh*0.25/(zeta*zeta))*np.cos(rx*dhx+ry*dhy+rz*dhz) * 2*vf/dh**2
            return 0.5*lmda*K*(AlphU+Alpha+Beta)

    # Build the Hardness matrix and the atomic electronegativity difference
    def mat(zeta): 
       Qtot = 0.0
       b = np.array([0.0]*len(x))
       b[0] = -Qtot
       A = np.matrix([[0.0]*len(x)]*len(x))
       A[0] = [1]*len(x) # First row = 1
       for i in range(1,len(x)):
           b[i] = chi[t[i]]-chi[t[0]]
           for j in range(len(x)):
               A[i,j] = JEEQeq(i,j,lmda,zeta)-JEEQeq(0,j,lmda,zeta)
       return b,A

    b,A = mat(zeta)
    print("elapsed time =",time.time()-start)

    Q = np.linalg.solve(A, -b)
    return Q

print(">>> running QEqLR charge calculation")

zeta = 0.15
lmda= 1.67
Q = QEqeLR(lmda,zeta)

for i in range(nat):
    print(t[i],Q[i])

print(">>> Done calculating the partial charge of each atom type")

