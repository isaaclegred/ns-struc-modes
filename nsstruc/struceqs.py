#!/usr/bin/python

import numpy as np
from scipy.special import hyp2f1
from .constants import *

exp = np.exp
log = np.log
fourpi = 4 * pi
# DEFINE TOV AND PERTURBATION EQUATIONS

def hydro(r,y,*args): # hydrostatic equilibrium

        props = args[-1] # args expected as mu(p),1/cs2(p),rho(p),props

        p = y[props.index('R')] # p in units of g/cm^3
        m = y[props.index('M')] # m in units of cm*c^2/G
        mu = args[0] # mu in units of g/cm^3
        
        return -G*(mu(p)+p)*(m+4.*np.pi*r**3*p)/(c**2*r**2*(1.-2.*G*m/(c**2*r)))
                
def mass(r,y,*args): # definition of the mass
        
        props = args[-1]
        
        p = y[props.index('R')] # p in units of g/cm^3
        mu = args[0] # mu in units of g/cm^3    

        return 4.*np.pi*r**2*mu(p) 


def derivativeofpsi(r,y,*args): # definition of psi
        
        props = args[-1]
        
        p = y[props.index('R')] # p in units of g/cm^3
        m = y[props.index('M')] # m in units of cm*c^2/G

        return (m+4.*np.pi*r**3*p) / (r*(r-2.*m*G/c**2)*(c**2/G))

def W(r,y,*args): #radial perturbation equation 
        
        props = args[-1]
        omega = args[-2]
        
        p = y[props.index('R')] # p in units of g/cm^3
        psi = y[props.index('H')] # psi is dimensionless
        m = y[props.index('M')] # m in units of cm*c^2/G
        w = y[props.index('omega')]
        v = y[props.index('v')]
        cs2i = args[1] # sound speed squared in units of c^2
        
        return cs2i(p)*(c**(-2)*omega**2*r**2*(1-2.*m*G/(c**2*r))**(-0.5)*np.exp(-psi)*v + 0.5*((m+4.*np.pi*r**3*p) / (r*(r-2.*m*G/c**2)*(c**2/G)))*w) - (2*(2+1)*((1-2.*m*G/(c**2*r))**(-0.5))*v)

def V(r,y,*args): #f-mode perturbation equation
        
        props = args[-1]
        
        p = y[props.index('R')] # p in units of g/cm^3
        psi = y[props.index('H')] # psi is dimensionless
        m = y[props.index('M')] # m in units of cm*c^2/G
        w = y[props.index('omega')]
        v = y[props.index('v')]
                
        return (v*((m+4.*np.pi*r**3*p) /  (r*(r-2.*m*G/c**2)*(c**2/G)))) - (w*((r-2.*m*G/c**2)**(-0.5)))/r**1.5
        

def baryonmass(r,y,*args): # definition of the baryonic mass
        
        props = args[-1]
        
        p = y[props.index('R')] # p in units of g/cm^3
        m = y[props.index('M')] # m in units of cm*c^2/G
        rho = args[2] # rho in units of g/cm^3  
        
        f = 1.-2.*G*m/(c**2*r)
                
        return 4.*np.pi*r**2*rho(p)/f**0.5 
        
def equad(r,y,*args): # gravitoelectric quadrupole tidal perturbation
        
        props = args[-1]
        
        p = y[props.index('R')] # p in units of g/cm^3
        m = y[props.index('M')] # m in units of cm*c^2/G
        eta = y[props.index('Lambda')] # dimensionless logarithmic derivative of metric perturbation
        mu = args[0] # mu in units of g/cm^3
        cs2i = args[1] # sound speed squared in units of c^2
                
        f = 1.-2.*G*m/(c**2*r)
        A = (2./f)*(1.-3.*G*m/(c**2*r)-2.*G*np.pi*r**2*(mu(p)+3.*p)/c**2)
        B = (1./f)*(6.-4.*G*np.pi*r**2*(mu(p)+p)*(3.+cs2i(p))/c**2)
                
        return (-1./r)*(eta*(eta-1.)+A*eta-B) # from Landry+Poisson PRD 89 (2014)

def slowrot_logderiv(r,y,*args): # slow rotation equation
        
        props = args[-1]
        
        p = y[props.index('R')] # p in units of g/cm^3
        m = y[props.index('M')] # m in units of cm*c^2/G
        beta = y[props.index('I')] # log derivative of frame-dragging function
        mu = args[0] # mu in units of g/cm^3
        
        f = 1.-2.*G*m/(c**2*r)  
        P = 4.*np.pi*G*r**2*(mu(p)+p)/(c**2*f)
                
        return -(1./r) * (beta*(beta + 3.) - P*(beta + 4.))


def slowrot_framedrag(r, y,*args):
        props = args[-1]
        beta = y[props.index('I')]
        frame_drag = y[props.index('framedrag')]
        return beta * frame_drag / r 

def nu_metric_param(r, y, *args):
        props = args[-1]
        p = y[props.index('R')]
        M = G/c**2*y[props.index('M')]
        
        return 2 * (fourpi * r * r * r * G / c**2 * p + M)/(r * (r - 2 * M))


# To see how these equations are derived See Y+Y https://arxiv.org/pdf/1303.1528.pdf
# and the example file eqn_generation.py.  these are the quadratic-in-Omega slow-rotation
# component functions that determine the quadrupole moment

def nuR(r, p, M):
        return (8*pi * p * r**3 + 2*M)/(r*(r-2*M))

def _h2_homog_helper(r, v2, h2, M, p, mu):
        # geometrized units, cm
        nuR_here = nuR(r, p, M)
        #print("nuR is", nuR_here)
        #print("counterterm is", r / ((r - 2*M) * nuR_here) * (8*pi * (mu + p) - 4*M / r**3))
        return (-nuR_here +  r / ((r - 2*M) * nuR_here) * (8*pi * (mu + p) - 4*M / r**3)) * h2 - 4 * v2 / (r * nuR_here * (r - 2*M))
        

def quadratic_h2_homog(r, y, *args):
        props = args[-1]
        p = y[props.index('R')]
        M = G / c**2 * y[props.index('M')]
        h2 = y[props.index('h2_homog')]
        v2 = y[props.index('v2_homog')]
        mu = args[0]

        f = 1 - 2 * M / r
        output = _h2_homog_helper(r, v2, h2, M, G/c**2*p, G/c**2*mu(p))
        #print("M is in cm", M)
        #print("4*M/r**3 is in cm^-2", 4*M/r**3)
        #print("8*pi*(p+eps)", 8*pi * G/c**2 * (p + mu(p)))
        
        #print("r is", r)
        #print("h2  is", h2)
        #print("dh2dr is", output)
        return output
def _v2_homog_helper(r, v2, h2, M, p):
        # Geometrized units, cm
        return -nuR(r, p, M) * h2
def quadratic_v2_homog(r, y, *args):
        props = args[-1]
        p = y[props.index('R')]
        M = G*y[props.index('M')]/c**2
        h2 = y[props.index('h2_homog')]
        v2 = y[props.index('v2_homog')]
        

        output = _v2_homog_helper(r, v2, h2, M, G*p/c**2)
        #print("v2_deriv is", output)
        #print("v2 is", v2)
        return output

def j_square(f, nu):
        return f*np.exp(-nu)
def j_square_deriv(M, r, nu, nuR, mu, f):
        return (2*M/r**2 - 8*pi*mu*r - nuR * f) * np.exp(-nu) 

def quadratic_v2(r, y, *args):
        props = args[-1]
        p = G/c**2*y[props.index('R')]
        M = G / c**2  * y[props.index('M')]
        omega1 = y[props.index('framedrag')]
        beta = y[props.index('I')]
        h2 = y[props.index('h2')]
        v2 = y[props.index('v2')]
        nu = y[props.index('nu')]

        mu = args[0]

        f = 1 - 2 * M / r
        # Eq 125 1967ApJ...150.1005H 
        def evaluate_inhomog(r, M, p, mu, f, beta, omega1, nu):
                nuR_here = nuR(r, p, M)
                jsquare = j_square(f, nu)
                jsquarederiv = j_square_deriv(M, r, nu, nuR_here, mu, f)
                return ((1  + r*nuR_here / 2) *  (omega1*r)**2 / 3 *
                        (-jsquarederiv  + 1 / (2*r) * jsquare * beta**2 ))

        homog_part = _v2_homog_helper(r, v2, h2, M, G/c**2*p)
        inhomog_part = evaluate_inhomog(r, M, G/c**2*p, G/c**2*mu(p), f, beta, omega1, nu)
        return homog_part + inhomog_part

def quadratic_h2(r, y, *args):
        props = args[-1]
        p = y[props.index('R')]
        M = G / c**2 * y[props.index('M')]
        omega1 = y[props.index('framedrag')]
        beta = y[props.index('I')]
        h2 = y[props.index('h2')]
        v2 = y[props.index('v2')]
        nu = y[props.index('nu')]
                
        mu = args[0]
        f = 1 - 2 * M / r 
        
        homog_part = _h2_homog_helper(r, v2, h2, M, G/c**2*p, G/c**2*mu(p))
        def evaluate_inhomog(r, M, p, mu, f, beta, omega1, nu):
                nuR_here = nuR(r, p, M)
                jsquare = j_square(f, nu)
                jsquarederiv = j_square_deriv(M, r, nu, nuR_here, mu, f)
                term1 = .5 * nuR_here * r
                term2 = 1 / ((r - 2*M) * nuR_here)
                return 1/3 * (r * omega1)**2 * (.5 * (term1 - term2) / r * jsquare * beta**2   -
                                              (term1 + term2) * jsquarederiv )

        inhomog_part = evaluate_inhomog(r, M, G/c**2*p, G/c**2*mu(p), f, beta, omega1, nu)
        return homog_part + inhomog_part



def eqsdict(): # dictionary linking NS properties with corresponding equation of stellar structure
        
        return {'R': hydro,'M': mass,'Lambda': equad,'I': slowrot_logderiv,'Mb': baryonmass, 'H': derivativeofpsi, 'omega': W,
                'v': V, 'framedrag' : slowrot_framedrag ,  'nu' : nu_metric_param, 'v2' : quadratic_v2,
                'h2' : quadratic_h2, 'h2_homog' : quadratic_h2_homog, 'v2_homog' : quadratic_v2_homog}

# INITIAL CONDITIONS

def initconds(pc, muc ,cs2ic, rhoc, psic, stp, props, nu_adjustment=0.0):
        """ 
        initial conditions for integration of eqs of stellar structure
        """
        Pc = pc - 2.*np.pi*G*stp**2*(pc+muc)*(3.*pc+muc)/(3.*c**2)
        mc = 4.*np.pi*stp**3*muc/3.
        Lambdac = 2.+4.*np.pi*G*stp**2*(9.*pc+13.*muc+3.*(pc+muc)*cs2ic)/(21.*c**2)
        omegac = 0.+16.*np.pi*G*stp**2*(pc+muc)/(5.*c**2)
        mbc = 4.*np.pi*stp**3*rhoc/3.
        Wc = 1e-14*stp**3
        Vc = -(1e-14*stp**2) / 2
        beta = 16.*pi * G/c**2 * stp**2 * (pc + muc) / 5.
        frame_drag = 1e-8
        nuc = 0.0 + nu_adjustment
        # Need to compute these 
        A= 1e-4 * frame_drag**2 # needs to be small compared to inhomogenous part of B
        # for numerical reasons I'm not sure I understand
        fc = 1 - 2*G*mc/c**2/stp
        jc = np.sqrt(fc)*np.exp(-nuc/2)
        B_homog = - 2*pi * G/c**2 * (pc + 1/3*muc) * A
        B = B_homog - fourpi/3 * G/c**2*(pc + muc) * (jc * frame_drag)**2
        print("B_homog is", B_homog)
        print("B_inhomog is", fourpi/3 * G/c**2*(pc + muc) * (jc * frame_drag)**2)
        h2 =  A * stp**2 # 1.0 is a test value, something else may lead to better convergence 
        v2 =  B * stp**4
        h2_homog = A*stp**2 # really is q*stp**2 with q = 1.0
        v2_homog = B_homog*stp**4
        
        return {'R': Pc,'M': mc,'Lambda': Lambdac,'I': omegac, 'Mb': mc, 'H': psic, 'omega': Wc, 'v': Vc, 'beta' : beta, 'framedrag' : frame_drag, 'nu':  nuc, 'h2' : h2, 'v2' : v2,
                'h2_homog': h2_homog, 'v2_homog' : v2_homog}

# SURFACE VALUES

def calcobs(vals,omega,props): # calculate NS properties at stellar surface in desired units, given output surficial values from integrator

        def Rkm(vals): # R in km
        
                R = vals[0]
        
                return R/1e5
                
        def MMsun(vals): # M in Msun
        
                M = vals[props.index('M')+1]
        
                return M/Msun
                
        def MbMsun(vals): # M in Msun
        
                Mb = vals[props.index('Mb')+1]
        
                return Mb/Msun
                
        def Lambda1(vals): # dimensionless tidal deformability
        
                etaR = vals[props.index('Lambda')+1] # log derivative of metric perturbation at surface
        
                C = G*vals[props.index('M')+1]/(c**2*vals[0]) # compactness
                fR = 1.-2.*C
        
                F = hyp2f1(3.,5.,6.,2.*C) # a hypergeometric function
                
                def dFdz():

                        z = 2.*C

                        return (5./(2.*z**6.))*(z*(-60.+z*(150.+z*(-110.+3.*z*(5.+z))))/(z-1.)**3+60.*np.log(1.-z))
        
                RdFdr = -2.*C*dFdz() # log derivative of hypergeometric function
                
                k2el = 0.5*(etaR-2.-4.*C/fR)/(RdFdr-F*(etaR+3.-4.*C/fR)) # gravitoelectric quadrupole Love number
        
                return (2./3.)*(k2el/C**5)
        
        def  PsiC(vals):
        
                psi = vals[props.index('H')+1] # psi is dimensionless
                R = vals[0]
                M = vals[props.index('M')+1]

                return  0.5 * np.log(1-2*G*M/(c**2*R)) - psi
        
        def  BC(vals):
        
                R = vals[0]
                M = vals[props.index('M')+1]
                psi = 0.5 * np.log(1-2*G*M/(c**2*R)) # psi is dimensionless
                WR = vals[props.index('omega')+1]
                VR = vals[props.index('v')+1]
                dpsidr = (M) / (R*(R-2.*M*G/c**2)*(c**2/G))

                return c**-2*omega**2*R**2*(1-2*G*M/(c**2*R))**(-0.5)*np.exp(-(psi))*VR+0.5*(dpsidr)*WR
                
        def MoI(vals):
        
                betaR = vals[props.index('I')+1] # value of frame-dragging function at surface
        
                return 1e-45*(betaR/(3.+betaR))*c**2*vals[0]**3/(2.*G) # MoI in 10^45 g cm^2

                
        
        def Quadrupole(vals):
                # helper function for doing the boundary matching for h2 and K2
                def _A_from_vals(vals, S):
                    R = vals[0]
                    M = G/c**2  * vals[props.index('M')+1]
                    betaR = vals[props.index('I')+1]
                    omega1 = vals[props.index('framedrag')+1]
                    h2_homog = vals[props.index('h2_homog')+1]
                    v2_homog = vals[props.index('v2_homog')+1]
                    h2 = vals[props.index('h2')+1]
                    v2 = vals[props.index('v2')+1]
                    K2_homog = v2_homog - h2_homog
                    K2 = v2 - h2
                    f = 1.0 - 2.0  * M / R
                    # Associated Legendre function of 2nd kind Q^2_2(R/M - 1)
                    Q22 = -3 * R / (M * f) *  (np.polyval([2/3, 4/3, -3, 1], M/R)  + R/(2*M) * f**2 * np.log(f))
                    # set up a linear system to solve a * (q; A) = b
                    # a = np.zeros((2, 2))
                    # a[0, 0] = h2_homog
                    # a[1, 0] = K2_homog
                    # a[0, 1] = -Q22
                    # a[1, 1] = -3 * R / M * (np.polyval([-2/3 , 1, 1], M/R) + R/(2*M) * (1-2*M**2/R**2) * np.log(f))

                    # # How much the particular solution missed the external solution
                    # b = np.array([1/(M*R**3) * (1+M/R) * S**2 - h2, -1/(M * R**3) * (1 + 2*M/R)*S**2 - K2]) 
                    #print(a)
                    # Analytical exterior solution -- no A dependence "inhomogenous"
                    def  diffh_I(M, R, S):
                            return S**2*(-4*M - 3*R)/(M*R**5)
                    # Analytical exterior solution -- A dependent terms (all linear in A)
                    def diffh_A( M, R, S):
                            x0 = M**2
                            x1 = R**2
                            x2 = R**4
                            x3 = R**3
                            x4 = M**3
                            x5 = log((-2*M + R)/R)
                            return (-2.66666666666667*M**5 + 2.66666666666667*M**4*R + 10.0*M*x2*x5 - 4.0*M*x2 - 2.0*R**5*x5 - 16.0*x0*x3*x5 + 16.0*x0*x3 + 8.0*x1*x4*x5 - 17.3333333333333*x1*x4)/(x0*x1*(-8.0*M*R + 8.0*x0 + 2.0*x1))
                    def evaluate_inhomog_h(r, M, p, mu, f, beta, omega1, nu):
                        nuR_here = nuR(r, p, M)
                        jsquare = j_square(f, nu)
                        jsquarederiv = j_square_deriv(M, r, nu, nuR_here, mu, f)
                        term1 = .5 * nuR_here * r
                        term2 = 1 / ((r - 2*M) * nuR_here)
                        return 1/3 * (r * omega1)**2 * (.5 * (term1 - term2) / r * jsquare * beta**2   -
                                              (term1 + term2) * jsquarederiv )

                    Z = np.zeros((2, 2))
                    Z[0, 0] = h2_homog
                    Z[1, 0] =  _h2_homog_helper(R, v2_homog, h2_homog, M, 0, 0)
                    Z[0, 1] = -Q22
                    Z[1, 1] = -diffh_A(M, R, S)

                    P=np.array([1/(M*R**3) * (1+M/R) * S**2 - h2,
                                -(evaluate_inhomog_h(R, M, 0, 0, 1-2*M/R, betaR, omega1, log(1-2*M/R))+ _h2_homog_helper(R, v2, h2, M, 0, 0)) + diffh_I(M, R, S) ])
                    sol = np.linalg.solve(Z,P)

                    
                    
                    print(sol)
                    return sol[1]
                R = vals[0]
                M = G / c**2 * vals[props.index('M')+1]
                betaR = vals[props.index('I')+1]
                omega1 = vals[props.index('framedrag')+1]
                chi = 1 / (6 * M**2) *  omega1 * R**3 * betaR
                A = _A_from_vals(vals, chi*M**2)
                print ("chi is computed as", chi)
                return 1 + 8 * A / (5*chi**2)
        store = lambda x : (lambda vals  : vals[props.index(x) + 1])
        



        print(vals)        
        return {'R': Rkm,'M': MMsun,'Lambda': Lambda1,'I': MoI, 'Mb': MbMsun, 'H': PsiC, 'omega': BC, 'v': BC, 'framedrag': store("framedrag"), 'nu' : store("nu"), 'h2': Quadrupole,
                'h2_homog':store("h2_homog"), 'v2':store("v2"), 'v2_homog':store("v2_homog")}       

        
