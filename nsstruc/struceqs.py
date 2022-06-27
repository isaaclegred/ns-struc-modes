#!/usr/bin/python

import numpy as np
from scipy.special import hyp2f1
from .constants import *

exp = np.exp
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


# This is not the RHS for any equation but it's needed in a couple places places
def _m2(r, m, nu, p, mu, omega_1, beta,  h2):
        f = 1 - 2 * m / r  
        return -r * (f) * h2 + 1/6 * r**3 * omega_1**2 * np.exp(-nu) * f * (
                f * beta**2 + 16 * np.pi  * r**2  * G * (mu + p)/c**2
        )
# To see how these equations are derived See Y+Y https://arxiv.org/pdf/1303.1528.pdf
# and the example file eqn_generation.py.  these are the quadratic-in-Omega slow-rotation
# component functions that determine the quadrupole moment
def quadratic_K2(r, y, *args):
        props = args[-1]
        p = y[props.index('R')]
        M = G / c**2  * y[props.index('M')]
        omega1 = y[props.index('framedrag')]
        beta = y[props.index('I')]
        h2 = y[props.index('h2')]
        K2 = y[props.index('K2')]
        nu = y[props.index('nu')]
        
        
        mu = args[0]

        m2 = _m2(r, M, nu, p, mu(p), omega1, beta, h2)
        f = 1 - 2 * M / r  
        def evaluate(r, K2, h2, M, p,  m2, mu, f, beta, omega1, nu):
                x0 = r**2
                x1 = -r
                x2 = f*r
                x3 = 4*pi
                x4 = p*r**3*x3
                x5 = omega1**2
                x6 = mu + p
                x7 = exp(nu)
                x8 = m2*x7
                return (-1/12*beta**2*f**2*r**4*x5 - f**4*r*x8*(8*pi*p*x0 + 1) + f**3*x0*x7*(-2*K2 + h2*(x0*x3*x6 - 3)) + (4/3)*pi*f*r**6*x5*x6 - h2*x2*x7*(3*M + x1 + x4) + x8*(-M + r + x4))*exp(-nu)/(f*x0*(M + x1 + x2 - x4))
        return evaluate(r, K2, h2, M, G/c**2*p, m2, G/c**2*mu(p), f, beta, omega1, nu)

def quadratic_h2(r, y, *args):
        props = args[-1]
        p = y[props.index('R')]
        M = G / c**2 * y[props.index('M')]
        omega1 = y[props.index('framedrag')]
        beta = y[props.index('I')]
        h2 = y[props.index('h2')]
        K2 = y[props.index('K2')]
        nu = y[props.index('nu')]
                
        mu = args[0]

        m2 = _m2(r, M, nu, p, mu(p), omega1, beta, h2)
        f = 1 - 2 * M / r 
        def evaluate(r, K2, h2, M, p, m2, mu, f, beta, omega1, nu):
            x0 = f**2
            x1 = r**3
            x2 = -r
            x3 = f*r
            x4 = 4*pi
            x5 = p*x1*x4
            x6 = M + x2 + x3 - x5
            x7 = omega1**2
            x8 = beta**2*r**4*x0*x7
            x9 = mu + p
            x10 = 16*pi*f*r**6*x7*x9
            x11 = r**2
            x12 = 12*exp(nu)
            x13 = m2*x12
            x14 = f**4*r*x13*(8*pi*p*x11 + 1)
            x15 = f**3*x11*x12*(2*K2 - h2*(x11*x4*x9 - 3))
            x16 = -M + r + x5
            return (1/12)*(-x10*x6 + x14*x6 + x15*x6 + x16*(h2*x12*x3*(3*M + x2 + x5) - x10 - x13*x16 + x14 + x15 + x8) + x6*x8)*exp(-nu)/(x0*x1*x6)

        return evaluate(r, K2, h2, M, G/c**2*p, m2, G/c**2*mu(p), f, beta, omega1, nu)


def quadratic_h2_homog(r, y, *args):
        props = args[-1]
        p = y[props.index('R')]
        M = G / c**2 * y[props.index('M')]
        omega1 = y[props.index('framedrag')]
        beta = y[props.index('I')]
        h2 = y[props.index('h2_homog')]
        K2 = y[props.index('K2_homog')]
        nu = y[props.index('nu')]
        mu = args[0]

        m2 = _m2(r, M, nu, p, mu(p), omega1, beta, h2)
        f = 1 - 2 * M / r  
        def evaluate(r, K2, h2, M, p, m2, mu, f, beta, omega1, nu):
                x0 = r**2
                x1 = 4*pi
                x2 = r**3*x1
                x3 = p*x2
                x4 = M + f*r - r - x3
                x5 = 2*K2
                return (r*x4*(-h2*(x0*x1*(mu + p) - 3) + x5) + (-M + r + x3)*(3*M*h2 - h2*mu*x2 + 2*h2*r + r*x5))/(f*x0*x4)
        output =  evaluate(r, K2, h2, M, G/c**2*p, m2, G/c**2*mu(p), f, beta, omega1, nu)
        print("r is", r)
        print("h2  is", output)
        return(output)
def quadratic_K2_homog(r, y, *args):
        props = args[-1]
        p = y[props.index('R')]
        M = G*y[props.index('M')]/c**2
        omega1 = y[props.index('framedrag')]
        beta = y[props.index('I')]
        h2 = y[props.index('h2_homog')]
        K2 = y[props.index('K2_homog')]
        nu = y[props.index('nu')]
        
        mu = args[0]

        m2 = _m2(r, M, nu, p, mu(p), omega1, beta, h2)
        f = 1 - 2 * M / r  
        def evaluate(r, K2, h2, M, p, m2, mu, f, beta, omega1, nu):
            x0 = 4*pi*r**3
            x1 = 2*r
            return (K2*x1 + 3*M*h2 - h2*mu*x0 + h2*x1)/(r*(-M - f*r + p*x0 + r))
        output=  evaluate(r, K2, h2, M, G/c**2*p, m2, G/c**2*mu(p), f, beta, omega1, nu)
        print("k2 is", output)
        return output


def eqsdict(): # dictionary linking NS properties with corresponding equation of stellar structure
        
        return {'R': hydro,'M': mass,'Lambda': equad,'I': slowrot_logderiv,'Mb': baryonmass, 'H': derivativeofpsi, 'omega': W,
                'v': V, 'framedrag' : slowrot_framedrag ,  'nu' : nu_metric_param, 'K2' : quadratic_K2,
                'h2' : quadratic_h2, 'h2_homog' : quadratic_h2_homog, 'K2_homog' : quadratic_K2_homog}

# INITIAL CONDITIONS

def initconds(pc, muc ,cs2ic, rhoc, psic, stp, props): # initial conditions for integration of eqs of stellar structure

        Pc = pc - 2.*np.pi*G*stp**2*(pc+muc)*(3.*pc+muc)/(3.*c**2)
        mc = 4.*np.pi*stp**3*muc/3.
        Lambdac = 2.+4.*np.pi*G*stp**2*(9.*pc+13.*muc+3.*(pc+muc)*cs2ic)/(21.*c**2)
        omegac = 0.+16.*np.pi*G*stp**2*(pc+muc)/(5.*c**2)
        mbc = 4.*np.pi*stp**3*rhoc/3.
        Wc = 1e-14*stp**3
        Vc = -(1e-14*stp**2) / 2
        beta = 16.*pi * G/c**2 * stp**2 * (pc + muc) / 5.
        frame_drag = 1.0
        nu = 2.*np.pi*G*stp**2*(3.*pc+muc)/(3.*c**2)
        h2 = 1e-10 * stp**2 # 1.0 is a test value, something else may lead to better convergence 
        K2 = -1e-10 * stp**2
        h2_homog = 1e-10*stp**2 # really is q*stp**2 with q = 1.0
        K2_homog = -1e-10*stp**2
        
        return {'R': Pc,'M': mc,'Lambda': Lambdac,'I': omegac, 'Mb': mc, 'H': psic, 'omega': Wc, 'v': Vc, 'beta' : beta, 'framedrag' : frame_drag, 'nu':  nu, 'h2' : h2, 'K2' : K2, 'h2_homog'
        : h2_homog, 'K2_homog' : K2_homog}

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
                    K2_homog = vals[props.index('K2_homog')+1]
                    h2 = vals[props.index('h2')+1]
                    K2 = vals[props.index('K2')+1]
                    f = 1.0 - 2.0  * M / R
                    # Associated Legendre function of 2nd kind Q^2_2(R/M - 1)
                    Q22 = 3 * R / (M * f) *  (np.polyval ([1, -3, 4/3, 2/3], M/R)  + R/(2*M) * f**2 * np.log(f))
                    # set up a linear system to solve a * (q; A) = b
                    a = np.zeros((2,2))
                    a[0, 0] = h2_homog
                    a[0, 1] = K2_homog
                    a[1, 0] = Q22
                    a[1, 1] = 3 * R / M * (np.polyval([1, 1, -2/3], M/R) + R/(2*M) * (1-2*M**2/R**2) * np.log(f))

                    # How much the particular solution missed the external solution
                    b = np.array([1/(M*R**3) * (1+M/R) * S**2 - h2, 1/(M * R**3) * (1 + 2*M/R)*S**2 - K2]) 

                    sol = np.linalg.solve(a, b)
                    return sol[1]
                R = vals[0]
                M = G / c**2 * vals[props.index('M')+1]
                betaR = vals[props.index('I')+1]
                omega1 = vals[props.index('framedrag')+1]
                chi = 1 / (6 * M**2) *  omega1 * R**3 * betaR
                A = _A_from_vals(vals, chi*M**2)
                return -1 - 8 * A / (5*chi**2)
        store = lambda x : (lambda _  : vals[props.index(x) + 1])
                
        return {'R': Rkm,'M': MMsun,'Lambda': Lambda1,'I': MoI, 'Mb': MbMsun, 'H': PsiC, 'omega': BC, 'v': BC, 'framedrag': store("framedrag"), 'nu' : store("nu"), 'h2': Quadrupole, 'h2_homog':store("h2_homog"), 'K2':store("K2"), 'K2_homog':store("K2_homog")}       

        
