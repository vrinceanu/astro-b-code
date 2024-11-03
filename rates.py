import scipy.special as spec, scipy.constants as const
import numpy as np

kelvin = const.value('Boltzmann constant')
hartree = const.value('atomic unit of energy')
fsc = const.value('fine-structure constant')
speed_of_light = const.c
bohr = const.value('atomic unit of length')
chi = hartree/2/kelvin
cm = 1.e-2
tau = const.value('atomic unit of time')
factor = 2**6/3*np.sqrt(np.pi/3)*fsc**4*speed_of_light*bohr**2/cm**3*chi**(3/2)
L = lambda T: 8*bohr**3*np.pi**(3/2)*(chi/T)**(3/2)/cm**3

def A(n1, l1, n2, l2):
    if n2 >= n1 or abs(l2-l1) != 1 or n1 < 1 or n2 < 1 or l1 < 0 or l2 < 0 or l1 >= n1 or l2 >= n2: return 0
    Z = 1
    fact = Z**4*const.fine_structure**3/tau*4/3*max(l1,l2)/(2*l1+1)
    logF = 3*np.log((n1**2-n2**2)/2/n1**2/n2**2)
    logF += 2*((n2+2)*np.log(4*n1*n2)+(n1-n2-2)*np.log(n1-n2) - (n1+n2+2)*np.log(n1+n2) - np.log(4))
    logF += spec.gammaln(n1+n2+1)-spec.gammaln(n1-n2)-spec.gammaln(2*n2)
    fact *= np.exp(logF)
    R = np.array((1,0))
    l = n2 - 1
    for _ in range(n2-max(l1,l2)):
        a = (2*l+1)/2/(l+1)*n2/n1*np.sqrt((n1+l+1)/(n2+l)*(n1-l-1)/(n2-l))
        d = (2*l+1)/2/(l+1)*n1/n2*np.sqrt((n2+l+1)/(n1+l)*(n2-l-1)/(n1-l))
        b = np.sqrt((n2+l+1)/(n2+l)*(n2-l-1)/(n2-l))/2/(l+1)
        c = np.sqrt((n1+l+1)/(n1+l)*(n1-l-1)/(n1-l))/2/(l+1)
        R = np.array(((a,b),(c,d))).dot(R)
        l -= 1
    if l2 == l1 - 1: return fact*R[0]**2
    if l2 == l1 + 1: return fact*R[1]**2

def A_reduced(n1, n2):
    return (np.sum([(2*l + 1)*A(n1,l,n2,l-1) for l in range(n1)])+
            np.sum([(2*l + 1)*A(n1,l,n2,l+1) for l in range(n1)]))/n1**2
        

nT = 501
transitions = []
rates = []
for i in range(2, nT):
    for j in range(1,i):
        transitions.append((i,j))
        rates.append(A_reduced(i,j))
np.savez("rates",np.array(transitions), np.array(rates))