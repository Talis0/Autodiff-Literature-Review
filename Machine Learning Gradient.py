import jax
import jax.numpy as jnp
from jax import random
import numpy as np

key = random.PRNGKey(0)
t = 4 #number_of_training_data
l = 4 #layers
m = 5 #Dimention of input data
n = 2 #Dimention of output data
d = [m,3,3,n] #Layer Dimentions

X = [random.uniform(key+t,(m,)) for t in range(t)] #Training input data
Y = [random.uniform(key-t,(n,))for t in range(t) ] #Training output data

w = [random.uniform(key+l,(d[l+1],d[l])) for  l in range(l-1)] #Weight Matricies
b = [random.uniform(key-l,(d[l+1],)) for  l in range(l-1)] #Bias Vectors

V = [w,b]

def Bprop(V):
    i = 0
    C = 0
    w,b = V

    for x in X:
        a = x

        for j in range(l-1): 
            a = w[j]@a + b[j] 
            a = jax.nn.sigmoid(a) 

        C = C + jnp.linalg.norm(Y[i] - a)**2
        i = i+1

    return C

dx = jax.grad(Bprop)
gradient = dx(V)

