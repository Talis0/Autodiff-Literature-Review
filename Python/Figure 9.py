import jax
import jax.numpy as jnp
import matplotlib.pyplot as plot
import math

steps = 100.
C = [steps]
MaxT = 10
u_0 = jnp.array([0,1])
h = MaxT/steps

def func(X,C):
        x,y = X
        a = C
        output = [y,-x]
        return jnp.array(output)

def f(X):
    return func(X,C)


def Euler(a):
    def f(X):
        return func(X,a)

    steps = a[0]
    
    h = MaxT/steps
    
    u = jnp.zeros([len(u_0),int(steps)])
    u = u.at[:,0].set(u_0)

    for i in range(int(steps)):
        k = u[:,i]+f(h*u[:,i])
        u = u.at[:,i+1].set(k)

    return u

sol = Euler(C)
der =  jax.jacfwd(Euler)(C)[0]
x = [s*h for s in range(int(steps))]
TrueSolx = [jnp.sin(t) for t in x]
TrueSoly = [jnp.cos(t) for t in x]

plot.subplot(211)
plot.title("Solution")
plot.ylabel("y")
plot.xlabel("x")
plot.plot(sol[0],sol[1])
plot.plot(TrueSolx,TrueSoly)


plot.subplot(212)
plot.title("Derivative")
plot.ylabel("y'")
plot.xlabel("x'")
plot.plot(der[0],der[1])

plot.subplots_adjust(top=1.5)

plot.show()
