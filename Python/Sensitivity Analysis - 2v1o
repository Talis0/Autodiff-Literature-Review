import jax
import jax.numpy as jnp
import matplotlib.pyplot as plot
import math

A = [1.]
h = 0.1
MaxT = 10
steps = int(MaxT/h)
u_0 = jnp.array([1,1])


def func(X,A):
        x,y = X
        a = A[0]
        output = [y,-a*x]
        return jnp.array(output)

def RK4(f,h,u_0,steps):
    u = jnp.zeros([len(u_0),steps])
    u = u.at[:,0].set(u_0)

    for i in range(steps):
        k1 = f(u[:,i])
        k2 = f(u[:,i] + (0.5*h)*k1)
        k3 = f(u[:,i] + (0.5*h)*k2)
        k4 = f(u[:,i] + h*k3)
        k = u[:,i]+h*(1/6)*(k1+ 2*k2+2*k3+ k4)

        u = u.at[:,i+1].set(k)
    return u


def ode(a):
    def f(X):
        return func(X,a)

    u = RK4(f,h,u_0,steps)
    
    return u


sol = ode(A)
der = jax.jacfwd(ode)(A)[0]
x = [x*h for x in range(steps)]

plot.subplot(211)
plot.title("Solution")
plot.ylabel("f(a)")
plot.plot(sol[0],sol[1])



plot.subplot(212)
plot.title("Derivative")
plot.ylabel("f ' (a)")
plot.xlabel("t")
plot.plot(der[0],der[1])

plot.subplots_adjust(top=1.5)

plot.show()
