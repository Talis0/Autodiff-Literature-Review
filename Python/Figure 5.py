import jax
import jax.numpy as jnp
import matplotlib.pyplot as plot
import math

A = [1.] #Constant Value
h = 0.1 #Step Size
MaxT = 5 #Max Time
steps = int(MaxT/h) #Number of Steps
u_0 = jnp.array([1]) #Initial COndition

#Definition of ODE
def func(X,A):
        x = X[0]
        a = A[0]
        output = [-a*x]
        return jnp.array(output)

#Implimentation of RK4
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
    return u[0]

#f'(a)
def ode(a):
    def f(X):
        return func(X,a)

    u = RK4(f,h,u_0,steps)
    
    return u


sol = ode(A) #Solution
der = jax.jacfwd(ode)(A)[0] #Derivative
x = [x*h for x in range(steps)]

#Plotting
plot.subplot(121)
plot.title("Solution")
plot.ylabel("f(a)")
plot.plot(x,sol)


plot.subplot(122)
plot.title("Derivative")
plot.ylabel("f ' (a)")
plot.xlabel("t")
plot.plot(x,der)

plot.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

plot.figure(figsize=(10,10)) 
plot.show()
