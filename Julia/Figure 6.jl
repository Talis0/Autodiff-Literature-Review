using Plots,ForwardDiff
function func(X,A) 
        x = X[1]
        a,b = A
        return [-a*x+b]
    end

A = [1.,0.] #Value of constant 

h = 0.01 #Step Size 
MaxT = 5 #Maximum time 
x_0 = [1.] #Initial Conditions

#Application of Classical Runge-Kutta method
function RK4(f,h,x_0,MaxT)
    X = []
    push!(X,x_0)

    for i in 1:Int(MaxT/h)
        k1 = f(X[i])
        k2 = f(X[i] .+ (0.5.*h.*k1))
        k3 = f(X[i] .+ (0.5.*h.*k2))
        k4 = f(X[i] .+ (h*k3))
        push!(X,X[i]+h*(k1+2 *k2+2 *k3+k4)/6)
    end
    
    return X
end

#Creation of f'(a)
function ode(A)
    f(x) = func(x,A)
    X = RK4(f,h,x_0,MaxT)
    return [x[1] for x in X]
end


Sol1 = ode(A)

Der = ForwardDiff.jacobian(ode,A)
Der1 = Der[:,1] #Each vector corresponds to the derivative with respect to each variable.
Der2 = Der[:,2]

x = [x*h for x in 1:Int(MaxT/h)+1]

p1 = plot(x,Sol1, 
    title = "Solution",
    ylabel = "f(t)",
    xlabel = "t",
    legend = false)

p2 = plot(x, [Der1,Der2],
    title = "Derivative",
    ylabel = "f '(t)",
    xlabel = "t",
    labels = ["da/dt" "db/dt"])

plot(p1,p2,layout = (2, 2),size = (800,500),legend = false)
