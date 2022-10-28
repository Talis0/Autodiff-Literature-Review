using Plots,ForwardDiff

function func(X,A) 
        x = X[1]
        a = A[1]
        return -a*x
    end

A = [1.] #Value of constant 

h = 0.1 #Step Size 
MaxT = 5 #Maximum time 
x_0 = [1.] #Initial Conditions

#Application of Classical Runge-Kutta method
function RK4(f,h,x_0,MaxT)
    X = []
    push!(X,x_0[1])

    for i in 1:Int(MaxT/h)
        k1 = f(X[i])
        k2 = f(X[i] .+ (0.5.*h.*k1))
        k3 = f(X[i] .+ (0.5.*h.*k2))
        k4 = f(X[i] .+ (h*k3))
        push!(X,X[i]+h*(k1+2 *k2+2 *k3+k4)/6) #push! statment used to avoid array mutation
    end
    
    return X
end

#Creation of f'(a)
function ode(A)
    f(x) = func(x,A)
    X = RK4(f,h,x_0,MaxT)
    return X
end


Sol = ode(A)

Der = ForwardDiff.jacobian(ode,A)[:,1] #Calculation of the derivative
x = [x*h for x in 1:Int(MaxT/h)+1]

p1 = plot(x,Sol, 
    title = "Solution",
    ylabel = "f(a)",
    xlabel = "t")

p2 = plot(x, Der,
    title = "Derivative",
    ylabel = "f '(a)",
    xlabel = "t")

plot(p1,p2,layout = (1, 2),legend = false)
