using Plots,ForwardDiff

function func(X,A) 
        x,y = X
        a = A[1]
        return [y,-a*x]
    end

A = [1.] #Value of constant 

h = 0.1 #Step Size 
MaxT = 10 #Maximum time 
x_0 = [1.,1.] #Initial Conditions
steps = Int(MaxT/h)

#Application of Classical Runge-Kutta method
function RK4(f,h,x_0,MaxT)
    X = []
    push!(X,x_0)

    for i in 1:steps
        k1 = f(X[i])
        k2 = f(X[i] .+ (0.5.*h.*k1))
        k3 = f(X[i] .+ (0.5.*h.*k2))
        k4 = f(X[i] .+ (h*k3))
        push!(X,X[i]+h*(k1+2 *k2+2 *k3+k4)/6)
    end
    
    return [x[1] for x in X],[x[2] for x in X]
end

#Creation of f'(a)
function ode(A)
    f(x) = func(x,A)
    X = RK4(f,h,x_0,MaxT)
    return append!(X[1],X[2]) #ForwardDiff only accepts single vectors as outputs, 
                                                    # hense the two solution curves are murged
end


Sol = ode(A)
Sol1 = Sol[1:steps+1] #Unpacking of solution curves
Sol2 = Sol[steps+2:2*steps+2]


Der = ForwardDiff.jacobian(ode,A) 

Der1 = Der[1:steps+1]
Der2 = Der[steps+2:2*steps+2]

t = [x*h for x in 1:steps+1]

p1 = plot(Sol1,Sol2, 
    title = "Solution",
    ylabel = "x",
    xlabel = "y")

p2 = plot(Der1,Der2,
    title = "Derivative",
    ylabel = "x'",
    xlabel = "y'")

plot(p1,p2,layout = (2, 2),legend = false,size = (800,500))
