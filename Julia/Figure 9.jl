using Plots,ForwardDiff

function func(X,A) 
        x,y = X
        return [y,-x]
    end

steps = 100
MaxT = 10 #Maximum time 
x_0 = [0.,1.] #Initial Conditions
A = [steps] #Value of constant
h = MaxT/steps

#Application of Eulers method
function Euler(f,steps,x_0,MaxT)  
    h = MaxT/steps
    X = []
    push!(X,x_0)
    i = 1
    while i <= steps
        push!(X,X[i]+h*(f(X[i])))
        i += 1
    end
    return [x[1] for x in X],[x[2] for x in X]
end

#Creation of f'(a)
function odeE(A)
    steps = A[1]
    f(x) = func(x,A)
    X = Euler(f,steps,x_0,MaxT)
    return append!(X[1],X[2])#ForwardDiff only accepts single vectors as outputs, 
                                                    # hense the two solution curves are murged
end


Sol = odeE(A)
Solx = Sol[1:steps+1] #Unpacking of solution curves
Soly = Sol[steps+2:2*steps+2]

DerE = ForwardDiff.jacobian(odeE,A)

DerEx = DerE[1:steps+1]
DerEy = DerE[steps+2:2*steps+2]

t = [x*h for x in 1:100]

TrueSolx = [sin(t) for t in t]
TrueSoly = [cos(t) for t in t]

p1 = plot([TrueSolx,Solx],[TrueSoly,Soly], 
    title = "Solution",
    ylabel = "y",
    xlabel = "x",
    zlabel = "t")

p2 = plot([DerEx],[DerEy],
    title = "Derivative",
    ylabel = "y'",
    xlabel = "x'")

plot(p1,p2,layout = (2, 2),legend = false,size = (800,500))
