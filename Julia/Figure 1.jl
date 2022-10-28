using Plots, Zygote

#Defines all of the algrithms
fdiff(f, x; h = sqrt(eps())) = (f(x+h) - f(x))/h 
cdiff(f, x; h = sqrt(eps())) = (f(x+0.5*h) - f(x-0.5*h))/h
cvar(f, x; h = sqrt(eps())) = imag((f(x+(im*h)))/h)

#Defines all of the Functions
f(x) = sin(x^2)
f_der(x) = 2x*cos(x^2) 

g(x) = sin(exp(x))
g_der(x) = exp(x)*cos(exp(x))

j(x) = atan(x)/(exp(1+x^2))
j_der(y) = gradient((x) -> atan(x)/(exp(1+x^2)),y)[1]

#start variables 
x0 = 2.
x1 = 2.
x2 = 2.
h_range = 10 .^ (-14:0.01:-2)



#Sine
    sin0 = [abs(fdiff(f, x0; h = h) - f_der(x0))/abs(f_der(x0)) for h in h_range]
    sin1 = [abs(cdiff(f, x0; h = h) - f_der(x0))/abs(f_der(x0)) for h in h_range]
    #sin2 = clamp.([abs(cvar(f, x0; h = h) - f_der(x0))/abs(f_der(x0)) for h in h_range],eps(),Inf)
#Results for CV clamped to ϵ to avoid errors.

#Exponential

    exp0 = [abs(fdiff(g, x1; h = h) - g_der(x1))/abs(g_der(x1)) for h in h_range]
    exp1 = [abs(cdiff(g, x1; h = h) - g_der(x1))/abs(g_der(x1)) for h in h_range]
    #exp2 = clamp.([abs(cvar(g, x1; h = h) - g_der(x1))/abs(g_der(x1)) for h in h_range],eps(),Inf)


#Arctan Thing
    arc0 = [abs(fdiff(j, x2; h = h) - j_der(x2))/abs(j_der(x2)) for h in h_range]
    arc1 = [abs(cdiff(j, x2; h = h) - j_der(x2))/abs(j_der(x2)) for h in h_range]
    #arc2 = clamp.([abs(cvar(j, x2; h = h) - j_der(x2))/abs(j_der(x2)) for h in h_range],eps(),Inf)

#Makes a line at machine epsilon
line = [eps() for _ in 1:length(h_range)]

#Plots the Functions
    fullplot = plot(h_range,[line sin1 exp1 arc1  sin0 exp0 arc0 ], Title = "Arc(x)", yaxis = :log,xaxis = :log,
    xlabel="h",ylabel="Absolute relative error",
    label = ["Machine ϵ" "Forward Difference "  false false "Centeral Differecnce"  false false  ], 
    c = [:black :red :red :red :blue :blue :blue :green :green :green ], legend = :bottomleft)  


#Finds the minimum of all the methods
sin0min = h_range[argmin(sin0)]
sin1min = h_range[argmin(sin1)]
sin2min = h_range[argmin(sin2)]

exp0min = h_range[argmin(exp0)]
exp1min = h_range[argmin(exp1)]
exp2min = h_range[argmin(exp2)]

arc0min = h_range[argmin(arc0)]
arc1min = h_range[argmin(arc1)]
arc2min = h_range[argmin(arc2)]


println("Optimal h:")
println("    ","\t", "FD \t \t ", " \t ", "CD \t \t", "\t ", "CV \t \t")
println("f(x): ","\t", sin0min, " \t ", sin1min, "\t", sin2min)
println("g(x): ","\t", exp0min, " \t ", exp1min, "\t", exp2min)
println("j(x): ","\t", arc0min, " \t ", arc1min, "\t", arc2min)

plot(fullplot)
