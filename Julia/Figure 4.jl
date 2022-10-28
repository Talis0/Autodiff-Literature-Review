using Plots
T = collect(0:0.1:10)
x1 = [exp(-t) for t in T]
x2 = [exp(-1.5t) for t in T]
x3 = [exp(-0.5t) for t in T]
p1 = plot(T,[x1,x2,x3],label = ["a = 1" "a = 1.5" "a = 0.5"],title = "Derivative",ylabel = "f(a)")

x4 = [-t*exp(-t) for t in T]
p2 = plot(T,x4,title = " Solution",xlabel = "t",ylabel = "f'(a)")


plot(p1,p2,layout = (2, 1),legend = [true false])
