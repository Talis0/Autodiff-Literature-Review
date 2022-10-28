using Zygote
using Random
using LinearAlgebra
using StatsFuns
Random.seed!(0)

t = 100 #Number of training_data
l = 4 #Layers
m = 5 #Dimention of input data
n = 2 #Dimention of output data
d = [m,3,3,n] #Layer dimentions

X = [rand(m,) for t in 1:t] #Training input data
Y = [rand(n,) for t in 1:t ] #Training output data

w = [rand(d[l+1],d[l]) for  l in 1:l-1] #Weight Matricies
b = [rand(d[l+1],1) for  l in 1:l-1] #Bias Vectors
V = [w,b]

function Bprop(V)
    i = 1
    C = 0
    w,b = V

    for x in X
        a = x
        
        for j in 1:l-1
            a = w[j]*a + b[j]
            a = logistic.(a)
        end

        C = C + norm(Y[i] - a)^2
        i = i+1
    end

    return C
end

println("Cost = ", Bprop(V))
println("")
G = gradient(Bprop,V)[1]

println("Weight Matricies")
for g in G[1]
   show(stdout, "text/plain", g)
    println("")
    println("")
end
println()
println("Bias Vectors")
for g in G[2]
    show(stdout, "text/plain", g)
    println("")
    println("")
end
