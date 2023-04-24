using Random
using Plots
using SparseArrays
using Statistics
using Printf
using LinearAlgebra
#using LaTeXStrings

#fonction potentiel
function V(x)
    return cos(2*pi*x)/2
end

#fonction de drift
function b(x)         
    return pi*sin(2*pi*x)     
end

# dérivée de b
function gradb(x)
    return (2*pi^2)*cos(2*pi*x)
end

# Lb 
function Lb(x)
    return b(x) .* gradb(x) .- (4*pi^3).*sin(2*pi*x)
end

# simulation de la mesure de Gibbs par méthode de rejet
function gibbs_RS()  
g = rand()
u = rand()

while u > exp(-cos(2*pi*g)/2) 
    g = rand()
    u = rand()
end
return g                     
end

# densite de la mesure invariante sans constante de normalisation
function densite_replin(q,eps)
    N = 10000
    S = 0

    for i = 1:N
        S += exp.( V.(q .+ (i-1)/N ) - eps*(i-1)/N )
    end

    S = exp.(-V.(q)) .* S ./N
end

# calcul de la constante de normalisation
function densite_replin_normalise(eps)
    Q = 0:0.001:1
    Z = 0
    N = length(Q)

    for n = 1:N
        Z += densite_replin(Q[n],eps)*0.001
    end

    return Z
end

# calcule du coefficient de transport en utilisant une quadrature sur la formule obtenue par Fokker Planck
function rho(h,eps)
    Z = densite_replin_normalise(eps) # on calcule la constante de normalisation une seule fois
    S = 0           # initilisation de l'intégrale
    N = floor(1/h)  # nombre d'itérations

    for n = 1:N
        S += b(n/N)*densite_replin(n/N,eps)
    end

    return S/Z/N/eps
end

function CLR2(T,h,s)
    N = floor(T/h)
    X = [gibbs_RS() for i = 1:s]   #début des trajectoires
    Y = [0 for i = 1:s]
    alpha = [0 for i = 1:s] 
    beta = [0 for i = 1:s]  

    for n = 1:N
        alpha += b.(X)/N
        beta += (2*pi^2) .* cos.(2*pi*X) ./ N
        dW = sqrt(h) .* [randn() for i = 1:s]
        Y += (dW .+ (2*pi^2) .* cos.(2*pi*X) .* dW .*h ./2)/sqrt(2)
        X += sqrt(2)*dW .+ b.(X)*h .+ gradb.(X).*dW *h/sqrt(2) .+ Lb.(X)*(h^2)/2
    end

    A = mean(alpha)

    return var( (alpha .- A).*Y + (h/2)*beta )
end

T = 5
h1 = 1/100
h2 = 1/1000
s = 50000
x = 0.1:0.1:T
#R = ones(length(x)) * -0.1157582389   # h = 0.00001, eps = 0.0001 
v1 = CLR2.(x,h1,s)
v2 = CLR2.(x,h2,s)
plot(x,v1, label ="h = 1/100")
plot!(x,v2,label="h = 1/1000")
#plot!(x,R, label = "rho")
savefig("variance.pdf")