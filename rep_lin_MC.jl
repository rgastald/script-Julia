using Random
using Plots
using SparseArrays
using Statistics
using Printf
using LinearAlgebra

function V(q)
    return (1-cos(q))/2
end
#_____________________________________________________________________________

function Vd(q)         # définition de la dérivée de la fonction potentiel 
    return sin(q)/2    # potentiel périodique 
end
#_____________________________________________________________________________

function gibbs_RS()  # simulation de la mesure de Gibbs par méthode de rejet
    #on initialise le while
    g = 2*pi*rand() - pi
    u = rand()
    
    while u > exp(-(1-cos(g))/2) # exp(-(1-cos(q))/2) la valeur de notre densité sans Z
        g = 2*pi*rand() - pi 
        u = rand()
    end
    return g                     #on termine le programme et renvoie la valeur de la VA ayant pour densité la mesure de Gibbs
    end

#_____________________________________________________________________________

function euler(q,pas)                          # q est la valeur au temps n et pas l'incrémentation
    return -pas*Vd(q) + sqrt(2*pas)*randn()    #méthode d'euler
end

#_____________________________________________________________________________

function densite_replin(q,eta)
    N = 5000
    S = 0
    for i = 1:N
        S += exp.( V.(q .+ 2*pi*(i-1)/N - pi) - eta*(2*pi*(i-1)/N - pi) )
    end
    S = exp.(-V.(q)) .* S .* 2*pi/N
end

#_____________________________________________________________________________

function densite_replin_normalise(q,eta)
    Q = -1:0.01:1
    Q = pi .* Q
    Z = 0
    N = length(Q)
    for n = 1:N
        Z += densite_replin(Q[n],eta)*0.01*pi 
    end
    return densite_replin(q,eta)/Z
end

#_____________________________________________________________________________

function gibbs_RS_forcing(eta)  # simulation de la mesure de Gibbs par méthode de rejet
#on initialise le while 
g = 2*pi*rand() - pi
u = rand()

while log(u) > log(densite_replin_normalise(g,eta)) # exp(-(1-cos(q))/2) la valeur de notre densité sans Z
    g = 2*pi*rand() - pi 
    u = rand()
end
return g                     #on termine le programme et renvoie la valeur de la VA ayant pour densité la mesure de Gibbs
end

#_____________________________________________________________________________

function rep_lin_MC(eta,N,T)   # réponse linéaire, n nbr d'étapes, eta coeff et T tps d'integr
    h = T/N               # on pose le pas de temps pour Euler-Maruyama
    qT2 = gibbs_RS_forcing(eta)        #on démarre une trajectoire, l'integrale donnera la même chose par ergodicité
    #qT2 = qT1
    #means = zeros(N)
    #means[1] = mean(Vd.(q_T))
    S = Vd(qT2) #- Vd(qT2)

    for n = 2:N
        dW = sqrt(h)*randn()
        #qT1 += -Vd(qT1)*h + sqrt(2)*dW   # construction au fur et à mesure d'une trajectoire
        qT2 += -Vd(qT2)*h + sqrt(2)*dW .+ eta*h
        #means[n] = mean(Vd.(q_T))
        S += Vd(qT2) #- Vd(qT1)
    end
    return 1 - S/N/eta
end 

function test_rep_lin(eta,N,T)
    h = T/N
    qT1 = gibbs_RS_forcing(0)
    qT2 = gibbs_RS_forcing(eta)

    for n = 1:N
        dW = sqrt(h)*randn()
        qT1 += -Vd(qT1)*h + sqrt(2)*dW   # construction au fur et à mesure d'une trajectoire
        qT2 += -Vd(qT2)*h + sqrt(2)*dW .+ eta*h
    end
    return qT2 - qT1
end

function integration_V(eta,h)
    x = -1:h:1
    x = pi*x
    #Z = 0
    S = 0
    for i in x
        S += Vd(i)*densite_replin(i,eta)
    end
    return S*h
end

Q = -1:0.01:1
Q = pi .* Q
I = densite_replin_normalise.(Q,0)
sum(I)*0.01*pi