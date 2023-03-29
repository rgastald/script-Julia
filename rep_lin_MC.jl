using Random
using Plots
using SparseArrays
using Statistics
using Printf
using LinearAlgebra


#_____________________________________________________________________________

function Vd(q)         # définition de la dérivée de la fonction potentiel 
    return sin(q)/2    # potentiel périodique 
end

#_____________________________________________________________________________

function euler(q,pas)                          # q est la valeur au temps n et pas l'incrémentation
    return -pas*Vd(q) + sqrt(2*pas)*randn()    #méthode d'euler
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



function rep_lin_MC(eta,N,T,J)   # réponse linéaire, n nbr d'étapes, eta coeff et T tps d'integr
    pas = T/N               # on pose le pas de temps pour Euler-Maruyama
    q_T = [gibbs_RS() for j = 1:J]       #on démarre une trajectoire, l'integrale donnera la même chose par ergodicité
    means = zeros(N)
    means[1] = mean(Vd.(q_T))
    for n = 2:N
        q_T += euler.(q_T,pas) .+ eta*pas    # construction au fur et à mesure d'une trajectoire
        means[n] = mean(Vd.(q_T))
    end
    return 1 .- means/(N*eta)
end 