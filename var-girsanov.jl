using Random
using Plots
using SparseArrays
using Statistics
using Printf
using LinearAlgebra

# "a" est le coefficient qu'on met devant le nouveau V

#_____________________________________________________________________________

function Vd(q)         # définition de la dérivée de la fonction potentiel 
    return sin(q)/2 #-10 .* q.* exp.(-10 .* (q).^2)   # potentiel périodique 
    #return (1/5)*(-4*cos(q)-20*cos(q)*sin(q))
end

#_____________________________________________________________________________

function gibbs_RS(beta)  # simulation de la mesure de Gibbs par méthode de rejet
#on initialise le while
g = 2*pi*rand() - pi
u = rand()

while u > exp(-beta*(1-cos(g))/2) # exp(-(1-cos(q))/2) la valeur de notre densité sans Z
    g = 2*pi*rand() - pi 
    u = rand()
end
return g                     #on termine le programme et renvoie la valeur de la VA ayant pour densité la mesure de Gibbs
end
#_____________________________________________________________________________


function var_girsanov(N,T,J,beta,alpha,eps)  # N : nbr de pas, T : temps final, J : nombre de réalisation, U nouveau potentiel

    pas = T/N                        # on définit le pas
    
    q0 = [gibbs_RS(beta) for i = 1:J]   # on initialise les J trajectoire
    qT1 = q0
    qT2 = q0
    
    S1, S2 = Vd.(q0).*Vd.(q0)*pas, Vd.(q0).*Vd.(q0)*pas       # début de l'intégration
    
    Y1 = zeros(J)
    Y2 = zeros(J)
    
    for n = 2:N                      # calcul des J trajectoires
        dW = sqrt(pas) .* [randn() for i = 1:J] 

        Y1 += -dW .* Vd.(qT2)*(sqrt(beta/2)*alpha) .+ pas*((alpha * Vd.(qT1)).^2)*(beta/4) # calcul du log LR
        Y2 += -dW .* Vd.(qT2)*(sqrt(beta/2)*(alpha + eps)) .+ pas*(((alpha + eps)*Vd.(qT2)).^2)*(beta/4) # calcul du log LR
        
        qT1 += -pas * (1+alpha) * Vd.(qT1) .+ sqrt(2/beta)*dW         # actualisation de la trajectoire
        qT2 += -pas * (1+alpha+eps) * Vd.(qT1) .+ sqrt(2/beta)*dW
        
        S1 = S1 .+ Vd.(qT1) .* Vd.(q0) .*pas .*exp.(-Y1)
        S2 = S2 .+ Vd.(qT2) .* Vd.(q0) .*pas .*exp.(-Y2)
        
    end
    return (var(S2)-var(S1))/eps
    end

