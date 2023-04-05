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
    N = 1000
    S = 0
    for i = 1:N
        S += exp.(Vd.(q .+ 2*pi*i/N)-eta*2*pi*i/N)
    end
    S = exp.(-Vd.(q)) .* S .* 2*pi/N
end

#_____________________________________________________________________________
function gibbs_RS_forcing(eta)  # simulation de la mesure de Gibbs par méthode de rejet
#on initialise le while 
g = 2*pi*rand() - pi
u = rand()

while u > densite_replin(g,eta) # exp(-(1-cos(q))/2) la valeur de notre densité sans Z
    g = 2*pi*rand() - pi 
    u = rand()
end
return g                     #on termine le programme et renvoie la valeur de la VA ayant pour densité la mesure de Gibbs
end



function rep_lin_MC(eta,N,T)   # réponse linéaire, n nbr d'étapes, eta coeff et T tps d'integr
    pas = T/N               # on pose le pas de temps pour Euler-Maruyama
    q_T = gibbs_RS()        #on démarre une trajectoire, l'integrale donnera la même chose par ergodicité
    #means = zeros(N)
    #means[1] = mean(Vd.(q_T))
    S = Vd(q_T)
    for n = 2:N
        q_T += euler.(q_T,pas) .+ eta*pas    # construction au fur et à mesure d'une trajectoire
        #means[n] = mean(Vd.(q_T))
        S += Vd(q_T)
    end
    return S/N/eta
end 

#a = rep_lin_MC(0.1,10000,1000)
#plot(a)

eta = 1/1000:1/1000:5
RL = rep_lin_MC.(eta,10000,100)
plot(eta,eta .* RL)