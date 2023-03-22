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

#_____________________________________________________________________________

function green_kubo(N,T,J)        # formule de Green-Kubo, N : découpe de T, T le temps, J nbr pour MC

    pas = T/N                        # construction du pas
    q_0 = [gibbs_RS() for i = 1:J]   # itialisation des J trajectoire
    q_T_int = q_0                    
    S = Vd.(q_0).*Vd.(q_0)*pas               # initialisation des J intégrales          
    means, vars = zeros(N), zeros(N)   #initialisation des moyennes et des variances empiriques
    means[1] = mean(S)
    vars[1] = var(S)     

    for n = 2:N
        q_T_int += euler.(q_T_int,pas)   # construction des J trajectoires
        S += Vd.(q_T_int).*Vd.(q_0)*pas              # intégration des J trajectoires
        means[n] = mean(S)
        vars[n] = var(S)
    end
    1 .-means, sqrt.(vars/J)*1.96

end

#_____________________________________________________________________________

function green_kubo_var(N,T,J)        # formule de Green-Kubo, N : découpe de T, T le temps, J nbr pour MC

    pas = T/N                        # construction du pas
    q_0 = [gibbs_RS() for i = 1:J]   # itialisation des J trajectoire
    q_T_int = q_0                    
    S = Vd.(q_0)                     # initialisation des J intégrales         

    for n = 1:N-1
        q_T_int += euler.(q_T_int,pas)   # construction des J trajectoires
        S += Vd.(q_T_int)                # intégration des J trajectoires
    end

    S = pas*S.*(-Vd.(q_0))        # on multiplie par -sin(q_0) comme dans la formule
    S2 =  S.^2
    moyenne = sum(S)/J
    var_emp = sum(S2)/J - (moyenne)^2

    return 1.96*sqrt(var_emp/J)           # on renvoie la variance (le 1 - ... est bien sur gômmé)

end

#________________________________________________________________________________

function einstein(n,T,J)     # méthode d'Einstein : n est le nombre de pas qu'on veut, T le temps d'intégration et J pour MC
    pas = T/n                # on pose le pas de temps pour Euler-...?
    Est = 0                  # initialisation de l'estimateur
    Est2 = 0                 # calcul de la variance

    for j=1:J                     # début de la sommation des trajectoire
        q_0 = gibbs_RS()
        q_T = q_0
        for k=1:n                 # construction d'une trajectoire
            q_T += euler(q_T,pas)
        end

        Est += abs(q_T-q_0)^2     #actualisation de l'estimateur (non normalisé)
        Est2 += ( abs(q_T-q_0)^2 )^2
    end
    moyenne = Est/2/T/J
    var_emp = Est2/4/(T^2)/J - (moyenne)^2       # calcul de la variance Empirique 
    var_estr = var_emp/J
    marge_err = 1.96 * sqrt(var_estr)            # calcul de l'intervalle de confiance à 95%
    @printf "On a une estimation de %.6f" moyenne
    @printf " +/- %.6f" marge_err 
end
"""
T = 1:0.01:10
GK = []
GKV = []
for t = 1:0.01:10
    push!(GK,green_kubo(10*t,t,10000))
    push!(GKV,green_kubo_var(10*t,t,10000))
end

plot(T,GK,ribbon=GKV)
"""