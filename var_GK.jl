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
    S = -Vd.(q_0)               # initialisation des J intégrales               

    for n = 1:N-1
        q_T_int += euler.(q_T_int,pas)   # construction des J trajectoires
        S += -Vd.(q_T_int)           # intégration des J trajectoires
    end

    S = pas*S.*(-Vd.(q_0))        # on multiplie par -sin(q_0) comme dans la formule
    return 1-sum(S)/J           # normalisation

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


T = 1:0.01:10
GK = []
GKV = []
for t = 1:0.01:10
    push!(GK,green_kubo(10*t,t,10000))
    push!(GKV,green_kubo_var(10*t,t,10000))
end

plot(T,GK,ribbon=GKV)
