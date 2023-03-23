using Random
using Plots
using SparseArrays
using Statistics
using Printf
using LinearAlgebra

# "a" est le coefficient qu'on met devant le nouveau V

#_____________________________________________________________________________

function Vd(q)         # définition de la dérivée de la fonction potentiel 
    return sin(q)/2    # potentiel périodique 
end

#_____________________________________________________________________________

function euler(q,pas,a)                          # q est la valeur au temps n et pas l'incrémentation
    return -pas*a*Vd(q) + sqrt(2*pas)*randn()    #méthode d'euler
end

#_____________________________________________________________________________

function gibbs_RS(a)  # simulation de la mesure de Gibbs par méthode de rejet
#on initialise le while
g = 2*pi*rand() - pi
u = rand()

while u > exp(-a*(1-cos(g))/2) # exp(-(1-cos(q))/2) la valeur de notre densité sans Z
    g = 2*pi*rand() - pi 
    u = rand()
end
return g                     #on termine le programme et renvoie la valeur de la VA ayant pour densité la mesure de Gibbs
end

#_____________________________________________________________________________

function LR(q,a,pas)    #fonction du likelihood ratio, où q est un vecteur décrivant la trajectoire et a "l'amortissement du potentiel
    N = length(q)
    S = 0           # somme que l'on passera à l'exponentielle à la fin
               
    for n = 1:N-1  
        S += (q[n+1]-q[n])*(Vd(q[n])-a*Vd(q[n])) + pas*((Vd(q[n]))^2-(a*(Vd(q[n]))^2))/4
    end

   exp(-S)          # valeure finale du ratio renvoyée
end

#_____________________________________________________________________________

function GK_girsanov(N,T,J,a)
    pas = T/N                        # on définit le pas
    q0 = [gibbs_RS(a) for i = 1:J]   # on initialise les J trajectoire
    qT = zeros(N,J) 
    qT[1,:] = q0 
    M = zeros(J)                     # initialisation du vecteur des J likelihood ratio
    Inte = zeros(J)                  # initialisation des J intégrales de temps

    for n = 2:N                      # calcul des J trajectoires
        qT[n,:] = qT[n-1,:] .+ euler.(qT[n-1,:],pas,a)
    end

    qT_new = qT       #nouveau "qT" pour pouvoir faire l'intégration
    for n = 1:N
        qT_new[n,:] = Vd.(qT_new[n,:]).*Vd.(q0)
    end

    for j = 1:J                      # remplissage des J likelihood ratio et intégrales de temps
        M[j] = LR(qT[:,j],a,pas)
        Inte[j] = pas*sum(qT_new[:,j])
    end


    mean(M.*Inte)

end
"""
function green_kubo_LR(N,T,J,a)        # formule de Green-Kubo, N : découpe de T, T le temps, J nbr pour MC

    pas = T/N                        # construction du pas
    q_0 = [gibbs_RS(a) for i = 1:J]   # itialisation des J trajectoire
    q_T = q_0                    
    S = Vd.(q_0).*Vd.(q_0)*pas               # initialisation des J intégrales          
    means, vars = zeros(N), zeros(N)   #initialisation des moyennes et des variances empiriques
    means[1] = mean(S)
    vars[1] = var(S) 
    q_T_LR = sparse(N,J)
    q_T_LR = q_0    

    for n = 2:N
        q_T += euler.(q_T,pas,a)   # construction des J trajectoires
        q_T_LR[n,:] = q_T
        S += Vd.(q_T).*Vd.(q_0)*pas*LR(q_T,a,pas)              # intégration des J trajectoires
        means[n] = mean(S)
        vars[n] = var(S)
    end
    1 .-means, sqrt.(vars/J)*1.96

end
"""
