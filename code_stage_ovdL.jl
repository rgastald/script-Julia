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

function euler(q,pas,beta)                          # q est la valeur au temps n et pas l'incrémentation
    return -pas*Vd(q) + sqrt(2*pas/beta)*randn()    #méthode d'euler
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


#________________________________________________________________________________

function einstein(n,T,J,beta)     # méthode d'Einstein : n est le nombre de pas qu'on veut, T le temps d'intégration et J pour MC
    pas = T/n                # on pose le pas de temps pour Euler-...?
    Est = 0                  # initialisation de l'estimateur
    Est2 = 0                 # calcul de la variance

    for j=1:J                     # début de la sommation des trajectoire
        q_0 = gibbs_RS(beta)
        q_T = q_0
        for k=1:n                 # construction d'une trajectoire
            q_T += euler(q_T,pas,beta)
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

#_____________________________________________________________________________

function test_gibbs(n,beta)  ### On voit si la fonction de Gibbs simulée à 
    x=[]                ### la répartition attendue
    for i=1:n
        push!(x,gibbs_RS(beta))
    end
    histogram(x)
end

#_____________________________________________________________________________

function rep_lin(eta,N,T,beta)   # réponse linéaire, n nbr d'étapes, eta coeff et T tps d'integr
    pas = T/N               # on pose le pas de temps pour Euler-...? 
    q_T = gibbs_RS(beta)        #on démarre une trajectoire, l'integrale donnera la même chose par ergodicité
    S = zeros(N)
    S[1] = -Vd(q_T)
    for n = 2:N
        q_T += euler(q_T,pas,beta) + eta*pas    # construction au fur et à mesure d'une trajectoire
        S[n] = S[n-1] - Vd(q_T)        # incrémentation de l'intégrale
    end
    return 1 .+ S/(N*eta)
end 

#_____________________________________________________________________________

function green_kubo(N,T,J,beta)        # formule de Green-Kubo, N : découpe de T, T le temps, J nbr pour MC

    pas = T/N                        # construction du pas
    q_0 = [gibbs_RS(beta) for i = 1:J]   # itialisation des J trajectoire
    q_T_int = q_0                    
    S = -Vd.(q_0)*pas                # initialisation des J intégrales               

    for n = 1:N-1
        q_T_int += euler.(q_T_int,pas,beta)   # construction des J trajectoires
        S += -Vd.(q_T_int)*pas           # intégration des J trajectoires
    end

    S = S.*(-Vd.(q_0))        # on multiplie par -sin(q_0) comme dans la formule
    return 1-beta^2*sum(S)/J           # normalisation

end

#_____________________________________________________________________________

function green_kubo_var(N,T,J)        # formule de Green-Kubo, N : découpe de T, T le temps, J nbr pour MC

    pas = T/N                        # construction du pas
    q_0 = [gibbs_RS() for i = 1:J]   # itialisation des J trajectoire
    q_T_int = q_0                    
    S = -Vd.(q_0)*pas                # initialisation des J intégrales               

    for n = 1:N-1
        q_T_int += euler.(q_T_int,pas)   # construction des J trajectoires
        S += -Vd.(q_T_int)*pas           # intégration des J trajectoires
    end

    S = S.*(-Vd.(q_0))        # on multiplie par -sin(q_0) comme dans la formule
    
    return 1.96*sqrt(var(S/J)/J)           # on renvoie la variance (le 1 - ... est bien sur gômmé)

end

#________________________________________________________________________________

function sol_poisson(h)   # solution de l'eq de Poisson par différences finies
    X = -1:h:1            # intervalle X sur lequel on a l'eq
    X = pi*X              # pour avoir quelque chose de correcte !
    h = pi*h
    V = Vd.(X)            # construction de Vd sur l'intervalle

    D = [V[i]/h - 2/h^2 for i=1:length(X)]       # diagonale principale
    DL = [1/h^2 for i=1:length(X)-1]             # lower diagonale
    DU = [1/h^2 - V[i]/h for i=1:length(X)-1]    # upper diagonale

    A = Tridiagonal(DL,D,DU)       # matrice pour résoudre le système et avoir PHI

    return A\V
    #det(A)

end

function diff_finies_coeff(h) 
    X = -1:h:1                 # intervalle X sur lequel on a l'eq
    X = pi*X                   # pour avoir quelque chose de correcte !
    PHI = sol_poisson(h)       # solution de l'eq de Poisson (mis ici car on actualise le h après)
    h = pi*h
    V = Vd.(X)                 # construction de Vd sur l'intervalle
    N = length(X)
    v = exp.((cos.(X).-1)./2)  # "vrai" potentiel V et non pas sa dérivée
    D = 0                      # initialisation du coeffecient de transport/ de l'integrale

    Z = 4.0528761338987   # constante de normalisation : merci Wolfram Alpha

    for i=1:N
        D += PHI[i]*V[i]*v[i]
    end

    return 1 + D*h/Z

end

"""
t = 10/1000:10/1000:10

plot(t,rep_lin(1,1000,10),label="eta = 1")
plot!(t,rep_lin(0.5,1000,10),label="eta = 0.5")
plot!(t,rep_lin(0.1,1000,10),label="eta = 0.1")
xlabel!("Temps T")
"""
