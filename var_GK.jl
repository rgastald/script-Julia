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

#________________________________________________________________________________

function einstein(N,T,J)     # méthode d'Einstein : n est le nombre de pas qu'on veut, T le temps d'intégration et J pour MC

    pas = T/N                        # construction du pas
    q_0 = [gibbs_RS() for i = 1:J]   # itialisation des J trajectoire
    q_T = q_0 
    means, vars = zeros(N), zeros(N)   #initialisation des moyennes et des variances empiriques 

    for n=2:N                     # début de la sommation des trajectoire
        
        q_T += euler.(q_T,pas)
        means[n] = mean( (q_T.-q_0).^2 ) /2/(n*pas)  # calcule de la moyenne empirique
        vars[n] = var( (q_T.-q_0).^2 /2/(n*pas))     # calcule de la variance empirique

    end
    means,sqrt.(vars/J)*1.96
    
end
D = 0.884175564 # valeur obtenue par différences finies
t=10/1000:10/1000:10
e,ve = einstein(100000,1000,10000)
C = D .+ zeros(1000)
gk,vgk = green_kubo(100000,1000,10000)
plot(t,e,ribbon=ve)
plot!(t,gk,ribbon=vgk)
plot!(t,C,color = "red", label = "coefficient")
#xlabel!("Temps T")
#ylabel!("Einstein")