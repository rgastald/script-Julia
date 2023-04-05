using Random
using Plots
using SparseArrays
using Statistics
using Printf
using LinearAlgebra


#_____________________________________________________________________________

function Vd(q)         # définition de la dérivée de la fonction potentiel 
    return sin(q)/2  -10 .* q.* exp.(-10 .* (q).^2)   # potentiel périodique 
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
    q0 = [gibbs_RS() for i = 1:J]   # itialisation des J trajectoire                    
    qT = q0
    S = Vd.(qT).*Vd.(qT)               # initialisation des J intégrales          
    means, vars = zeros(N), zeros(N)   #initialisation des moyennes et des variances empiriques
    means[1] = mean(S)
    vars[1] = var(S)     

    for n = 2:N
        qT += euler.(qT,pas)   # construction des J trajectoires
        S = Vd.(qT).*Vd.(q0)             # intégration des J trajectoires
        means[n] = mean(S)
        vars[n] = var(S)
    end
    means, sqrt.(vars/J)*1.96

end

#_____________________________________________________________________________

"""
on fait ici des fonctions plus générales où l'on passe de V à un potentiel U (dérivé) général
"""
function euler_general(q,pas,U)                          # q est la valeur au temps n et "pas" l'incrémentation
    return -pas*Vd(q) + sqrt(2*pas)*randn() + U(q)*pas    #méthode d'euler
end

function girsanov_general(N,T,J,U)  # N : nbr de pas, T : temps final, J : nombre de réalisation, U nouveau potentiel

    pas = T/N                        # on définit le pas
    q0 = [gibbs_RS() for i = 1:J]   # on initialise les J trajectoire
    qT = q0
    S = Vd.(q0).*Vd.(q0)         # début de l'intégration
    means, vars = zeros(N),zeros(N)
    means[1] = mean(S)               # initialisation des moyennes et variances empiriques
    vars[1] = var(S)
    Y = zeros(J)                          

    for n = 2:N                      # calcul des J trajectoires
        qT += euler_general.(qT,pas,U)         # actualisation de la trajectoire
        Y += sqrt(pas)*randn()*U.(qT)/sqrt(2) .+ pas*((U.(qT)).^2)/4  # calcul du log LR
        #qT = qT_new
        S = Vd.(qT).*Vd.(q0)
        means[n] = mean(S.*exp.((-1).*Y))   # mise à jours des moyennes et des variances
        vars[n] = var(S.*exp.((-1).*Y))
    end
    means, sqrt.(vars/J)*1.96
    end


U(x) = 0.2*Vd(x)
t=10/1000:10/1000:10
Z = zeros(1000)
gk,vgk = green_kubo(1000,10,10000)
gkg,vgkg = girsanov_general(1000,10,10000,U)
plot(t,gk,label = "autocorrélation GK",title="Autocorrélation Green-Kubo N = 1000, T = 10, J = 10000")
plot!(t,gkg,label = "autocorrélation GKG")
plot!(t,Z)
xlabel!("Temps T")