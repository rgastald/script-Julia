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


function GK_girsanov_new(N,T,J,a)
    pas = T/N                        # on définit le pas
    q0 = [gibbs_RS(1) for i = 1:J]   # on initialise les J trajectoire
    qT = q0
    S = Vd.(q0).*Vd.(q0)*pas
    means, vars = zeros(N),zeros(N)
    means[1] = mean(S)
    vars[1] = var(S)
    Y = zeros(J)                            # log LR

    for n = 2:N                      # calcul des J trajectoires
        qT_new = qT .+ euler.(qT,pas,a)
        Y += (qT_new .- qT).*(Vd.(qT) - a*Vd.(qT))/2 .+ pas*((Vd.(qT)).^2 - (a*(Vd.(qT)).^2))/4
        qT = qT_new
        S += Vd.(qT).*Vd.(q0)*pas
        means[n] = mean(S.*exp.((-1).*Y))
        vars[n] = var(S.*exp.((-1).*Y))
    end
    1 .- means, sqrt.(vars/J)*1.96
    end

#_____________________________________________________________________________

function green_kubo(N,T,J)        # formule de Green-Kubo, N : découpe de T, T le temps, J nbr pour MC

    pas = T/N                        # construction du pas
    q_0 = [gibbs_RS(1) for i = 1:J]   # itialisation des J trajectoire
    q_T_int = q_0                    
    S = Vd.(q_0).*Vd.(q_0)*pas               # initialisation des J intégrales          
    means, vars = zeros(N), zeros(N)   #initialisation des moyennes et des variances empiriques
    means[1] = mean(S)
    vars[1] = var(S)     

    for n = 2:N
        q_T_int += euler.(q_T_int,pas,1)   # construction des J trajectoires
        S += Vd.(q_T_int).*Vd.(q_0)*pas              # intégration des J trajectoires
        means[n] = mean(S)
        vars[n] = var(S)
    end
    1 .-means, sqrt.(vars/J)*1.96

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
    q0 = [gibbs_RS(1) for i = 1:J]   # on initialise les J trajectoire
    qT = q0
    S = Vd.(q0).*Vd.(q0)*pas         # début de l'intégration
    means, vars = zeros(N),zeros(N)
    means[1] = mean(S)               # initialisation des moyennes et variances empiriques
    vars[1] = var(S)
    Y = zeros(J)                          

    for n = 2:N                      # calcul des J trajectoires
        qT += euler_general.(qT,pas,U)         # actualisation de la trajectoire
        Y += sqrt(pas)*randn()*U.(qT)/sqrt(2) .+ pas*((U.(qT)).^2)/4  # calcul du log LR
        #qT = qT_new
        S += Vd.(qT).*Vd.(q0)*pas
        means[n] = mean(S.*exp.((-1).*Y))   # mise à jours des moyennes et des variances
        vars[n] = var(S.*exp.((-1).*Y))
    end
    1 .- means, sqrt.(vars/J)*1.96
    end



U(x) = 0.1 .* Vd.(x)#-10 .* x.* exp.(-10 .* (x).^2)     # potentiel avec bosse
t=10/1000:10/1000:10                              # intervalle de temps
gk,vgk = green_kubo(1000,10,10000)                # green kubo et son intervalle de confiance
gkg,vgkg = girsanov_general(1000,10,10000,U)      # green kubo (girsanov) et son intervalle de confiance
plot(t,gk,label="GK",ribbon=vgk)                  
plot!(t,gkg,label="GKG",ribbon=vgkg)
xlabel!("Temps T")
ylabel!("Green-Kubo")