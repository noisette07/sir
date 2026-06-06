# Modeles compartimentaux en epidemiologie
# Date : 2023/12/18
import numpy as np
import pandas as pd
import altair as alt
from scipy import integrate

# ###############################################################
# Pour modeliser notre systeme d'equations differentielles ordinaires,
# nous devons definir notre probleme dans une fonction def comprenant 
# en parametres d'entree :
#    N, un vecteur comprenant le nombre de personnes susceptibles (S), infectees (I) et retablies (R)
#    t, la variable par rapport a laquelle deriver
#    les parametres b et c
# Ecrivons les equations differentielles definies pour S/I/R.
def dN_dt(N, t, b, c):
    S = N[0]
    I = N[1]
    R = N[2]
    dS_dt = -b * S * I
    dI_dt = b * S * I - c * I
    dR_dt = c * I
    return np.array([dS_dt, dI_dt, dR_dt])

# ###############################################################
# Etablissons certains parametres inspires de la COVID-19.
# Il s'agit d'un cas fictif, et non pas d'une simulation credible ayant des vertus predictives.

# temps, jours
time = np.linspace(0, 150, 1000)

# taux d'infection en 1/(millions_personnes . jour)
b = 0.04

# taux de guerison, 1/jour
c = 0.07

# population initiale en millions de personnes
N_i = [7.5, 0.005, 0]

# ###############################################################
# Ces parametres nous permettent d'emblee de calculer le R0.
R0 = N_i[0] * b / c
print("R0 =", R0)

# La fonction odeint assemble les parametres et la fonction pour solutionner le systeme.
solution = integrate.odeint(dN_dt, N_i, time, args=(b, c))
print(solution[:5, :])

# Transformons ces resultats en un tableau pret pour le graphique.
solution_df = pd.DataFrame(solution, columns=["S", "I", "R"], index=time).reset_index()
solution_df = pd.melt(solution_df, id_vars="index")

# ###############################################################
# Creons un graphique avec une ligne dont la valeur en x est le temps,
# la valeur en y est le nombre de personnes, segmentee par couleur selon le compartiment.
sir_chart = alt.Chart(solution_df).mark_line().encode(
    x=alt.X("index", axis=alt.Axis(title="Temps (jours)")),
    y=alt.Y("value", axis=alt.Axis(title="Personnes (millions)")),
    color=alt.Color("variable", legend=alt.Legend(title="Compartiments")),
)
sir_chart.save("sir.html")

# On observe : 3 millions de personnes infectees en environ 40 jours.
# Rappel : c'est un cas fictif.

# ###############################################################
# Aplatir la courbe
# Supposons que le parametre c est fixe etant donne que nous n'avons pas de remede
# pour guerir de la COVID-19 plus rapidement.
# Le nombre de personnes susceptibles pourrait etre diminue avec un vaccin.
# Nous avons toutefois le pouvoir de modifier b avec des mesures de distanciation physique.
# Voyons la courbe d'infection pour differentes valeurs de b.

# temps, jours
time = np.linspace(0, 100, 201)

# taux de guerison, 1/jour
c = 0.07

# population initiale en millions de personnes
N_i = [7.5, 0.005, 0]

# Calcul pour differentes valeurs de b.
b_v = np.linspace(0, 0.1, 10)
solution_l = []
for i in range(b_v.shape[0]):
    b_i = b_v[i]
    solution = integrate.odeint(dN_dt, N_i, time, args=(b_i, c))
    solution_df = pd.DataFrame(solution, columns=["S", "I", "R"], index=time).reset_index()
    solution_df["b"] = b_i
    solution_l.append(solution_df)

# Transformons ces resultats en un tableau pret pour le graphique.
solution_b = pd.concat(solution_l, axis=0, sort=False)
solution_b = pd.melt(solution_b, id_vars=["index", "b"])

# On peut ainsi creer un graphique avec les differentes valeurs de b.
beta_chart = alt.Chart(solution_b.query("variable == 'I'")).mark_line().encode(
    x=alt.X("index", axis=alt.Axis(title="Temps (jours)")),
    y=alt.Y("value", axis=alt.Axis(title="Infectes (millions)")),
    color="b",
)
beta_chart.save("sir_b.html")

# En diminuant b, on aplatit ainsi la courbe.
# D'ou les mesures importantes prises pendant le COVID-19.
