# %% [markdown]
# # Homework 2 - Structural VAR Analysis
# ## Cecilie Brevig & Victoria Bojesen Hviid
# 
# En samlet notebook til spørgsmål 1-8. Kør cellerne i rækkefølge. Opbygning og notation følger øvelsesnotebooks: `y`, `p`, `Beta`, `SIGMA`, `A`, `B0inv`, `GAMMA`, `Pi`. Egne hjælpefunktioner ligger i `hw2_functions.py` og er kommenteret på dansk.
# 
# **Start:** Pak ZIP-filen ud og åbn denne notebook. `result.csv` og `hw2_functions.py` skal ligge i samme mappe. Der er ingen MATLAB-afhængighed og ingen krav om `s_master.py`, JCV/PSSCV eller SigLevels. Der bruges grundlæggende NumPy-matrixregning og SciPy-solvers, ikke færdige VAR/SVAR/VECM-, bootstrap-, IRF- eller FEVD-estimatorer. Tabellen `result.csv` er den uploadede, uændrede datafil. De udleverede kursusfunktioner er ikke kopieret ind eller vedlagt.
# 
# **Fast specifikation:** Sverige 1983Q1-2004Q4; VAR(3), konstant, trend og dummyer for 1992Q3, 1993Q1 og 1995Q4. Rækkefølgen er altid Bjørnlands $(i^*,y,\pi,i,\Delta e)$, også efter omordning ved identifikation.
# 
# **Adskil modellerne:** Q1-2 og Q4-8 bruger den stabile, ubegrænsede VAR. Q3 bruger den rangbegrænsede VEC-model fra HW1 med $r=3$, H1 og uden dummyer. VEC-modellens to enhedsrødder er forventede; de bruges ikke som begrundelse for at kassere denne model og indsættes ikke i $(I-\sum A_j)^{-1}$.
# 
# **Indstillinger:** Standard er 500 **godkendte** bootstrap-replikationer pr. identifikation, 68 % punktvise bånd (16.-84.-percentil, som i artiklen), og 24 viste kvartaler. Sæt `confidence=0.95` for 95 % bånd. Båndene er ikke simultane over alle horisonter.
# 
# **Kilder:** HW2.pdf; HW1-besvarelsen og HW1_Q8-Q11.ipynb; Lectures2to5_handout.pdf, slide 24-28; EC7 Slides.pdf; EC8.pdf; Exercise 7 og 8; Bjørnland (2009), afsnit 2-3 og tabel 1. Deltametodens generelle form: https://jrenne.github.io/TimeSeries_Bookdown/append.html#delta-method.

# %%
# 0. Pakker og indstillinger
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from hw2_functions import (
    MODELS, estimate_var, comp, stabVAR, lag_matrices, MAcoef,
    longrun_var, identify, check_identification, identify_full_solver,
    structural_irf, MPresponse, FEVD, model_results, estimate_vec_h1,
    vec2var, bootstrap_models, percentile_bands, delta_method)

p = 3
r = 3
h = 23                 # h=0,...,23 = 24 viste kvartaler
nboot = 500             # godkendte replikationer pr. model
confidence = 0.68       # kan ændres til fx 0.95
seed = 20261119
H_table = [1,4,8,12,24] # forecast-horisonter: H=1 bruger kun impact
OUT = Path('results')
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False})

def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / (filename+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT / (filename+'.pdf'),bbox_inches='tight')
    plt.show()

data = pd.read_csv('result.csv',parse_dates=['Dates'])
names = ['rfor','lgdp','pi','rdom','dlrexc']
# CSV har disse to variable i decimaler. Renter er allerede i procent.
data[['pi','dlrexc']] = data[['pi','dlrexc']]*100
y = data[names].to_numpy(dtype=float)
dum = data[['du93Q1','du92Q3','du95Q4']].to_numpy(dtype=float)
N,K = y.shape
T = N-p
# D er de deterministiske regressorer, trimmet til estimeringsstikprøven.
D = np.column_stack([np.ones(T),np.arange(1,T+1),dum[p:]])
Beta,residuals,X,SIGMA = estimate_var(y,p,D)
A = comp(Beta,p)
e = stabVAR(A)
print('y:',y.shape,'| X:',X.shape,'| Beta:',Beta.shape)
print('Rækkefølge:',names)
print('Største egenværdimodulus:',round(e[0],6))
assert e[0] < 1, 'Den ubegrænsede VAR er ikke stabil.'

# %% [markdown]
# ## Question 1 - To Cholesky-identifikationer og impulsresponser
# 
# Vi skriver $u_t=S\varepsilon_t$, hvor $\operatorname{Var}(\varepsilon_t)=I$ og $SS'=\widehat\Omega$. $S$ hedder `B0inv` i koden.
# 
# - Model 0: rente før valutakurs, $S_{45}=0$. Renten kan ikke reagere på ER-shocket i samme kvartal.
# - Model 1: valutakurs før rente, $S_{54}=0$. Valutakursen kan ikke reagere på MP-shocket i samme kvartal.
# 
# Begge impactmatricer gemmes i artikelrækkefølge, og MP er altid kolonneindeks 3 (shock 4). IRF har formen `(horisont, variabel, shock)`. `IRF_raw` er til enhedsvarians-shocks; kun MP-responserne til figurer normaliseres til en initial rentestigning på 1 procentpoint. Valutakursniveauet fås ved at summere responserne i $\Delta e$.
# 
# **Figurhenvisning:** B1-B2 i opgaveteksten er Canada i den uploadede artikel. Sverige er D1-D2. En stigning i realvalutakursen er en **depreciering**.

# %%
# 1. Identifikation og impulsresponser
point = {}
for model in [0,1]:
    point[model] = model_results(Beta,SIGMA,p,h,model)
    B0inv = point[model]['B0inv']
    print('\n'+MODELS[model])
    print('S45 =',round(B0inv[3,4],6),'| S54 =',round(B0inv[4,3],6))
    print('Initial renterespons:',point[model]['mp'][0,3])

quarters = np.arange(h+1)+1
fig,axs = plt.subplots(1,2,figsize=(11,4))
for model,style in [(0,'-'),(1,'--')]:
    axs[0].plot(quarters,point[model]['mp'][:,3],style,color='black',label=MODELS[model])
    axs[1].plot(quarters,point[model]['mp'][:,4],style,color='black',label=MODELS[model])
for ax,title,unit in zip(axs,['Interest rate','Real exchange rate'],
                         ['Percentage points','Approximate percent deviation']):
    ax.axhline(0,color='grey',lw=0.8)
    ax.set(title=title,xlabel='Quarter (1 = impact)',ylabel=unit)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
save_figure(fig,'Q1_Cholesky_IRF')
print('Begge modeller viser en depreciering efter stramningen: exchange rate puzzle.')
print('Kvalitativt som Sverige D1-D2; størrelsen replikerer ikke figuren præcist.')

# %% [markdown]
# ## Question 2 - Forecast error variance decomposition
# 
# Ved forecast-horisont $H$ er bidraget fra shock $j$ til variabel $i$:
# 
# $$\operatorname{FEVD}_{ij}(H)=100\frac{\sum_{s=0}^{H-1}\Theta_{ij,s}^2}
# {\sum_{k=1}^{K}\sum_{s=0}^{H-1}\Theta_{ik,s}^2}.$$
# 
# $\Theta_s=\Phi_sS$ er **ikke** normaliseret til 1 procentpoints rentestigning. Alle shocks har varians 1. Nævneren indeholder alle fem shocks, selv om tabellen kun viser MP og ER.
# 
# Vi gemmer to varianter. `fevd_diff` er for de variable, som står i VAR'en, herunder $\Delta e$. `fevd_level` er for realvalutakursens **niveau** $e$ og svarer til figurernes variable. For niveauet beregnes først $\sum_{j=0}^s\Theta_{5k,j}$ for hvert shock, og derefter kvadreres og summeres disse kumulerede responser over $s$. Man må hverken summere FEVD-andele eller kvadrere efter at have summeret på tværs af forecast-fejlens forskellige shocktidspunkter.
# 
# Impact ($H=1$) er ens for niveau og ændring. $H=4$ bruger impact og de næste tre kvartalers respons; det er ikke samme indekskonvention som en IRF ved $h=4$.

# %%
# 2. FEVD - tabeller for alle fem variable og de to udvalgte shocks
def make_fevd_table(point, models, key):
    rows = []
    for model in models:
        for H in H_table:
            for i,var in enumerate(names):
                for j,shock in [(3,'MP'),(4,'ER')]:
                    rows.append([MODELS[model],H,var,shock,point[model][key][H-1,i,j]])
    return pd.DataFrame(rows,columns=['Model','H','Variable','Shock','FEVD_percent'])

q2_level = make_fevd_table(point,[0,1],'fevd_level')
q2_diff = make_fevd_table(point,[0,1],'fevd_diff')
q2_level.to_csv(OUT/'Q2_FEVD_level.csv',index=False)
q2_diff.to_csv(OUT/'Q2_FEVD_diff.csv',index=False)
print(q2_level[q2_level['Variable'].isin(['rdom','dlrexc'])].round(3).to_string(index=False))

fig,axs = plt.subplots(2,2,figsize=(11,7))
for row,(i,var) in enumerate([(3,'Interest rate'),(4,'Real exchange rate level')]):
    for col,(j,shock) in enumerate([(3,'MP'),(4,'ER')]):
        ax = axs[row,col]
        for model,style in [(0,'-'),(1,'--')]:
            ax.plot(quarters,point[model]['fevd_level'][:,i,j],style,
                    color='black',label=MODELS[model])
        ax.set(title=var+': '+shock+' shocks',xlabel='Forecast horizon H',ylabel='Percent')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
save_figure(fig,'Q2_FEVD_orderings')

maxdiff = np.max(abs(point[0]['fevd_level']-point[1]['fevd_level']))
print(f'Største FEVD-forskel, alle variable/shocks/horisonter: {maxdiff:.3f} procentpoint.')

# %% [markdown]
# ## Question 3 - Fra VEC-estimater til companion-matrix
# 
# Vi viderefører HW1 Q8-11: H1 med ubegrænset konstant, to laggede første differenser og rang $r=3$, uden dummyer. Rangen er et input fra HW1, ikke et nyt rangvalg fra Q3. Estimationen udføres eksplicit med residualisering og det generaliserede egenværdiproblem for Johansen, uden færdig VECM-estimator.
# 
# $$\Delta y_t=\Pi y_{t-1}+\Gamma_1\Delta y_{t-1}+\Gamma_2\Delta y_{t-2}+c+u_t,
# \qquad \Pi=\alpha\beta'.$$
# 
# For VAR(3) gælder:
# 
# $$A_1=I+\Pi+\Gamma_1,\qquad A_2=\Gamma_2-\Gamma_1,\qquad A_3=-\Gamma_2.$$
# 
# Funktionen `vec2var` er vores egen implementering af opgavens krævede funktion. Med $K=5$ og $r=3$ forventer vi $K-r=2$ enhedsrødder i niveau-VAR'ens companion-matrix. De repræsenterer to fælles stokastiske trends. De øvrige egenværdier skal ligge inden for enhedscirklen for en sædvanlig I(1)-repræsentation. Vi sammenligner **modulus**, ikke om en kompleks egenværdi er 'mindre' end 1.

# %%
# 3. VEC-model og egenværdier
vec = estimate_vec_h1(y,p,r)
beta = vec['beta']
alpha = vec['alpha']
GAMMA = vec['GAMMA']
Pi = vec['Pi']
A_vec,Alags_vec = vec2var(GAMMA,Pi)
eig_vec = np.linalg.eigvals(A_vec)
order_eig = np.argsort(-np.abs(eig_vec))
q3 = pd.DataFrame({'Real':eig_vec[order_eig].real,
                   'Imaginary':eig_vec[order_eig].imag,
                   'Modulus':np.abs(eig_vec[order_eig])})
q3.to_csv(OUT/'Q3_VEC_eigenvalues.csv',index=False)
print(q3.round(6).to_string(index=False))
print('\nNormaliseret beta-transponeret:')
print(pd.DataFrame(beta.T,columns=names).round(5).to_string(index=False))
n_unit = np.count_nonzero(np.abs(eig_vec-1)<1e-7)
print(f'Enhedsrødder: {n_unit}; forventet K-r = {K-r}.')
print('De to enhedsrødder skyldes de fælles stokastiske trends.')
print('Den ubegrænsede VAR fra Q1 bruges fortsat til Q4-Q8.')
pd.DataFrame(beta.T,columns=names).to_csv(OUT/'Q3_beta.csv',index=False)

# %% [markdown]
# ## Question 4 - Kombination af kort- og langsigtsrestriktioner
# 
# Her tillades **begge** samtidige interaktioner mellem rente og valutakurs. De ni nuller i Bjørnlands ligning (3) beholdes, og den tiende restriktion er i stedet
# 
# $$\Theta_{54}(1)=\left[(I-A_1-A_2-A_3)^{-1}S\right]_{54}=0.$$
# 
# Da femte variabel er $\Delta e$, sætter dette den **samlede effekt på niveauet** $e$ til nul på lang sigt. Det sætter ikke den aktuelle valutakursrespons til nul. Vi bruger det uendelige langsigtsudtryk, ikke blot summen frem til kvartal 24.
# 
# Først løser `identify_full_solver` de 25 ligninger i samme stil som `restrictions` og `least_squares` i øvelserne: 15 kovariansligninger, ni kortsigtsnuller og én langsigtsrestriktion. Alle variable skaleres for at undgå problemer med forskellige enheder.
# 
# Til gentagelserne i bootstrap og numeriske delta-afledte bruges en ækvivalent, hurtigere solver: $S=LQ(\theta)$, hvor $L$ er Cholesky-faktoren, og $Q$ roterer de sidste to shockkolonner. Kovariansen og de ni nuller er da opfyldt for enhver vinkel. `root_scalar` løser den sidste langsigtsligning for vinklen. Vi kontrollerer, at de to solvere finder samme matrix, efter fortegnskonventionerne er ensrettet.

# %%
# 4. Numerisk identifikation med kort- og langsigtsrestriktioner
B0inv_solver,solver_result = identify_full_solver(Beta,SIGMA,p)
B0inv_rotation = identify(Beta,SIGMA,p,2)
B1 = longrun_var(Beta,p)
Theta1 = B1 @ B0inv_solver

print('B0inv = S:')
print(pd.DataFrame(B0inv_solver,index=names,
                   columns=['foreign','output','cost_push','MP','ER']).round(6).to_string())
print('\nLangsigtseffekt Theta1[4,3]:',Theta1[4,3])
print('S45:',B0inv_solver[3,4],'| S54:',B0inv_solver[4,3])
print('Solverens Jacobian-rang:',np.linalg.matrix_rank(solver_result.jac),'af 25')
print('Største forskel mellem to solvermetoder:',
      np.max(abs(B0inv_solver-B0inv_rotation)))
assert np.allclose(B0inv_solver,B0inv_rotation,atol=1e-8)
pd.DataFrame(B0inv_solver,index=names).to_csv(OUT/'Q4_impact_matrix.csv')

# %% [markdown]
# ## Question 5 - IRF og FEVD med den alternative identifikation
# 
# **Henvisning:** Artiklen viser Sverige i figur 5. Figur 3 viser Canada. Derfor sammenligner vi svenske estimater med figur 5 og Sveriges kolonne i tabel 1. I tabel 1 er de svenske impactandele 25 % (MP til realvalutakurs) og 23 % (ER til rente).
# 
# Vi viser rente, realvalutakursniveau, BNP og inflation. BNP er i omtrentlige procent, renter og inflation i procentpoint. Positiv realvalutakursrespons er en depreciering; negativ respons er en appreciering. MP-responser normaliseres separat i hver model til +1 procentpoint i renten på impact. FEVD bruger fortsat de oprindelige enhedsvarians-shocks.

# %%
# 5. Den tredje models punktestimater
point[2] = model_results(Beta,SIGMA,p,h,2)
plot_variables = [(3,'Interest rate','Percentage points'),
                  (4,'Real exchange rate','Approximate percent'),
                  (1,'Real GDP','Approximate percent'),
                  (2,'Inflation','Percentage points')]
fig,axs = plt.subplots(2,2,figsize=(10,7))
for ax,(i,title,unit) in zip(axs.flat,plot_variables):
    ax.plot(quarters,point[2]['mp'][:,i],color='black',lw=1.8)
    ax.axhline(0,color='grey',lw=0.8)
    ax.set(title=title,xlabel='Quarter (1 = impact)',ylabel=unit)
    ax.grid(alpha=0.2)
save_figure(fig,'Q5_Structural_IRF')

q5 = pd.DataFrame({
    'Contribution':['MP to real exchange rate','ER to interest rate'],
    'Our_estimate_percent':[point[2]['fevd_level'][0,4,3],point[2]['fevd_level'][0,3,4]],
    'Bjornland_Sweden_percent':[25,23]})
q5.to_csv(OUT/'Q5_Table1_comparison.csv',index=False)
print(q5.round(3).to_string(index=False))
print('Initial realvalutakursrespons:',round(point[2]['mp'][0,4],3),'ca. procent.')
q5_all = make_fevd_table(point,[2],'fevd_level')
q5_all.to_csv(OUT/'Q5_FEVD_level.csv',index=False)

# Vis konvergens mod nul over en længere horisont; LR er pålagt ved uendelig horisont.
long_raw = structural_irf(Beta,point[2]['B0inv'],p,400)
long_mp = MPresponse(long_raw,point[2]['B0inv'])
print('Realvalutakursniveau efter 400 kvartaler:',long_mp[-1,4])
print('Sammenlign disse kurvers form med figur 5, uden at påstå præcis replikation.')

# %% [markdown]
# ## Question 6 - Residual-bootstrap og deltametode for IRF
# 
# **Bootstrap:** (1) Centrér residualerne. (2) Træk hele femdimensionale residualvektorer med tilbagelægning. (3) Generér en ny dataserie rekursivt med den estimerede VAR, de første tre observerede vektorer som initialværdier og de samme deterministiske regressorer. (4) Reestimér VAR'en. (5) Kontrollér stabilitet, reidentificér hver model, og kontrollér kovarians, nuller og langsigtsrestriktion. (6) Beregn IRF og FEVD. (7) Kassér ugyldige replikationer og fortsæt, indtil hver model har 500 godkendte træk. Tællerne gør det klart, at 500 er godkendte træk, ikke nødvendigvis 500 forsøg. Det samme kandidatdatasæt bruges til alle modeller, som endnu mangler træk.
# 
# Bootstrap følger den almindelige empiriske residualfordeling uden ekstra bias- eller frihedsgradskorrektion. Stabilitetskassation gør inferensen betinget på de accepterede stabile modeller. Både antallet og andelen af kasserede træk skal omtales i besvarelsen.
# 
# **Deltametode:** Sæt $\theta=(\operatorname{vec}(A_1,\ldots,A_p),\operatorname{vech}(\Sigma))$, og lad $g(\theta)$ give de relevante responser. Vi beregner den numeriske Jacobian $G=\partial g/\partial\theta'$ og bruger
# 
# $$\widehat{\operatorname{Var}}[g(\widehat\theta)]
# =G\widehat{\operatorname{Var}}(\widehat\theta)G'.$$
# 
# Der medtages usikkerhed i **både** dynamiske koefficienter og residualkovariansen, og identifikation, normalisering og kumulering indgår inde i den differentierede funktion. `V` er allerede kovariansen for estimatoren, så vi dividerer ikke med T igen. Deltabånd er punktestimat $\pm z\cdot SE$ med normalfordelingskvantilen $z$.
# 
# Deltametoden bruger her den klassiske iid-Gaussiske asymptotiske parameterkovarians; bootstrap genbruger de empiriske residualer. Deltametoden kræver ikke 500 tilfældige replikationer. De 500 replikationer bruges til bootstrap. I sammenligningstabellerne medtages også normalbånd baseret på bootstrap-standardafvigelsen; disse er **ikke** den egentlige Jacobian-baserede deltametode.
# 
# HW1 fandt ARCH og ikke-normalitet. Den krævede iid-residual-bootstrap bevarer ikke tidsafhængig volatilitet, og den klassiske delta-approksimation har også begrænsninger her. Det skal nævnes ved fortolkning; vi erstatter ikke den efterspurgte metode med en anden bootstrap.

# %%
# 6a. 500 godkendte bootstrap-træk pr. model
boot,counts = bootstrap_models(y,Beta,residuals,p,D,h,nboot=nboot,
                              seed=seed,progress=True)
q6_counts = pd.DataFrame(counts).T
q6_counts.insert(0,'Model',MODELS)
q6_counts['discarded'] = q6_counts['attempted']-q6_counts['accepted']
q6_counts['discarded_percent'] = 100*q6_counts['discarded']/q6_counts['attempted']
q6_counts.to_csv(OUT/'Q6_Bootstrap_counts.csv',index=False)
print(q6_counts.round(3).to_string(index=False))

to_save = {}
for model in range(3):
    for key in ['mp','fevd_level','fevd_diff']:
        to_save[f'model{model}_{key}'] = boot[model][key]
np.savez_compressed(OUT/'bootstrap_draws.npz',**to_save)

# %%
# 6b. Egentlig deltametode - beregnes én gang og genbruges i Q8
delta = {}
for model in range(3):
    delta[model] = delta_method(Beta,SIGMA,X,p,h,model,confidence=confidence)
    print('Delta færdig:',MODELS[model])

q6_rows = []
for model in range(3):
    low,high,se_boot = percentile_bands(boot[model]['mp'],confidence)
    normal_z = norm.ppf((1+confidence)/2)
    fig,axs = plt.subplots(2,2,figsize=(11,7))
    for ax,(i,title,unit) in zip(axs.flat,plot_variables):
        curve = point[model]['mp'][:,i]
        ax.fill_between(quarters,low[:,i],high[:,i],color='grey',alpha=0.25,
                        label='Bootstrap percentile')
        ax.plot(quarters,delta[model]['mp']['low'][:,i],'--',color='teal',label='Delta')
        ax.plot(quarters,delta[model]['mp']['high'][:,i],'--',color='teal')
        ax.plot(quarters,curve,color='black',lw=1.6,label='Point estimate')
        ax.axhline(0,color='grey',lw=0.7)
        ax.set(title=title,xlabel='Quarter (1 = impact)',ylabel=unit)
        ax.grid(alpha=0.2)
        for hh in range(h+1):
            q6_rows.append([MODELS[model],names[i],hh,curve[hh],
                            se_boot[hh,i],low[hh,i],high[hh,i],
                            delta[model]['mp']['se'][hh,i],
                            delta[model]['mp']['low'][hh,i],
                            delta[model]['mp']['high'][hh,i],
                            curve[hh]-normal_z*se_boot[hh,i],
                            curve[hh]+normal_z*se_boot[hh,i]])
    axs[0,0].legend(fontsize=8)
    fig.suptitle(MODELS[model]+f' | {100*confidence:.0f}% pointwise IRF bands')
    save_figure(fig,f'Q6_IRF_bands_model{model}')

q6_irf = pd.DataFrame(q6_rows,columns=[
    'Model','Variable','h_after_impact','Estimate','Bootstrap_SE',
    'Percentile_low','Percentile_high','Delta_SE','Delta_low','Delta_high',
    'Bootstrap_normal_low','Bootstrap_normal_high'])
q6_irf.to_csv(OUT/'Q6_IRF_confidence_bands.csv',index=False)

# %% [markdown]
# ## Question 7 - Efrons percentilbånd for FEVD
# 
# Vi genbruger de **samme godkendte bootstrap-modeller** fra Q6. For hvert shock, hver variabel og hver horisont tager vi de empiriske 16.- og 84.-percentiler (ved 68 % bånd). Punktestimatet er den oprindelige models estimat; standardfejlen er standardafvigelsen på tværs af bootstrap-trækkene. Det er ikke standardafvigelsen divideret med kvadratroden af 500.
# 
# Figurerne viser de fire centrale bidrag: MP og ER til rente og realvalutakursniveau. CSV-tabellerne indeholder alle fem variable og de to shocks ved de udvalgte horisonter. Der gemmes også en separat tabel for $\Delta e$.

# %%
# 7. Efrons percentilmetode
q7_tables = {}
for key in ['fevd_level','fevd_diff']:
    rows = []
    for model in range(3):
        low,high,se = percentile_bands(boot[model][key],confidence)
        for H in H_table:
            for i,var in enumerate(names):
                for j,shock in [(3,'MP'),(4,'ER')]:
                    rows.append([MODELS[model],H,var,shock,point[model][key][H-1,i,j],
                                 se[H-1,i,j],low[H-1,i,j],high[H-1,i,j]])
    tab = pd.DataFrame(rows,columns=['Model','H','Variable','Shock','Estimate',
                                     'Bootstrap_SE','Percentile_low','Percentile_high'])
    tab.to_csv(OUT/f'Q7_{key}_percentile.csv',index=False)
    q7_tables[key] = tab

print(q7_tables['fevd_level'][q7_tables['fevd_level']['Variable'].isin(['rdom','dlrexc'])]
      .round(3).to_string(index=False))

for model in range(3):
    low,high,se = percentile_bands(boot[model]['fevd_level'],confidence)
    fig,axs = plt.subplots(2,2,figsize=(11,7))
    for row,(i,var) in enumerate([(3,'Interest rate'),(4,'Real exchange rate level')]):
        for col,(j,shock) in enumerate([(3,'MP'),(4,'ER')]):
            ax = axs[row,col]
            ax.fill_between(quarters,low[:,i,j],high[:,i,j],color='grey',alpha=0.3)
            ax.plot(quarters,point[model]['fevd_level'][:,i,j],color='black')
            ax.set(title=var+': '+shock+' shocks',xlabel='Forecast horizon H',ylabel='Percent')
            ax.grid(alpha=0.2)
    fig.suptitle(MODELS[model]+f' | Efron {100*confidence:.0f}% pointwise bands')
    save_figure(fig,f'Q7_FEVD_percentile_model{model}')

# %% [markdown]
# ## Question 8 - Deltabånd for FEVD og sammenligning
# 
# Den samme delta-Jacobian indeholder FEVD. Deltametoden differentierer altså også kvadrering, summering, brøken i FEVD og eventuel kumulering for realvalutakursniveauet. Det er ikke en efterfølgende kumulering af færdige standardfejl.
# 
# Percentilbånd kan være asymmetriske og ligger i [0,100], fordi de gyldige bootstrap-FEVD'er gør det. De normale førsteordens deltabånd er symmetriske omkring punktestimatet og kan gå under 0 eller over 100. Vi **beholder de rå bånd** i tabeller og figurer, så denne forskel er synlig. Ved strukturelt pålagte nuller på impact er både bidrag og usikkerhed nul. Ved andre punkter, hvor førsteafledte er nul, kan førsteordens delta være degenereret; et nul standarderror skal derfor ikke generelt fortolkes som høj præcision.
# 
# Vi sammenligner usikkerhed ved samme konfidensniveau, samme variabeldefinition og samme horisonter. Tabellerne viser begge metoder side om side. Kommentarerne skal baseres på de beregnede forskelle; der er ingen generel garanti for, at én metode altid giver bredere bånd.

# %%
# 8. Sammenligning af Efron og deltametoden
q8_tables = {}
for key in ['fevd_level','fevd_diff']:
    base = q7_tables[key].copy()
    values = []
    for model in range(3):
        for H in H_table:
            for i,var in enumerate(names):
                for j,shock in [(3,'MP'),(4,'ER')]:
                    values.append([delta[model][key]['se'][H-1,i,j],
                                   delta[model][key]['low'][H-1,i,j],
                                   delta[model][key]['high'][H-1,i,j]])
    base[['Delta_SE','Delta_low','Delta_high']] = np.array(values)
    base['Percentile_width'] = base['Percentile_high']-base['Percentile_low']
    base['Delta_width'] = base['Delta_high']-base['Delta_low']
    base.to_csv(OUT/f'Q8_{key}_method_comparison.csv',index=False)
    q8_tables[key] = base

for model in range(3):
    low,high,se = percentile_bands(boot[model]['fevd_level'],confidence)
    fig,axs = plt.subplots(2,2,figsize=(11,7))
    for row,(i,var) in enumerate([(3,'Interest rate'),(4,'Real exchange rate level')]):
        for col,(j,shock) in enumerate([(3,'MP'),(4,'ER')]):
            ax = axs[row,col]
            ax.fill_between(quarters,low[:,i,j],high[:,i,j],color='grey',alpha=0.25,
                            label='Efron percentile')
            ax.plot(quarters,delta[model]['fevd_level']['low'][:,i,j],'--',color='teal',
                    label='Delta')
            ax.plot(quarters,delta[model]['fevd_level']['high'][:,i,j],'--',color='teal')
            ax.plot(quarters,point[model]['fevd_level'][:,i,j],color='black',label='Point estimate')
            ax.axhline(0,color='grey',lw=0.7)
            ax.set(title=var+': '+shock+' shocks',xlabel='Forecast horizon H',ylabel='Percent')
            ax.grid(alpha=0.2)
    axs[0,0].legend(fontsize=8)
    fig.suptitle(MODELS[model]+f' | {100*confidence:.0f}% FEVD bands')
    save_figure(fig,f'Q8_FEVD_comparison_model{model}')

summary = q8_tables['fevd_level']
summary = summary[summary['Variable'].isin(['rdom','dlrexc'])]
print('Gennemsnitlig båndbredde ved de fem rapporterede horisonter:')
print(summary.groupby(['Model','Variable','Shock'])[['Percentile_width','Delta_width']]
      .mean().round(3).to_string())
print('Alle figurer og tabeller er gemt i:',OUT.resolve())

# %% [markdown]
# ## Kontrol af resultater og praktiske noter
# 
# Følgende celle kontrollerer blandt andet:
# 
# - Kovarians og pålagte identifikationsrestriktioner i alle tre punktestimater.
# - FEVD summerer til 100 % over **alle fem shocks**, også for valutakursniveauet.
# - Impact-FEVD er identisk for niveau og ændring af valutakursen.
# - Normaliseringen giver præcis +1 procentpoint i impactperioden.
# - VEC-til-VAR-omregningen bevarer $\Pi$ og $\Gamma_i$ og giver præcis to enhedsrødder.
# - MA-rekursionen stemmer med companion-metoden.
# - De numeriske delta-standardfejl ændrer sig kun lidt, når differensskridtet halveres.
# - Der er 500 godkendte bootstrap-træk, og optællingen af kasserede træk stemmer.
# 
# **Aflevering:** Notebooken er en arbejdsversion med kode og resultater. `hw2_functions.py` indeholder de egne nye funktioner. `.py`-filen `HW2_all_questions.py` indeholder samme beregninger som notebooken og kan også åbnes i VS Code med `# %%`-celler. Ingen udleverede masterfunktioner er vedlagt. Spørgsmålene kræver også, at I selv beskriver og diskuterer resultaterne i PDF-besvarelsen.
# 
# **Datadefinition:** Vi viderefører inflationen fra jeres `result.csv` (simpel årlig CPI-vækst). Artiklen definerer årlig ændring i log CPI. Det er dokumenteret, så forskelle fra artiklens figurer ikke fejlagtigt omtales som præcis replikation. Sammenligninger af resultater i denne notebook er på samme datatransformation hele vejen.

# %%
# Kontroller - køres efter spørgsmål 8
assert n_unit == K-r
assert np.all(np.abs(eig_vec[np.abs(eig_vec-1)>=1e-7]) < 1)
assert np.allclose(Alags_vec.sum(axis=0)-np.eye(K),Pi,atol=1e-9)
for j in range(1,p):
    assert np.allclose(-Alags_vec[j:].sum(axis=0),GAMMA[:,(j-1)*K:j*K],atol=1e-9)

Phi = MAcoef(Beta,p,h)
for s in range(h+1):
    assert np.allclose(Phi[s],np.linalg.matrix_power(A,s)[:K,:K],atol=1e-9)

for model in range(3):
    check_identification(Beta,SIGMA,p,model,point[model]['B0inv'])
    assert np.isclose(point[model]['mp'][0,3],1)
    for key in ['fevd_level','fevd_diff']:
        assert np.allclose(point[model][key].sum(axis=2),100,atol=1e-8)
        assert np.allclose(boot[model][key].sum(axis=3),100,atol=1e-8)
        assert np.all((boot[model][key]>=-1e-9)&(boot[model][key]<=100+1e-9))
    assert np.allclose(point[model]['fevd_level'][0],point[model]['fevd_diff'][0])
    assert counts[model]['accepted'] == nboot
    assert counts[model]['attempted'] == sum(counts[model][k] for k in
        ['accepted','unstable','identification_invalid','estimation_invalid'])
    half_step = delta_method(Beta,SIGMA,X,p,h,model,confidence=confidence,
                             step_multiplier=0.5)
    for key in ['mp','fevd_level','fevd_diff']:
        original = delta[model][key]['se']
        alternative = half_step[key]['se']
        error = np.max(abs(original-alternative)/(1+abs(original)))
        print(MODELS[model],key,'delta-step error:',f'{error:.2e}')
        assert error < 2e-4
print('Alle kontroller bestået.')
