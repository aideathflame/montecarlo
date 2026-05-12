# Monte Carlo-teori

Teoretisk bakgrund till `montecarlo.py`. För praktisk användning, se `montecarlo.md`.

## 1. Monte Carlo-metoden — Ulam, von Neumann & Metropolis

Metoden föddes på **Los Alamos** i mitten av 1940-talet. **Stanislaw Ulam** fick idén under en konvalescens 1946 när han försökte räkna ut sannolikheten för att lägga en viss patiens — han insåg att slumpmässig simulering var snabbare än kombinatorisk analys. Tillsammans med **John von Neumann** tillämpades tekniken på neutrondiffusion i väte­bomben. **Nicholas Metropolis** gav metoden sitt namn (efter kasinot i Monaco där Ulams farbror spelade) och publicerade med Ulam *"The Monte Carlo Method"*, *JASA* 1949.

### Kärnidé
När en analytisk lösning är omöjlig eller intraktabel — simulera många slumpmässiga utfall från den underliggande sannolikhets­modellen och skatta storheter av intresse empiriskt (medelvärde, kvantiler, sannolikheter). Felet avtar som `1/√N` oberoende av dimensionalitet, vilket gör metoden särskilt kraftfull för problem med många tillståndsvariabler.

### Intåg i finans
**Phelim Boyle** (1977), *"Options: A Monte Carlo Approach"*, *Journal of Financial Economics* — första tillämpningen på optionsprissättning. Öppnade dörren för MC som standardverktyg inom derivatvärdering, riskhantering (VaR) och strategiutvärdering.

## 2. Geometrisk Brownsk rörelse & log-avkastningar

**Louis Bachelier** (1900), *Théorie de la Spéculation* — första matematiska beskrivningen av prisrörelser som slumpvandring (aritmetisk Brownsk rörelse). **Paul Samuelson** (1965) föreslog den *geometriska* varianten för att undvika negativa priser:

```
dSₜ = μ Sₜ dt + σ Sₜ dWₜ
```

Itôs lemma ger att log-priset `ln Sₜ` följer aritmetisk Brownsk rörelse, vilket motiverar att arbeta med **log-avkastningar** `rₜ = ln(Pₜ / Pₜ₋₁)`:

- Additiva över tid (summa av log-avkastningar = totalavkastning)
- Garanterar positiva priser vid exponentiering: `Pₜ = Pₜ₋₁ · exp(rₜ)`
- Mer symmetriska än enkla avkastningar

**Black & Scholes** (1973) och **Merton** (1973) byggde vidare på detta till den berömda optionsprismodellen. `montecarlo.py` använder log-returer precis i den här andan, men ersätter den konstanta `σ` med en GARCH-process.

## 3. ARCH/GARCH — Engle, Bollerslev & GJR

### ARCH — Engle (1982)
**Robert Engle**, *"Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of U.K. Inflation"*, *Econometrica*. Nobelpris 2003. Genomslaget: variansen är inte konstant utan villkorad på senaste tidens chocker.

### GARCH — Bollerslev (1986)
**Tim Bollerslev**, *"Generalized Autoregressive Conditional Heteroskedasticity"*, *Journal of Econometrics*. Lade till en autoregressiv term i variansen — standardformen är GARCH(1,1):

```
σ²(t) = ω + α·ε²(t−1) + β·σ²(t−1)
```

Fångar **volatility clustering** (Mandelbrot 1963, Fama 1965): stora rörelser följs av stora rörelser, små av små.

### GJR-GARCH — Glosten, Jagannathan & Runkle (1993)
**Glosten, Jagannathan & Runkle**, *"On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks"*, *Journal of Finance*. Lägger till en asymmetrisk term för att fånga **leverage-effekten** (Black 1976) — att negativa chocker ökar framtida volatilitet mer än positiva av samma storlek:

```
σ²(t) = ω + (α + γ·I(ε<0))·ε²(t−1) + β·σ²(t−1)
```

Exakt modellen `montecarlo.py` skattar via `arch_model(..., vol="Garch", p=1, o=1, q=1)`. `o=1` är det som gör den till GJR.

### Stationäritet
För att variansprocessen ska vara välbeteende krävs:

```
α + γ/2 + β < 1
```

Annars exploderar den villkorade variansen över tid. Detta är persistence-diagnostiken i skriptet.

## 4. Student-t-innovationer — Bollerslev (1987)

**Bollerslev**, *"A Conditionally Heteroskedastic Time Series Model for Speculative Prices and Rates of Return"*, *Review of Economics and Statistics*. Även efter GARCH-filtrering har residualerna *feta svansar* — normalfördelningen underskattar sannolikheten för extrema utfall. Student-t med låga frihetsgrader (~5–8 för aktieavkastningar) fångar detta bättre. `dist="t"` i skriptet.

## 5. Filtered Historical Simulation — Barone-Adesi et al.

**Barone-Adesi, Giannopoulos & Vosper** (1999), *"VaR without Correlations for Portfolios of Derivative Securities"*, *Journal of Futures Markets*. Kärnidén:

1. Anpassa en GARCH-modell till historisk data.
2. Extrahera **standardiserade residualer** `zₜ = εₜ / σₜ` — dessa är approximativt i.i.d.
3. Bootstrappa (dra med återläggning) från denna empiriska fördelning vid simulering, istället för att dra från en parametrisk fördelning.

### Fördelar
- **Feta svansar bevaras** — de faktiska extremchockerna i historiken finns i urvalet.
- **Skevhet bevaras** — asymmetri mellan upp- och nedrörelser behöver inte antas bort.
- **GARCH-strukturen bibehålls** — clustering simuleras genom att chockerna skalas med den dynamiska σₜ.

Detta är precis vad `simulate()` gör: `z = np.random.choice(std_resid, size=num_paths)` drar från de empiriska standardiserade residualerna, och den villkorade variansen uppdateras varje steg enligt GJR-GARCH-rekursionen.

### Alternativ
- **Parametrisk MC** (ren normal eller Student-t) — enklare men fångar inte alla svans­egenskaper.
- **Historical simulation** (utan GARCH-filtrering) — förutsätter i.i.d. avkastningar, missar clustering.
- **Extreme Value Theory** (McNeil & Frey 2000) — hybrid som modellerar svansarna separat med Generalized Pareto.

## 6. Modelldiagnostik

### Ljung-Box-testet
**Ljung & Box** (1978), *"On a Measure of Lack of Fit in Time Series Models"*, *Biometrika*. Testar om det finns kvarvarande autokorrelation i de kvadrerade standardiserade residualerna. Lågt p-värde ⇒ modellen missar volatilitetsdynamik.

### ARCH-LM
**Engle** (1982). Lagrange multiplier-test för kvarvarande heteroskedasticitet. Komplement till Ljung-Box.

### Persistence
`α + γ/2 + β`. Närmar sig 1 ⇒ chocker dör ut extremt långsamt (integrerad GARCH, Engle & Bollerslev 1986). ≥ 1 ⇒ icke-stationär, simuleringar blir opålitliga.

## 7. Kelly-kriteriet & Half Kelly

**John Kelly Jr.** (1956), *"A New Interpretation of Information Rate"*, *Bell System Technical Journal*. Formeln för optimal positionsstorlek som maximerar långsiktig geometrisk tillväxt:

```
f* = (p·b − q) / b
```

där `p` = win rate, `q = 1 − p`, `b` = win/loss-kvot (risk/reward).

### Varför Half Kelly?
**Edward Thorp** (1969, *"Optimal Gambling Systems for Favorable Games"*) populariserade Kelly i handel, men påpekade också:

- Full Kelly har hög volatilitet — 50% drawdowns är normala.
- Estimeringsfel i `p` och `b` gör Full Kelly instabilt.
- Half Kelly (`f*/2`) ger ~75% av tillväxten med ~25% av variansen.

`montecarlo.py` rapporterar Half Kelly av dessa praktiska skäl. **MacLean, Thorp & Ziemba** (2011), *The Kelly Capital Growth Investment Criterion*, World Scientific — definitiv referens.

## 8. Break-even win rate & EV

För en strategi med given risk/reward `R` är minsta win rate för lönsamhet:

```
p_break-even = 1 / (1 + R)
```

Härlett direkt ur `EV = p·R − (1−p)·1 = 0`. Trendföljande strategier fungerar ofta med `p < 50%` så länge `R > 1`, eftersom tight stop-loss ger asymmetrisk payoff — en central insikt från **Van Tharp** (*Trade Your Way to Financial Freedom*, 1999) och **Ed Seykota**.

## 9. Begränsningar & utvidgningar

### Gap-risk
När en bootstrappad dagsavkastning tar priset förbi stoppnivån bevaras det simulerade priset som exit-kurs (snarare än att klampa till stoppet), vilket approximerar slippage vid gap. Verkliga overnight-gap (t.ex. efter rapport) kan dock vara större än något enskilt bootstrappat dagsvärde. **Merton** (1976), *"Option Pricing When Underlying Stock Returns Are Discontinuous"*, *JFE*, föreslog *jump-diffusion* som alternativ — lägger till en Poisson-process av hopp ovanpå GBM. **Kou** (2002) generaliserade med dubbelsidiga exponentiella hopp.

### Regimeskiften
GARCH fångar volatilitetsdynamik *inom* en regim. Strukturella brott (2008, covid, QE-slut) är per definition inte i historiken på samma sätt. **Hamilton** (1989), *"A New Approach to the Economic Analysis of Nonstationary Time Series"*, *Econometrica* — Markov-switching-modeller som alternativ, där volatiliteten hoppar mellan diskreta regimer.

### Modellrisk
**Derman & Wilmott** har båda varnat för övertro på MC-output: precisionen i siffrorna döljer osäkerheten i modellvalet. Simuleringsresultat ska läsas som *villkorliga* på GARCH-antagandet, inte som sanningar.

### Stationäritets-antagande
Bootstrappen antar att de historiska residualerna är representativa för framtiden. För långa horisonter och i förändrade marknadsförhållanden är detta svagt. Skriptet begränsas till kort swing-tidshorisont (default 20 dagar) delvis av detta skäl.

## Sammanfattning

| Byggsten | Källa | År | Roll i `montecarlo.py` |
|----------|-------|------|-------------------------|
| Monte Carlo-metoden | Ulam, von Neumann, Metropolis | 1946–49 | Hela simuleringsramverket |
| MC i finans | Boyle | 1977 | Motivering |
| Geometrisk Brownsk rörelse | Samuelson / Black-Scholes | 1965 / 1973 | Log-avkastningar, `exp(rₜ)` |
| ARCH | Engle | 1982 | Volatilitetsmodell |
| GARCH | Bollerslev | 1986 | `p=1, q=1` |
| GJR-GARCH | Glosten, Jagannathan, Runkle | 1993 | `o=1` (leverage) |
| Student-t-innovationer | Bollerslev | 1987 | `dist="t"` |
| Filtered Historical Simulation | Barone-Adesi, Giannopoulos, Vosper | 1999 | Bootstrap från `std_resid` |
| Volatility clustering | Mandelbrot / Fama | 1963 / 1965 | Empirisk motivering |
| Leverage-effekten | Black | 1976 | Motivering för GJR |
| Ljung-Box-testet | Ljung & Box | 1978 | Diagnostik |
| Kelly-kriteriet | Kelly | 1956 | Positionsstorlek |
| Half Kelly | Thorp | 1969 | Robust positionsstorlek |
| Jump-diffusion | Merton | 1976 | Alternativ för gap-risk |
| Regime-switching | Hamilton | 1989 | Alternativ för regimeskiften |

## Vidare läsning
- Metropolis & Ulam (1949), "The Monte Carlo Method", *JASA* 44(247)
- Boyle (1977), "Options: A Monte Carlo Approach", *JFE* 4(3)
- Engle (1982), "Autoregressive Conditional Heteroskedasticity", *Econometrica* 50(4)
- Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity", *Journal of Econometrics* 31(3)
- Glosten, Jagannathan & Runkle (1993), "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks", *Journal of Finance* 48(5)
- Barone-Adesi, Giannopoulos & Vosper (1999), "VaR without Correlations for Portfolios of Derivative Securities", *Journal of Futures Markets* 19(5)
- Kelly (1956), "A New Interpretation of Information Rate", *Bell System Technical Journal* 35(4)
- Thorp (1969), "Optimal Gambling Systems for Favorable Games", *Review of the International Statistical Institute* 37(3)
- MacLean, Thorp & Ziemba (2011), *The Kelly Capital Growth Investment Criterion*, World Scientific
- McNeil, Frey & Embrechts (2015), *Quantitative Risk Management*, Princeton University Press — kapitel om GARCH, FHS och EVT
- Hamilton (1994), *Time Series Analysis*, Princeton University Press — standardreferens för ARCH/GARCH/regime-switching
