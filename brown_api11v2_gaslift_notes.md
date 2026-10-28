# Brown Vol. 2a (Ch. 3, Gas Lift) + API RP 11V2: technical digest for continuous gas lift

Sources: Kermit E. Brown, *The Technology of Artificial Lift Methods, Vol. 2a* (PennWell 1980). OCR'd one file per page,
PDF page = book page + 14. API RP 11V2 (2nd ed., 2001 reprint), *Gas-Lift Valve Performance Testing*.
Page references are **book pages** ("B p.NNN") for Brown and **RP pages** for 11V2.
Companion digests (not repeated here): `api11v6_digest.md` (11V6 procedures, Ct table, examples) and `papers_digest.md`
(Winkler-Eads, Asheim, Alhanati, Kanu, ...).

Conventions: Brown's pressures are psig unless noted. Brown writes `Pc` for casing (injection) pressure and `Pt` for tubing
(production) pressure regardless of the flow path (B p.96). **[?]** marks something I could not read reliably from the
OCR, or a value that disagrees with the book's own arithmetic.

Skipped (exist in the book, not continuous single-string lift): intermittent design 3.43 (B p.260-320), chambers
3.35/3.43512, macaroni 3.36, duals 3.37/3.45 (B p.326-336), pack-off 3.38, two-step gas lift 3.428 (B p.211-223),
compressor systems 3.5 (B p.336-355), intermittent troubleshooting and acoustic surveys (B p.372-435).

---------------------------------------------------------------------------------------------------------------------

## 1. Valve mechanics (Brown 3.2, B p.96-121)

### 1.1 Classification (B p.96-98)
* **Casing-pressure-operated ("pressure") valve**: 50-100 % sensitive to Pc when closed, 100 % to Pc when open. Opens
  on a Pc build-up and closes on a Pc reduction.
* **Throttling pressure valve** (the "proportional" or "continuous flow" valve): the same as a pressure valve when closed;
  **tubing-sensitive when open**. It closes on a reduction in Pt (or in Pc).
* **Fluid (tubing-pressure-operated, PPO) valve**: 50-100 % sensitive to Pt when closed, 100 % to Pt when open.
* **Combination valve** (fluid-opened, pressure-closed).
* Continuous-flow requirement (B p.98): when open, the valve should respond in proportion to Pt. It should throttle
  closed as Pt falls and open as Pt rises (Fig. 3.21), which holds the established flowing tubing pressure. "The opening
  mechanism is of little consequence for continuous flow ... as long as the valve can be opened under operating casing
  and tubing pressure combinations."

### 1.2 Unbalanced bellows valve, N2 dome only (3.2511-3.2515, B p.101-105)
Symbols: `Ab` is the effective bellows area (in²). `Ap` is the seat (port) area (in²). `R = Ap/Ab`. `Pd` is the dome
pressure at valve temperature (psig). `Pvo` is the casing pressure at depth that opens the valve, and `Pvc` the one at
which it closes. `Pt` is the tubing pressure at depth.

Force balance just before opening (Eq. 3.23-3.26), with Pt acting on the port:
```
Pc (Ab - Ap) + Pt Ap = Pd Ab
Pvo = (Pd - Pt R) / (1 - R)                       (3.27)
    =  Pd/(1-R) - Pt R/(1-R)                      (3.28)
TE  = Pt R/(1-R)       "tubing effect"            (3.29)
TEF = R/(1-R)          "tubing effect factor"     (3.210)   (= 11V6 PPEF)
```
Just before closing (Eq. 3.212-3.215), Pc is under the ball as well:
```
Pvc = Pd                                          (3.215)
Spread  dP = Pvo - Pvc = TEF (Pd - Pt)            (3.216, 3.217)
Max spread (Pt = 0) = TEF * Pd
```
Test-rack opening (Pt = 0, Eq. 3.221-3.222):
```
Ptro = Pd@60F / (1 - R)                            (3.222)
Pd@60F = Pd * z60 T60 / (zT T)  (real gas, 3.223);  in practice Pd@60F = Ct * Pd(T)  (Table 3D.1)
```
The Ct values quoted in the examples agree with the API linear form `Ct = 1/(1+0.00215(T-60))`: 148 F gives 0.841 and
180 F gives 0.795. One Brown table is consistent with Ct(126 F) of about 0.870, against 0.876 from the formula
(see 2.6.2) **[?]**.

**Examples (regression tests)**
| # | Data | Answers |
|---|---|---|
| Ex 1 (B p.102) | Pd = 700, Pt = 500 at 6000 ft, Ab = 1.0, Ap = 0.1 (R = 0.1) | Pvo = (700-50)/0.9 = **722 psi** |
| Ex 2 (B p.102) | same | TEF = 0.1111; TE = **56 psi**; "max opening" (Pt = 0) = **778 psi** |
| Ex 3 (B p.103) | same | spread = 0.1111(700-500) = **22 psi**; Pvc = 700 |
| Ex 4 (B p.104) | Pso 800, 8000 ft, 0.65 gas, Tavg 120 F | F = 0.1979 (chart, 80 F basis); Fcorr = 0.1979*540/580 = **0.1843**; Pc(8000) = 800*1.1843 = **947 psi**; gradient 18 psi/1000 ft |
| Ex 5 (B p.104) | surface opening 750, 6000 ft, 0.60 gas, Tavg 110 F **[?: read from (110+460)]**, Pt 500, R 0.1 | F = 0.1330, Fcorr = 0.126; Pvo(depth) = 750*1.126 = **845**; Pvc = 845(0.9)+500(0.1) = **811 (810.5)**; surface close = 811-95 = **716** (720 if the gradient at the lower pressure is used); spread 34 at depth |
| Ex 6 (B p.105) | Ex 5 with T = 70+1.3 F/100 ft, so 148 F at 6000 ft | Ct = 0.841; Pd60 = 0.841*811 = **682**; Ptro = 682/0.9 = **758 psi** |
| 3.2516 summary (B p.105-106) | 8000 ft, Pso 800, 0.7 gas, Ts 100 F / Tv 180 F, Pt 655, 1.5-in valve with 1/2-in port **[OCR "4%"; R confirms 1/2-in on Ab 0.77]**, R = 0.2562 | F = 0.2145, Fcorr = 0.2145*540/600 = 0.1931, Pvo = 800*1.1931 = **955**; Pvc = 955(0.7438)+655(0.2562) = **879**; spread at depth **76** (TEF 0.344); surface close 879/1.1931 = **737**; surface spread **63**; average **70**; Ct(180) = 0.795, Pd60 = 699, Ptro = 699/0.7438 = **940 psi** |

### 1.3 Gas pressure at depth (3.2514, B p.103-104)
Static column, average T and z (Eq. 3.220):
```
P_depth = P_surface * exp( 0.01877 * gamma_g * L / (Tavg[R] * zavg) )          (3.220)
```
In the worked examples Brown uses the chart (App. 3A.1) form `P_depth = P_surf (1+F)` with F read at an 80 F basis and
corrected by `F_corr = F * 540/(Tavg+460)`. **Observation:** the chart values reproduce the ideal-gas form with z = 1,
`F = exp(0.01877 gamma L / 540) - 1`. For 0.70, 8000 ft this gives 0.2149 (chart 0.2145). For 0.65, 8000 ft it gives
0.1981 (chart 0.1979). The linear correction 540/T is an approximation. Gradients quoted elsewhere for kick-off lines
(for example 26.2 psi/1000 ft at 1000 psi, 0.7 gas, B p.227) are larger and do include z. Use the examples only as
loose (+/-2 %) tests of a real-gas column. Annulus friction is normally negligible. For annular flow with gas injected
down 2-3/8-in tubing, friction offsets the column weight at about 2.75 MMscf/d (Pso 1000, 0.60 gas) (B p.145).

### 1.4 Dome + spring (double-element) pressure valve (3.252, B p.106-109)
`St` is the spring pressure equivalent acting on (Ab - Ap).
```
Pvo = (Pd + St(1-R) - Pt R) / (1 - R) = (Pd - Pt R)/(1-R) + St                  (3.226)
spring only (Pd = 0):  Pvo = St - Pt R/(1-R)                                       (3.227)
Pvc = Pd + St (1 - R)                                                              (3.230)
spread = TEF [Pd + St(1-R) - Pt]                                                   (3.231)
Ptro  = Pd@60F/(1-R) + St      (spring needs no temperature correction)
```
* **Summary Ex 1, spring only (B p.107-108)**: same well as 3.2516. St = (Pvo(1-R)+Pt R)/(1-R) = 955 + 655(0.2562/0.7438)
  = **1181**; Pvc = 1181(0.7438) = **879**; spread **76**; surface close **737**; **Ptro = St = 1181 psi**.
* **Summary Ex 2, St = 600 (B p.109)**: Pvc = **879**; Pd = 879 - 600(0.7438) = **433**; Pd60 = 0.795*433 = **344**;
  Ptro = 344/0.7438 + 600 = **1062 psi**. Setting procedure: set the spring to open at 600 with no dome charge, charge
  the dome, cool to 60 F, then bleed until the valve opens at 1062.

### 1.5 Balanced and pilot valves (B p.109-111)
* A balanced valve (Fig. 3.213, or the flexible-sleeve valve of Fig. 3.214) opens and closes at the dome pressure, with
  zero spread for any port. It behaves as an expanding-orifice regulator. Balanced valves need a drop in surface
  opening pressure between valves, 25 psi in Brown's examples (B p.236, 242).
* Pilot valves (intermittent use): the control port sets R and the spread, and a separate power port carries the gas.

### 1.6 Gas passage through a port (3.255, B p.111)
Craft, Holden and Graves (Thornhill-Craver form):
```
Q[Mscf/d @14.65 psia, 60F] = 155.5 Cd A P1 sqrt( 2 g k/(k-1) (r^(2/k) - r^((k+1)/k)) ) / sqrt(gamma T1[R])
r = P2/P1 ; r_c = (2/(k+1))^(k/(k-1)) ; k ~ 1.31 for natural gas (Brown) ; critical r ~ 0.55
```
Brown warns that a valve is a series of chokes, so it passes less gas than these charts predict.
The chart correction used in all the design examples is `CF = 0.0544 sqrt(gamma T[R])` (Eq. 3.410). The chart volume is
`CF * q_actual`. Note that 1/0.0544² = 338 = 0.65 x 520, so **Fig. 3C.1 is built on a 0.65-gravity, 60 F basis**.

### 1.7 Throttling (continuous-flow) pressure valve (3.26, B p.111-112)
The valve is made tubing-sensitive when open by choking the entrance upstream of the seat, or by a tapered stem or
seat. Once open, it closes on a fall in Pt even with Pc held constant (Fig. 3.220: Pc held at 800, the valve closes at
Pt = 300). The linear part of q vs Pt is the **throttling range** ("proportional response"). Large-ported standard
valves also show throttling (Fig. 3.221 with a 3/8-in port; Fig. 3.230 with a 16/64-in port closing on Pt at constant
Pc). B p.138 adds that small ports (<= 1/4 in) approach balance and are less tubing-sensitive; larger ports (>= 3/8 in)
are more tubing-sensitive and recent tests show they throttle better. Standard intermittent valves used in continuous
flow must be choked or they intermit (B p.138; recorder chart Fig. 3.618G).

### 1.8 Fluid (PPO) valve (3.27, B p.112-116)
Pt acts on (Ab - Ap) and Pc on Ap:
```
Pt_open = (Pd + St(1-R) - Pc R)/(1-R) = Pd/(1-R) + St - Pc R/(1-R)       (3.235)
CE  = Pc R/(1-R)  ;  CEF = R/(1-R)                                       (3.236-3.238)
Pt_close = Pvc = Pd + St(1-R)       (port must choke Pc down to Pt when open)   (3.242)
Ptro = Pd@60F/(1-R) + St            (Pc = 0 in the rack)                (3.243, 3.244)
```
* **Ex 8 (B p.115)**: 8000 ft, Tv 180 F, Pvc = 1800, St = 600, R = 0.0407. Pd = 1800 - 600(0.9593) = **1224**;
  Pd60 = 0.795(1224) = **973**; Ptro = 973/0.9593 + 600 = **1614 psi**.
* **Summary, spring only (B p.116)**: Pso 800 gives Pc(8000) = 955, R = 0.0407, Pvc = 800 = St (dome 0).
  Pt_open = 800/0.9593 - 955(0.0407/0.9593) = **793 psi**; Ptro = St/(1-R) = 800/0.9593 = **834 psi**
  **[?: the book divides St by (1-R) here, unlike Eq. 3.243; follow the book number only as a check]**.
* **Summary, dome only**: Pd = 800, Pt_open = **795 psi**; Pd60 = 0.795(800) = 636; Ptro = 636/0.9593 = **663 psi**.
* **Summary, dome + St = 300**: Pd = 800 - 300(0.9593) = **512**; Pt_open = 512/0.9593 + 300 - 955(0.0424) = **793**;
  Pd60 = 0.795(512) = **407**; Ptro = 407/0.9593 + 300 = **724 psi**.
* Gas passage (3.275, B p.117): the port must be small enough that the whole bellows senses Pt. Within the throttling
  range the valve is a **variable orifice** whose size grows with Pt (Fig. 3.226). For larger ports the throttling range
  starts at the opening pressure.

### 1.9 Differential valve (3.276, B p.117-118)
`Pt_open = Pc - Sd` and `Pt_close = Pc - Sd`, with spring differential Sd of 150 psi typical (50-500). The valve is
normally open. With a packer, provide U-tubing communication (sliding sleeve or similar). Example: Sd = 150, Pc = 850,
so the valve opens at Pt = 700.

### 1.10 Dynamic considerations (3.29, B p.119-121)
* Bellows travel is limited by protection devices. The stem must travel some minimum distance before the equivalent
  area, the frustum of a right cone (Fig. 3.229, ball OD = seat ID + 1/16 in), reaches the port area. Gas flow can
  keep rising beyond that travel for some seat designs.
* **Bellows load rate** = pressure change per unit stem travel (psi/in). It should be linear over the operating range.
  Stacking of the convolutions, small dome volumes and high dome pressures (> about 1500 psi) cut it prematurely and
  reduce gas passage. Quantified in API 11V2 Section 4 and App. C (Part 4 below).
* A single-element valve in real flow closes at a *higher* Pc than the force balance predicts, because the pressure
  under the ball falls toward Pt as the ball nears the seat. Small ports with downstream choking do not show this.

### 1.11 Valve-setting rules of thumb (B p.144, 224, 230)
A drop of 15-25 psi per valve in surface opening or closing pressure (B p.144); at least 10 psi per valve for single
completions (B p.224); 10-20 psi drop in surface **closing** pressure, with larger ports needing less (B p.230).
Unbalanced valves can be set at the same surface opening pressure because "the lower valve [opens] first because of
the higher tubing back pressure acting against it" (B p.144). Balanced valves must take a drop. Continuous-flow valves
often have non-constant closing pressures, so they must be set on opening pressures (B p.224).

---------------------------------------------------------------------------------------------------------------------

## 2. Continuous-flow design (Brown 3.42, B p.137-259)

### 2.1 Basics, top valve, rules of thumb (3.421-3.4229, B p.137-144)
* `Pwf = Pwh + Ga L + Gb (D - L)`, where Ga and Gb are the average flowing gradients above and below the injection
  point (Eq. 3.41).
* A design rate is a criterion that the unloading valves can reach the optimum injection point *while the well feeds in
  at that rate*. It does not mean the well will make that rate. A high design rate only adds one or two valves (B p.137).
* Top valve (B p.138-139): it should be "as deep as practical". With kill fluid to surface it goes at
  `(Pko - Pwh)/Gs`, neglecting the gas column. Example: 800 psi and 0.50 psi/ft balance 1600 ft, or 1400 ft with
  100 psi back pressure. If the static fluid level is lower (3000 ft, say), place the valve at the level or slightly
  below it, never below. If the well can be unloaded against a lower pressure (to pit or atmosphere) for the first
  valve, it goes deeper.
* Rules of thumb (3.4229, B p.144):
  * pressure: about 100 psi per 1000 ft, minimum 300, maximum about 1500 psi (1800-2200 has been used; above 1800 is not
    recommended for valve life);
  * gas: 150-250 scf/B per 1000 ft of lift;
  * maximum continuous-flow depth for 2-3/8 and 2-7/8 tubing: `(Pso - Pwh)/0.15` (Eqs. 3.42/3.47). Example:
    (900-100)/0.15 = **5333 ft**. Annulus: `(Pso - Pwh)/0.0x` **[? divisor unreadable, Eq. 3.48]**;
  * Table 3.42 lists maximum and minimum rates for good continuous flow by tubing size: 2-in 2500 / 200-250;
    2-1/2-in 3000 / 350-500; 3-in 4000 / 500-750; 4-in 10000 / 1000-1500 B/D. Useful as UI warnings.
* Temperature (3.4225, B p.139-143): use a flowing temperature, not geothermal. Kirkpatrick's chart is linear. Shiu's
  correlation gives the Ramey relaxation distance:
  `A = 0.0075065 W^0.4882 d^-0.3476 API^0.2519 gamma_g^4.7240 rho_L^2.9303 Pwh^0.2219` **[exponents partly unreadable;
  the example result is A = 2085.7 ft]**, with
  `T_surf = T_bh - g_G Z + g_G A (1 - e^(-Z/A))`.
  * Shiu example: qo 2218.85, qw 11.15 STB/d, qg 1761.17 Mscf/d, d = 3.0 in, 24 API, gamma_g 0.7, gamma_w 1.10,
    gG = 1.0625 F/100 ft, Z = 5985 ft, Tbh 173 F, Pwh 150 psig, measured Tsurf 134 F.
    Kirkpatrick gives 138.29 F; Shiu gives W = 9.329 lbm/s **[?]**, rho_L = 56.781 lb/ft³, A = 2085.7 ft and
    **Tsurf = 130.3 F** **[OCR reads "180.3"; arithmetic gives about 130]**.

### 2.2 Point of gas injection (3.4231, B p.144-147). Defines the operating point.
Procedure for a single design rate:
1. Static BHP at depth.
2. Drawdown for the rate from the IPR; Pwf = Pr - dP.
3. Flowing traverse *below* injection from Pwf, at the formation GLR.
4. Operating (not kick-off) surface pressure, extended down with the gas column.
5. Where the casing line crosses the flowing gradient is the **point of balance**.
6. **Subtract 100 psi** and move up the formation traverse to the depth where Pt = Pc - 100. That is the **point of
   gas injection**. "The selection of this differential (100 psi here) is extremely important and is controlled by the
   valve spacing."
7. Find the total GLR whose traverse connects Pwh to (Dinj, Pt_inj).
8. Injection gas = (TGLR - Rs) q.
9. Size the operating port.

**Example Problem #9 (B p.145-146; Fig. 3.48)**
* Data: 8000 ft, 1000 B/D oil, 2-3/8 tubing, Pwh 100, Pr 2650, PI 2 (constant), Rs 200, 0.70 gas, Pso 900, 40 API,
  BHT 200 F, surface flowing 120 F.
* Results: drawdown 500, Pwf **2150**; static level 1978 ft (0.357 psi/ft); gas gradient 21.2 psi/1000 ft;
  balance at **4800 ft**; injection at **4625 ft**; TGLR **600** scf/B; gas = (600-200)(1000) = **400 Mscf/d**.
* Port: Tinj = 120 + 80(4625/8000) = 166 F, but the book uses 176 F (CF = 0.0544 sqrt(0.7 x 636) = **1.148**)
  **[? book inconsistency; 166 F gives 1.139]**. Corrected volume **459,131** scf/d. Upstream/downstream 1000/900 psi
  gives a port between 11/64 and 12/64, so **12/64 in** is selected.

### 2.3 Rate possible by gas lift: pressure-flow-rate diagram and equilibrium curve (3.4232, Regnault/Marin, B p.147-161)
*Constant wellhead pressure.* Both methods use a gas-injection line equal to the casing line minus 100 psi, parallel to
the operating-pressure line.
* **Pressure-flow-rate diagram** (3.423221, B p.148-149). For each assumed q:
  1. Traverse down from Pwh at the TGLR. Unlimited gas means the minimum-gradient GLR. Limited gas means
     `GLRinj = Qgas/q` (3.412) and `TGLR = GLRinj + Rs` (3.413).
  2. The injection point is where that traverse meets the injection line.
  3. Traverse at the formation GLR down to the perforations gives Pwf(q).
  4. Plot Pwf(q) against the IPR. The intersection is the rate, and depth-vs-rate gives the injection depth.
* **Equilibrium curve** (3.423222, B p.149-150). For each q:
  1. Pwf from the IPR, traversed *up* at the formation GLR.
  2. Traverse *down* from Pwh at the TGLR (minimum gradient, or limited).
  3. Their intersection is an "equilibrium point". The locus over q is the equilibrium curve.
  4. Its intersection with the injection line gives the maximum injection point. Interpolate Pwf, then q = PI(Pr - Pwf).
  * The curve is independent of Pso, so it also shows the effect of other injection pressures, gas volumes, and future
    Pr or PI (B p.149).
* **Example (Marin, B p.150-154)**. Data: 8000 ft, 2.441-in ID, Rs 200, PI 3 (linear), Pso 900, Pr 2400, Pwh 120
  constant, 0.65 gas, 100 % oil; gas column 21.3 psi/1000 ft, so 1070 at 8000 ft. From gradient curves:

  | Case | q | TGLR | Pwf HB @ inj | Pwf Ros @ inj | Note |
  |---|---|---|---|---|---|
  | Unlimited (T3.43) | 1000 | 1500 | 1400 @ 6680 | 1005 @ 7600 | |
  | | 1500 | 1500 | 1900 @ 6350 | 1600 @ 6050 | |
  | | 2000 | 1200 | 2260 @ 4300 | 2100 @ 4500 | |
  | IPR (T3.45) | 1000 / 1500 / 2000 | | Pwf 2060 / 1900 / 1730 | | |
  | Max rate (T3.44) | **1500** (HB) / **1710** (Ros) | | Pwf 1900 / 1830 | | inj **5350 / 5325** ft |
  | Equilibrium pts (T3.46) | 1000 / 1500 / 2000 | | 3600 / 5300 / 7825 ft (HB) | 2260 **[?]** / 4500 / 6625 ft (Ros) | max injection 5300 ft (Fig. 3.411) |
  | Limited gas | 600 / 400 / 200 Mscf/d | GLRinj at 1000/1500/2000 B/D = 600/400/300 (600 Mscf/d) | | | T3.48 |
  | Pressure-rate result (T3.410, HB) | **1420 / 1320 / 1190** **[?: OCR 1790]** | | Pwf 1930 / 1975 / 2010 | | |
  | Pressure-rate result (T3.410, Ros) | 1586 / 1440 / 1260 | | Pwf 1875 / 1920 / 1980 | | |
  | Equilibrium result, 600 Mscf/d (T3.412) | HB **1395** / Ros 1560 | | Pwf 1935 / 1880 | | inj 5230 / 5220 ft |

  These come from gradient curves, so use them only as +/-5-10 % sanity checks on a performance-curve solver.

*Variable wellhead pressure* (Marin, 3.42323, B p.155-161). Horizontal (Eaton) and vertical correlations are combined.
For each (q, TGLR) the "permissible" Pwh comes from the vertical traverse and IPR, and the "required" Pwh from the
flowline. Equivalently, compute Pwh from the flowline, then Pwf, and intersect with the IPR. The optimum TGLR is
**below** the minimum-gradient GLR because extra gas adds flowline friction.
* **Example #10 (B p.157-160)**. Data: 7000 ft, 3-in tubing, 3000 ft 3-in flowline, Pr 2600, Pso 900, Psep 80, PI 5,
  GOR 400 (GLR 200), 50 % water, 0.70 gas.
  * Maximum rate: HB **2360-2365 B/D** at TGLR 600, Pwh 260, Pwf 2120; Ros 2160 at 600.
  * HB rate vs TGLR (T3.418): 400: 2290; 600: 2365; 800: 2280; 1000: 2180; 1500: 1830. Ros: 2120, 2160, 2140, 2040, 2000.
* **Example #11 (Fonseca, B p.160-161, Table 3.419; Fig. 3.422-3.423)**. Data: 8000 ft, Pr 2400, PI 3, Pso 900, 35 API,
  Psep 80, Rs 200, 4000-ft flowline, 2-3/8 **[?]** tubing. A complete gas-lift performance curve (the total GLR column
  equals gas/q + 200, and q = 3(2400 - Pwf), so the rows are internally consistent):

  | Qgi Mscf/d | TGLR | q B/D | Pwf | Dinj ft | Pwh |
  |---|---|---|---|---|---|
  | 100 | 298 | 1024 | 2059 | 4553 | 112 |
  | 200 | 369 | 1180 | 2007 | 4724 | 127 |
  | 300 | 436 | 1273 | 1976 | 4833 | 140 |
  | 400 | 501 | 1330 | 1957 | 4894 | 150 |
  | 500 | 566 | 1366 | 1945 | 4937 | 160 |
  | 600 | 633 | 1387 | 1938 | 4968 | 170 |
  | 700 | 700 | 1400 | 1934 | 4973 | [?] |
  | 800 | 768 | 1408 | 1931 | 4984 | 187 |
  | 900 | 838 | 1414 | 1930 | 4983 | 195 |
  | 1000 | 909 | 1411 | 1930 | 4987 | 203 |
  | 1200 | 1056 | 1402 | 1933 | 4977 | 219 |
  | 1400 | 1211 | 1385 | 1939 | 4957 | 236 |
  | 1600 | 1376 [?] | 1360 | 1947 | 4930 | 249 |
  | 1800 | 1554 | 1330 | 1957 | 4895 | 264 |
  | 2000 | 1742 | 1297 | 1968 | 4853 | 278 |

  Maximum is 1412 B/D at 4985 ft and about 1 MMscf/d; there is little gain above 600 Mscf/d.

### 2.4 Economics and optimum gas (3.424-3.426, B p.162-202). Methods only.
* 3.424 (Regnault): project economics only (payout, ROR, NPV). Not a design rule.
* **3.425 Economical slope (Mitchell; Pacheco for variable Pwh, B p.166-185).** The most economical TGLR is reached when
  the profit from the oil lifted by one more unit of TGLR equals the cost of that gas. On the curve of tubing pressure
  at the injection depth vs TGLR, the economic point is where the slope equals
  ```
  X = dP_inj/dTGLR = q G / (PI * P)      [psi per scf/STB]        (3.416, 3.416B)
  q B/D, G $/scf, P = profit $/bbl oil, PI B/D/psi ;  assumes dP_inj ~ dPwf
  ```
  The zero-slope point is the minimum-gradient (maximum) TGLR. Short cut (Tables 3.430-3.431): economic TGLR equals the
  minimum-gradient TGLR minus a tabulated difference that depends on tubing ID, rate and slope.
  * **Ex 12 (B p.169-170)**: 2-7/8 tubing, Pwh 100 psia, 2000 B/D, wc 0, PI 1, $4/bbl, $0.2/Mscf, injection at 5000 ft.
    X = 2000 x 0.0002/(1 x 4) = **0.1**. Minimum-gradient TGLR is 1450; the table difference is 200, so the economic
    TGLR is **1250 scf/B**.
  * **3.42523 example (B p.170-173)**: 2.441-in, 8000 ft, Pr 2400, Pwh 100 psia, PI 3, 30 API, Rs 200, profit $6/bbl,
    gas $0.1/Mscf. X = 1500(0.0001)/(3 x 6) = **0.008**; 2000 B/D gives 0.01 and 2500 B/D gives 0.012.
    Gas gradients 26.1 / 23.6 / 19 psi/1000 ft for Pso 1200 / 1000 / 800 psia (1409 / 1189 / 952 at 8000 ft).
    Results (T3.432): economic rates **2030 / 2400 / 2450 B/D** for Pso 800 / 1000 / 1200.
  * **Pacheco example, variable Pwh (B p.179-185)**: 8000 ft, 2.441 tubing, 4000 ft x 2.5-in flowline, Pr 2400, PI 3,
    Psep 80, GLR 200, wc 0, $4/bbl, $1/Mscf. X(500 B/D) = 500 x 1/(4 x 3 x 1000) = **0.0417**. Natural flow
    **632 B/D**, Pwf 2189.7, Pwh 91.1.
    * Method I (T3.433): Pso 800/1000/1200 give injection at 3950/4900/5800 ft, 770/1015/1265 psia, TGLR 450/525/625,
      **885/1025/1180 B/D**.
    * Method II (incremental profit equals gas cost): **852/1008/1164 B/D** at TGLR 434/515/584.
    * T3.434 (Pso 800) performance rows (q, GLRinj, TGLR, Pinj, Dinj, Pwh, Pwf): (728.3, 50, 250, 758.7, 3676.8, 98.1,
      2157.2) ... (845.7, 200, 400, 760.7, 3805.2, 115.8, 2118.3); (853.7, 250, 450, 760.0, 3815.2, 120.2, 2115.3);
      (853.7, 300, 500, 761.0, 3816.0, 123.8, 2115.1). These are a good regression set for a variable-Pwh performance
      curve.
* **3.426 Optimization (Gomez/Hong, B p.186-202)**: the same performance-curve loop (natural flow first, then add GLR in
  steps of 50-300 scf/B; inject at the bottom if Pc - 100 >= Pwf, else march up in 200-ft steps to the casing/tubing
  crossing). Fit q(Qgi) with a quadratic. Allocation over a field: if the sum of the per-well maxima fits in the gas
  available, give each well its maximum. Otherwise start equal and hand increments to the well with the largest
  incremental oil (a greedy equal-slope allocation). Field case: 13 wells, 30 MMscf/d, 32,168 STBO/d. Same idea as Kanu
  (see `papers_digest.md`).

### 2.5 Operating differential and spacing near the injection point (3.4292, B p.224-226; summary 3.4211, B p.257)
* Pressure-flow-rate results are **very sensitive to the assumed "operating differential" dP**, the injection pressure
  at depth minus Pt at depth (Fig. 3.476-B). The sensitivity grows with PI. The designer controls dP **through valve
  spacing**, not through the port: `dP_max = Gs * dD` (Gs is the flowing gradient below injection).
  Example: dP = 100 psi with Gs = 0.40 gives spacing of **250 ft** near the injection point. The table gives
  dP 50 → 125 ft, 100 → 250, 150 → 375, 200 → 500, 300 → 750 ft.
* "This operating differential is often confused with the pressure drop across a port ... Once a desirable operating
  differential is selected, the designer must select a port size sufficient to pass the necessary amount of gas with the
  selected casing-to-tubing differential." If no port is big enough, a larger dP must be chosen (B p.226).
* Do not space valves closer than 3 joints (about 90 ft) (B p.257). After surveys, 2-3 joints have been used on high-PI
  water-drive wells: 90 ft deeper changes Pwf by (0.45 - 0.15)(90) = 27 psi, which is **270 B/D at PI 10** (B p.365).
  With valves 3 joints apart, 2-3 valves may pass gas, which "is not critical" (B p.257).

### 2.6 Spacing procedures (3.429, B p.223-259)
Objectives (B p.224): inject as deep as possible; conserve casing pressure; all upper valves closed once at the
operating point; able to open a valve with kick-off pressure or less whatever the tubing level.

**Common example data (3.42921, B p.224).**
* 2-7/8 OD tubing, 0.70 gas, kill fluid 0.40 psi/ft, GOR 400 scf/B, 50 % water (GLR 200), Pr 1920, 35 API, PI 3,
  linear down to Pb **[? Pb value lost]**, Pwh 120.
* Pso 900 (950 available), Pko 1000 (1050 available), 8000 ft, desired rate = maximum, BHT 170 F, surface flowing
  109 F. The geothermal gradient prints as "1.2 F/100 ft" **[? conflicts with 170 F at 8000 ft]**.
* Maximum rate **1000 B/D at a TGLR of about 1000 scf/B** (Fig. 3.475); injection at **5800 ft**, where Pc = 1037 and
  Pt = 937 (the 800-psi "Pc - 100" line meets the traverse at 940).

#### 2.6.1 Universal design, standard pressure valves, constant surface opening pressure (3.42922, B p.224-230)
1. Rate from the pressure-flow-rate diagram: 1000 B/D.
2. Injection point: 5800 ft.
3. **Design tubing-pressure line** from the surface at `Pwh + max(0.20 Pso, 200)` (some use 0.25 Pso) to Pt at the
   injection point. Here 120 + 200 = **320 psi**.
4. Valve 1: kill-fluid line (0.40) from Pwh = 120 to the kick-off line (1000 psi + 26.2 psi/1000 ft) gives
   **2350 ft**. An alternative locates it 50 psi short of the kick-off line (Fig. 3.477-B).
5. Next valves: go horizontally left to the design tubing line, then down parallel to 0.40 psi/ft to the *operating*
   casing line (Pso 900). Valves at **2350, 3350, 4150, 4750, 5250, 5650, 5900** ft.
6. **Bracket the injection point.** Envelope from Pwh +/-20 % (P1 = 144, P2 = 96) to Pt_inj +/-20 % (1128, 752). Where
   the lines P2-P2' and P1-P1' cross the (Pso - 100) line gives **4450 ft and 7850 ft**. Space valves inside the
   envelope at dP/Gs. 10 % (more realistic) or 20 % envelopes are shown; offshore, valve down to the packer.
   Final string: 2350, 3350, 4150, 4750, 5250, 5650, 5800, 5900 ft (plus envelope valves).
7. Valve temperatures on a straight line from 109 F (surface flowing, from Kirkpatrick at 1000 B/D) to 170 F:
   127, 135, 141, 145, 149, 152, 153, 154 F (Shiu for high rates).
8. Gas = (1000-200)(1000) = **800 Mscf/d**. CF = 0.0544 sqrt(0.7 x 613) = **1.127**, corrected **901.5 Mscf/d**.
9. Operating port: at Pc 1037 / Pt 937 the chart gives slightly more than 16/64, so **5/16 in** (or 9/32 if available).
10. **Unloading ports: "it is common practice to use the same port size on all valves"**. Either all 5/16 (fewer
    setting errors) or **1/4-in unloading valves with 9/32 or 5/16 on the bottom three**. No unloading gas rate is
    computed for pressure valves.
11. Settings: 1.5-in valve, Ab = 0.77. 5/16 port gives R = 0.0996 (1-R = 0.9004, TEF 0.1106); 1/4 port gives
    R = 0.0638; 9/32 port gives R = 0.08068. Use `Pvc = Pvo(1-R) + Pt_design R`, then Ct, then `Ptro = Pd60/(1-R)`.
    **Valves are set on the design tubing pressure, not the operating one.**

    Table 3.464 (all 5/16, Pso 900 for every valve):

    | # | D ft | T F | Pvo(depth) | Pt design | Pvc | Psc | Pd60 | Ptro (final) |
    |---|---|---|---|---|---|---|---|---|
    | 1 | 2350 | 127 | 955 | 570 | 917 | 862 | 800 | 886 (885) |
    | 2 | 3350 | 135 | 979 | 676 | 949 | 876 | 820 | 908 (910) |
    | 3 | 4150 | 141 | 998 | 762 | 975 | 881 | 830 | 919 (920) |
    | 4 | 4750 | 145 | 1012 | 825 | 993 | 884 | 840 | 930 |
    | 5 | 5250 | 149 | 1024 | 879 | 1010 | 888 | 845 | 935 |
    | 6 | 5650 | 152 | 1033 | 921 | 1022 | 890 | 855 | 946 (945) |
    | 7 | 5800 | 153 | 1037 | 937 | 1027 | 892 | 860 | 952 (950) |
    | 8 | 5900 | 154 | 1039 | 948 | 1027-1029 | 892 | 860 | 952 (950) |

    Check: valve 1 gives 955(0.9004) + 570(0.0996) = 916.7, and Ct(127) = 0.874 gives 801. Table 3.465 (1/4-in
    unloading, 9/32-in bottom three) gives Pvc 930, 960, 983, 1000, 1015, 1024, 1029, 1032 and Ptro 865, 880, 890, 900,
    905, 930, 935, 935.

#### 2.6.2 Pressure valves with a 10-20 psi drop in surface **closing** pressure (3.4293, B p.230-232)
Unloading logic (Fig. 3.479): while transferring, both valves must be open. Only the upper one should then close,
because the casing falls to its surface closing pressure. Each lower valve therefore needs a lower surface close. Set
the top valve's close as high as possible while it can still open with the kick-off casing pressure and the transfer
tubing pressure. Do not set it above the operating pressure if it might become the operating valve.

Procedure:
1. Draw the kick-off and operating gas lines, the design-rate traverse, and the 0.4 psi/ft kill line from Psep = 80.
2. Valve 1 sits where the kill line is **50 psi below** the kick-off casing: **2350 ft**.
3. Choose the **transfer tubing pressure on or to the right of the design traverse** (Fig. 3.479-D marks which transfer
   points are acceptable for pressure and for fluid valves): 420 psi at valve 1.
4. Set valve 1's surface close at 950 psi (the operating pressure). Its surface opening with Pt = 420 must not exceed
   Pko: the book gets about 986 against Pko 1000, "OK". The text notes that because there is no feed-in during the
   first transfer, **the first valve's port "can be based on other considerations than gas passage"**; 1/4-in
   (R 0.0720) is used.
5. Valve 2: 0.4 line from 420 psi at valve 1 to the 950 line, moved up until Pt is 50 psi below Pc, gives **3750 ft**,
   surface close 935.
6. Valve 3: from transfer 620 at valve 2 to the 935 line gives **4700 ft**, close 920.
7. Then valves every 250 ft through the bracket, dropping 15 psi per valve.

Table 3.466 depths: 2350, 3750, 4700, 5250, 5500, 5750, 6000, 6250, 6500 ft. Surface closes 950 down to 830 in 15-psi
steps. Ptro rows (valve 1) read 1049 / 935 / 1008 **[? the table is garbled: Pd60 = 935 and Ptro = 1008 agree with
R = 0.072 and Ct(118 F) = 0.889, but the implied Pvc = 1052 at 2350 ft is inconsistent with a 950 surface close;
do not use as a test]**.

#### 2.6.3 Fluid-operated valves, pressure-charged dome, universal spacing (3.4294, B p.232-233)
Use the same depths as 2.6.1. Opening is in terms of **tubing** pressure: Pt_open = the design tubing pressure at each
valve, and `Pt_close = Pvc = Pd = Pt_open(1-R) + Pc R` (3.419). `Ptro = Pd60/(1-R)` (3.420). Example: valve 7 opens
at Pt 937 and closes at Pt 945, so the closing tubing pressure exceeds the opening one. Table 3.467: Pvc 595, 695,
777, 837, 888, 930, 945, 955; Ptro 555, 640, 700, 755, 795, 850, 860, 870. 1/4-in unloading ports, 9/32 on the
bottom 3.

#### 2.6.4 Combination fluid-opened, pressure-closed valves (3.4295, B p.233-234)
Calibrate the bellows section as in 2.6.1. The spring of the differential section is the operating casing pressure
minus the design tubing pressure at each valve: 385, 305, 235, 180-190, 145, 110, 100, 100 psi (Tables 3.468-3.469).

#### 2.6.5 Spring-loaded fluid valves: the **only Brown procedure that computes unloading gas per valve** (3.4296, B p.233-236)
* Data: 8000 ft (mid-perf), Pr 1920, q 800 B/D (50 % water), 2-7/8 tubing (2.5-in ID), Pwh 120, J 4, 0.70 gas,
  BHT 170 F, surface flowing 110 F, GOR 300, Pko 1000, Pso 900, kill 0.468 psi/ft loaded to the top, unloaded to an
  80-psi separator. Injection at **5775 ft**; FBHP 1720.
* Procedure:
  1. Valve 1 where the 0.468 line from 120 psi meets the 1000-psi gas line: **2000 ft**.
  2. **Pvc (closing tubing pressure) for every valve except the operating one** lies on a line from
     `Pt_design_surface + 0.25 Pso` down to the injection point. Its opening pressure must stay above the
     design-gradient pressure, so the valves cannot interfere.
  3. **Transfer point = Pvc + 50 psi.** This keeps the upper valve passing gas when the next is uncovered (Fig. 3.480-D).
  4. The next valve is where the 0.468 line from the transfer point meets the 900-psi line.
  5. Operating valve Pvc is about 80 psi below the minimum flowing-gradient pressure, to throttle.
  6. **Port: for each valve as it is spaced, "use the smallest port size required for transfer from each location".**
     The gas required at transfer is `GLR at transfer x q_design`, and **70 % of the orifice capacity must cover it**.
* Table 3.470:

  | # | D | Pc | Pvc | GLR @ transfer | Pt transfer | Q req Mscf/d | Port /64 | Orifice cap. | 70 % | 1-R | Pd | Ptro |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 2000 | 1050 | 545 [?] | 150 | 600 | 120 | 8 | 320 | 224 | .9614 | 530 | 572 |
  | 2 | 2800 | 970 [?] | 610 | 190 | 660 | 152 | 8 | 300 | 210 | .9614 | 596 | 635 |
  | 3 | 3500 | 980 | 675 | 250 | 725 | 200 | 10 | 470 [?] | 329 | .9373 | 654 | 720 |
  | 4 | 4110 | 1000 | 720 | 350 | 770 | 280 | 10 | 460 | 322 | .9373 | 701 | 768 |
  | 5 | 4700 | 1040 [?] | 780 | 420 | 830 | 336 | 12 | 630 [?] | 441 | .9053 | 756 | 862 |
  | 6 | 5150 | 1020 [?] | 830 | 480 | 880 | 384 | 16 | 750 | 525 | .8263 | 790 | 1004 |
  | 7 | 5510 | 1030 | 870 | 600 | 920 | 480 | 16 | 720 | 504 | .8263 | 836 | 1052 |
  | 8 | 5775 | 1035 | 780 | 1000 | 860 | 800 | 16 | 850 | – | .8263 | 727 | 944 |

  Q_req = GLR x 800/1000 exactly. The "GLR at transfer" is presumably the injection GLR; the formation GLR is 150, so a
  total-GLR reading would give zero injection at valve 1 **[? reading]**.

#### 2.6.6 Analytical spacing, 25-psi drop in surface **opening** pressure (3.42972, B p.236-237): **exact regression test**
Spacing on the average operating flowing gradient Gu requires a casing drop between valves.
```
D1 = (Pko - Pwh)/Gs                                   (3.421)
D(n+1) = Dn + (Pso_n - Pwh - Gu Dn)/Gs                (3.422-3.426)   gas column neglected
```
Pso_n is the surface opening pressure of the valve *above*: "we can use the opening pressure of Valve 1 to space
Valve 2 ... common field practice is to use Pso2".

Data: Pko 1000, drops of 25 psi (1000, 975, 950, 925, 900, ...), Pwh 120, Gs 0.40, Gu 0.14 (the average flowing
gradient from Pwh to the injection point; the 3F.2 chart gives 0.13).

Result: **2200, 3630, 4497, 4998, 5261, 5370 ft** (I recomputed these exactly). The steps are 1430, 867, 501, 263,
108, then 8 ft. When a spacing falls below 200 ft, go to 200-ft spacing until one valve is below the injection point:
2200, 3630, 4497, 4998, 5261, 5461, 5661, 5861, 6061. The last two are readjusted to **5800 and 5950**.

Table 3.471 (1/4-in ports on valves 1-6, 5/16 on 7-9; tubing pressures adjusted so each valve opens with Pt 100 psi
below casing):

| # | D | T | Pso | Pvo | Pt | Ptro |
|---|---|---|---|---|---|---|
| 1 | 2200 | 126 | 1000 | 1058 | 428 | 945 |
| 2 | 3630 | 137 | 975 | 1068 | 628 | 950 |
| 3 | 4497 | 143 | 950 | 1062 | 750 | 945 |
| 4 | 4998 | 147 | 925 | 1046 | 820 | 930 |
| 5 | 5261 | 149 | 900 | 1024 | 857 | 910 |
| 6 | 5461 | 151 | 875 | 1000 | 884 | 885 |
| 7 | 5661 | 152 | 850 | 977 | 877 | 895 |
| 8 | 5800 | 153 | 825 | 954 | 851 | 870 |
| 9 | 5950 | 154 | 800 | 925 | 825 | 845 |

"If the kick-off and operating pressures are the same, over 100 psi would be lost."

#### 2.6.7 Graphical spacing with a 25-psi drop (3.42973, B p.238)
Valve 1 from the kill line to the 1000-psi kick-off line: 2350 ft. Then go horizontally to the **operating flowing
gradient** and down at 0.40 to the next (P - 25) line. Valves at **2350, 3900, 4950, 5500, 5800**, plus **6000 ft**
below. Ports 5/16 on the bottom three, 1/4 above. Table 3.472 gives Ptro 945, 950, 945, 935, 950, 925.

#### 2.6.8 Discussion on spacing (3.4298, B p.238-241)
"The greatest fallacy is failure to space valves close enough near the point of injection." For PI > 5, and especially
> 10, uncovering a valve 100 ft lower is worth 150-300 B/D. Run 2-3 extra valves offshore.

Brown's example: 8000 ft, Pr 2900, 700 B/D (95 % water), 2-3/8 tubing, Pwh 100, J = 7, BHT 210, surface flowing 150,
Pso 900, Pko 950, kill 0.50, first valve unloaded against zero back pressure. Injection at **3820 ft**; valves at
1880, 2500, 3000, 3400, 3700, 3820, 4000 ft; T 164-180 F.
* **T3.473**, 1/4-in ports, 10-psi surface-opening drop from 850 (design Pt 540, 615, 680, 725, 765, 775, 800;
  operating Pt 380, 490, 595, 680, 750, 778, 800). Using the design rather than the operating Pt gives *actual* opening
  drops of 16, 13, 11, 18, 12 psi. When operating from valve 6 (795 to open), all upper valves are closed (Psc of
  valve 5 is 802.8). Operating from valve 4 (820), valves 2-3 (close 819.6, 815.5) could stay open unless they are
  variable-orifice valves.
* **T3.474**, 3/8-in ports (R 0.1434), 10-psi drop: the larger port means lower closing pressures. Valves 1-3 would
  probably never close when operating from valve 4. **A 3/8-in intermittent valve gives trouble in continuous flow;
  a true continuous (throttling) valve does not have to fall to dome pressure to close.**
* **T3.475** (1/4, constant surface opening 800) and **T3.476** (1/4, constant surface close 800): the closes all sit
  below the bottom valve's opening, so the design works only with variable-orifice valves. A 10-psi drop "should
  [still] be taken".

#### 2.6.9 Differential valves (3.4299) and balanced valves (3.42910) (B p.241-244)
* Differential valves, graphical: 800 and 650 (800 - Sd 150) lines; valves at **1880, 2190, 2500, 2800, 3100, 3400,
  3700, 4000 ft**. Analytical: `D1 = Pko/Gs = 900/0.5 = 1800`, then `D(n+1) = Dn + Sd/Gs`, which gives 1800, 2100, ...,
  3900 (3.427-3.428).
* Balanced (Davis): 25-psi drop per valve, natural-gas charge at 80 F.
  * Data: 8000 ft, 500 BFPD (90 % water), 2-3/8 x 5-1/2, 0.65 gas, Psep 50, Pwh 100, kill 0.5, Pr 2600, PI 5,
    Pko 950, Pso 900, 300 Mscf/d available, GOR 500, GLR 50.
  * Valves at 1650, 2800, 3550 **[?: T3.477 prints 4550]**, 4000, 4250 ft. Surface opening 900, 875, 850, 825, 800;
    T 136-162 F; test-rack set at 80 F: 820, 800, 780, 760, 740.

#### 2.6.10 Proportional response system (Merla; 3.42911, B p.244-253): explicit per-valve gas requirement
* Throttling valve characteristic (Fig. 3.487-3.488), the straight-line part:
  ```
  Q_inj = M (Pt - Ptc)                    (3.429)   M = slope (Mscf/d per psi), Ptc = tubing pressure at which it closes
  ```
  Valve data (Fig. 3.492; 1-in? valve, Ab 0.31 spring-charged): trims S / M / L have port 0.25 / 0.25 / 0.40 in,
  `Fs` = 0.22 / 0.57 / 0.72, M = 1.0 / 2.6 / 3.4. The adjustment range is 600-1500 psi. The peak-flow chart gives
  `Qm/Pvo` as a function of `Pvo/Pt` **[chart, not tabulated]**.
* **Transfer** (3.429112): injection starts at the upper valve with Pt just below Pc (valve wide open). Pt falls and the
  valve throttles. When Pt reaches the **transfer pressure Ptp**, the next valve is uncovered. Both inject, the larger
  share through the lower valve, and the upper closes at its Ptc (set on the design gradient for the operating valve).
  The bottom valve is not set to close; it regulates.
* **Transfer point selection**: raising Ptp raises valve throughput and lowers the gas required (lower rate, higher
  Pwf). Ptp must be high enough that `M (Ptp - Ptc) >= Q_required`. A higher Ptp also means closer spacing. Start
  50-100 psi above Pt_min (the design traverse at the valve).
* **Gas requirement at the transfer point** (3.4291122):
  ```
  Pwf_t = Ptp + (Dw - Dv) G_below          (load-fluid gradient while unloading; static-fluid gradient after feed-in)
  q_t   = J (Pr - Pwf_t)  (or IPR);  if Pwf_t >= Pr: no inflow, use an assumed unloading rate (200 B/D) and Rs = 0
  R_T   = GLR whose traverse joins Pwh at 0 to Ptp at Dv, at rate q_t
  Q_inj = q_t (R_T - Rs)                    (3.430)
  Q_corr = Q_inj / TCF,  TCF = sqrt(520/(Tv+460))   (to 60 F test basis)
  ```
* **Valve adjustment**: `Pvc_adj = Pvo - Fs (Pvo - Pt_min)` (3.431), with Pvo the gas pressure at the valve. Check
  `Qm = (Qm/Pvo) Pvo >= Q_corr`. If M (Ptp - Pt_min) < Q, raise Ptp or move to the next trim.
* **Operating (bottom) valve**: the "transfer pressure" is the expected flowing Pt (930). The valve passes the design
  gas: `Pt_close = Pt - Q/M = 930 - 342 = 588`, Pvo/Pt = 1155/588 = 1.96, Qm/Pvo = 0.5, Qm = 577 >= 342 OK,
  Pvc_adj = 1155 - 0.22(1155-588) = **1030**.
* **Design example (B p.247-252)**. Data: 600 B/D (100 % oil), 7200 ft, perforations 6900-7100, 5-1/2 casing,
  2-3/8 tubing, Pr 2000 at 7000 ft, static level 2550 ft, geothermal 0.016 F/ft, Rs 100, static gradient 0.37,
  PI 0.8, Pso 1000, 0.65 gas, **300 Mscf/d available**, Pwh 150, load fluid 0.45.
  * TGLR = 300/0.6 + 100 = **600**; Pwf = 2000 - 600/0.8 = **1250**; injection needed at **about 6100 ft**.
  * Top valve at the U-tube depth from Pwh with 0.45 psi/ft: 2000 ft.

  Table 3.478 (I checked the arithmetic column by column):

  | Valve | 1 | 2 | 3 | 4 (S trim) | 4 (M trim, final) | 5 | 6 (operating) |
  |---|---|---|---|---|---|---|---|
  | D ft | 2000 | 3450 | 4550 | 5350 | 5350 | 5850 | 6200 |
  | Pvo (gas at valve) | 1050 | 1085 | 1115 | 1135 | 1135 | 1145 | 1155 |
  | Pt_min (design traverse) | 355 | 525 | 680 | 805 | 805 | 875 | 588 (computed) |
  | Ptp | 430 | 625 | 780 | 930 | 930 | 1000 | 930 |
  | Pwf_t | 2680 | 2225 | 1685 | 1540 | 1540 | 1425 | 1225 |
  | q_t B/D | 200 assumed | 200 assumed | 252 | 368 | 368 | 575 [?] | 620 |
  | R_T | 300 | 375 | 450 | 475 | 475 | 550 | 600 |
  | Rs | 0 | 0 | 100 | 100 | 100 | 100 | 100 |
  | R_inj | 300 | 375 | 350 | 375 | 375 | 450 | 500 |
  | Q_inj Mscf/d | 60 | 75 | 88 | 138 | 138 | 259 | 310 |
  | Tv F | 106 | 130 | 148 | 163 | 161 [?] | 169 | 174 |
  | TCF | .958 | .939 | .925 | .915 | .915 | .910 | .906 |
  | Q_corr | 63 | 80 | 95 | 151 | 151 | 285 | 342 |
  | Ptp - Pt_min | 75 | 100 | 100 | 125 | 125 | 125 | 342 |
  | M | 1.0 | 1.0 | 1.0 | 1.0 | 2.6 | 2.6 | 1.0 |
  | Q supplied | 75 | 100 | 100 | 125 (< 151, fails) | 325 | 325 | 342 |
  | Pvo/Pt | 2.96 | 2.07 | 1.64 | – | 1.41 | 1.31 | 1.96 |
  | Qm/Pvo | 0.66 | 0.52 | 0.48 | – | 0.59 | 0.48 | 0.50 |
  | Qm | 695 | 565 | 535 | – | 670 | 550 | 577 |
  | Fs | .22 | .22 | .22 | – | .57 | .57 | .22 |
  | Pvc_adj | 897 | 962 | 1019 | not enough | 947 | 990 | 1030 |

  How the columns relate:
  * Valves 1-2: `Pwf_t = Ptp + (7000 - D) 0.45`.
  * From valve 3 on, the book switches to the **0.37 static gradient**: 780 + 2450(0.37) = 1686; 930 + 1650(0.37) =
    1541; 1000 + 1150(0.37) = 1425.5; 930 + 800(0.37) = 1226. Then q = 0.8(2000 - Pwf), except valve 5:
    0.8(2000 - 1425) = 460, while the table uses 575 (= 259/0.45) **[book inconsistency]**.
  * Spacing: 0.45 psi/ft from Ptp of the valve above down to the gas line. Ptp is chosen as Pt_min + 75 / +100 /
    +100 / +125 / +125.
* **Computer version (B p.251-253)**: the same well recomputed with Ros at 950 B/D, PI 5, Pr 1935, 8000-ft reference,
  TGLR 1150, gas 902.5 Mscf/d, Pso 915 psia. Stations at 2351, 3446, 4141, 4602, 5105, 5503, 5728, 5870 ft; valve
  adjustments 869 ... 850. **Every unloading station has QTT = 275** and QR (gas required) of 54.6, 76.1, 89.2, 115.0,
  128.2, 137.1, 137.3 Mscf/d; operating 902.5 Mscf/d. PRF / PRT / PRCF columns are the flowing Pt, transfer Pt and
  casing pressure per station.

### 2.7 Directional wells (3.4210, Brown and Beggs, B p.253-257)
* Treating the well as vertical is common but unsafe. Older safety factors: `Pwf' = Pwf + 0.10 Pwf` for 10-30 degrees,
  `+0.20 Pwf` for 30-70 degrees.
* The gradient is `dp/dz = (g/gc) rho_m sin(theta) + f rho_m v²/(2 gc d) + rho_m v dv/(gc dz)` with the elevation term
  over vertical distance and the friction term over measured length. Only Beggs and Brill varies holdup with angle.
* Procedure: compute the traverse along MD, **project the pressures to TVD**, then design and space as for a vertical
  well. A deviated well needs more gas, or more pressure, for the same rate; this matters for port sizing.
* T3.479 compares Pwf at TVD 15,000 ft for 0-75 degrees, 2-in, 300-1500 B/D, GOR 300/1000 with four correlations.
* T3.480: angle 0/45/75 gives injection at 5350/4600/3200 ft and Pwf 1650-1850 **[?]** / 1940 / 2370.
* T3.481 (1000 B/D, Pr 2400): vertical injects at 4500 ft, TGLR 520, injection gas **220 Mscf/d**; 45 degrees at
  4680 ft, TGLR 1000, **700 Mscf/d**.
* T3.482 (8000 ft TVD, 2-7/8, Pwh 120, Pso 900, Rs 300, injection GLR 700, B&B), rates in B/D:

  | Angle | PI 1 | PI 2 | PI 4 | PI 8 | tubing length |
  |---|---|---|---|---|---|
  | 0 | 1095 | 1500 | 1720 | 1830 | 8000 ft |
  | 45 | 970 | 1320 | 1530 | 1640 | 11914 ft |
  | 75 | 720 | 1020 | 1200 | 1280 | 30910 ft |

### 2.8 Summary and logical sequence (3.4211, B p.257)
1. Build intake (rate vs operating dP) curves for several dP and PIs (Fig. 3.476-B).
2. Pick dP from rate, spacing, PI and cost. Use the lowest practical dP on high-PI wells. dP < 50-75 psi means near
   minimum spacing (dP = 50 at Gs 0.40 gives 125 ft). Never closer than 3 joints. Without good data, **100 psi is a
   good choice**.
3. Design graphically with Psep, Pwh, Pso and so on.
4. Draw the objective gradient for the rate and gas, and use dP to locate the injection point.
5. Construct the bracketing envelope.
6. Valve 1: kill-fluid line from the separator pressure to the kick-off line, **moved back 40 psi**.
7. A transfer point for each succeeding valve controls the spacing: on the objective gradient for pressure valves, per
   the fluid-valve section for fluid valves, and always to the right of the objective line.
8. Space to the top of the envelope, then at `dP/Gs` down to its bottom.

---------------------------------------------------------------------------------------------------------------------

## 3. Unloading process and troubleshooting, continuous flow (Brown 3.6, B p.355-444)

### 3.1 Continuous unloading sequence (3.6211, B p.356-357; Fig. 3.61): a simple regression test
Inject slowly through a choke. Kill fluid U-tubes up the tubing as casing pressure rises. Valve 1 is uncovered when
fluid velocity jumps, and the casing levels at valve 1's operating pressure. Each further transfer lowers the casing to
the next valve's setting. Feed-in starts partway and slows the process. If continuous unloading fails, unload
intermittently.

Worked spacing: surface operating pressures 625/600/575/550, kill 0.50 psi/ft, Pwh 50, the lifted gradient above an
injecting valve 0.10 psi/ft, gas column neglected.
* V1 at 625/0.5 = **1250 ft**.
* Pt(V1) = 50 + 125 = 175, so (625-175)/0.5 = 900 and **V2 = 2150 ft**.
* Pt(V2) = 265, so (600-265)/0.5 = 670 and **V3 = 2820 ft**.
* **V4 = 3306 ft** (recomputed: Pt = 332, (575-332)/0.5 = 486).
* An intermittent unloading gradient of 0.04 instead gives 1250, 2300, 3216, **4008 ft** (B p.358).
* The intermittent unloading cycle time is about `1.5 x depth/1000` minutes.

### 3.2 Injection control for continuous flow (3.622, B p.358-360)
* **Surface choke** is the simplest and best control for continuous flow. Size it by trial; adjustable chokes freeze.
* A **regulator in series** with the control line on the tubing shuts gas off at high Pt, which suits wells that will
  flow naturally.
* A **regulator in parallel** (or upstream) of the choke holds a constant casing pressure against fluctuating line
  pressure. Fig. 3.618I(2): the heading disappeared.
* A time-cycle controller is not recommended for continuous flow except against freezing.

### 3.3 Diagnostic checklist and methods (3.631, B p.360-361)
23 surface questions (wing valve, chokes, freezing, Pinj, Pwh, Psep, rates, two-pen chart, installation record, ...).
Methods:
* two- and three-pen pressure charts;
* gas measurement: total output gas minus injection gas should equal formation gas, otherwise retest and check meters;
* temperature at the tree: a cold, sweating tree means a tree leak, or a plug below the first valve;
* gauging;
* **flowing pressure and temperature surveys**;
* acoustic fluid levels.

### 3.4 Flowing pressure surveys (3.632, B p.361-367)
Run while the well performs well, from a lubricator without interrupting flow. Stop just below each valve, and several
times below the last valve to get the gradient below injection.

**The interpretation rules are implementable as diagnostics against a model traverse:**
* The operating valve is where the gradient breaks (heavy below, light above).
* The **balance point** (casing pressure at depth equal to the tubing traverse) below the operating valve shows whether
  the next valve can be reached: Fig. 3.611C has its balance at 4300 ft with valve 3 at 4500, so spacing is too wide.
* Casing pressure is sufficient to open a deeper valve but the gradient does not break there (Fig. 3.611D, 3.614I):
  the valve is plugged, has gained pressure, or failed closed.
* Breaks at two valves mean multipoint injection or interference (Fig. 3.611E, 3.612B). The input GLR is high
  (800 vs 400).
* A break between valves means a tubing leak (Fig. 3.612E-G: leak at 2070 ft; average gradient 0.275 above it).
* High Pwh limits the casing-tubing dP (400 psi in Fig. 3.611F) and keeps injection shallow.
* Effect of Pinj: PI 0.75 goes from 600 to 900 B/D for Pso 810 to 1520 (Fig. 3.611G). In Fig. 3.612A (PI >= 10) the
  annulus level was a few feet short of the bottom valve. Respaced with the last valves 3 and 2 joints apart, the
  well went from **1000 to 1800 B/D**. Fig. 3.612D: raising casing and lowering tubing pressure uncovered valve 5
  (372 to 650 B/D).
* Temperature surveys (3.633-3.634) find the operating valve from Joule-Thomson cooling (hard above about 6000 B/D in
  4-in tubing). Take a stop 30-100 ft below the surface, because the lubricator reading is wrong by 13-27 F. Temperature
  often shows small gas entries (upper valves) that pressure misses, and vice versa. A model needs a JT cooling term at
  injection to reproduce this.

### 3.5 Two-pen surface charts, continuous flow (3.6372, B p.378-383)
| Case (Fig. 3.618) | Before | After |
|---|---|---|
| A: upper valve lost pressure | Pso 1300, 1800 B/D, injection GLR 600, Pwh 300 | Pso 1000, 1200-1300 B/D, GLR 1000 |
| B: lower Pwh | Pwh 400, 600 B/D, GLR 1000 | Pwh 150, 1200 B/D, GLR 500 |
| C: bad valve, then repaired | Pso 675, Pwh 350, 200 B/D, GLR 3000 | Pso 1000, Pwh 200, 1000 B/D, GLR 400 |
| D: raise system pressure | Pso 600, 400 B/D | Pso 1100, 1500 B/D |
| E: sand bridge | Pso 1400, q = 0 (low Pt on a large seat raises Pc) | Pso 1100, 1200 B/D |
| F: tubing hole near surface | Pso 450, 25 B/D, GLR 50,000 | Pso 1200, 3000 B/D, GLR 500 |
| G: choked intermittent valve intermits | 2000 B/D | continuous valve: 2500 B/D |
| H: fluid valve with too large a port (5/16) heads | 1200 B/D | 1/4 port: 1300 B/D, smoother |

Fig. 3.618J/K: **port too large gives heading**; the valve tries to close when casing is near its closing pressure, or
the gas volume is too low. Fig. 3.618M: an intermitting neighbour drawing down a common line.

Other chart signs: a valve-alternation pattern (Pc oscillating between two valves' levels); loss of dome charge (Pc
far below any valve's setting); freezing (no injection, Pt falls to Psep).

### 3.6 Surface back pressure and injection pressure (3.638-3.6315, B p.435-444)
* Continuous flow: Pwh directly sets Pwf. 100 psi more Pwh can cost hundreds of B/D on high-PI wells.
* Beadle: choke 1 to 3/8 in raised Pwh 175 to 225 and cut the rate 1745 to 633 B/D. A 4-in line gave 2375 B/D.
* Table 3.63 (required total GLR vs Pwh at 600 STB/d, 2-in, 9000 ft, Pr 2800, PI 10, WOR 49, formation GLR 16):
  Pwh 140 / 200 / 235 / 270 / 284 / 284 needs TGLR 600 / 800 / 1000 / 1500 / 2000 / 3000.
* Table 3.64 (Pr 2600, 4-in): Pwh 200 to 100 raises the rate 9400 to 10600 B/D. Fig. 3.622H: 140 to 100 psi tubing
  pressure, PI 5, Pr 2700: 4550 to 4900 BOPD.
* Injection pressure (3.639): PI 6 well goes from 4700 B/D at 1000 psi to 5800 at 1400 psi.
* Field case 3.6315: 2 to 3-in tubing, 8 to 14 valves, 800 to 1200 psi, 2 to 4-in flowline, streamlined tree.
* Leak test: pressure tubing and casing equal, bleed the casing. If Pt holds, the valve is bad; if Pt falls, the tubing
  has a hole. "Rocking" (alternate bleeding) can unplug a valve.

---------------------------------------------------------------------------------------------------------------------

## 4. API RP 11V2: gas-lift valve performance testing and flow models

### 4.1 Symbols (Sec. 2, RP p.1-2)
* Ab: bellows area. Ap: port area (nominal diameter). As: area where the stem contacts the seat.
* B_lr: bellows-assembly load rate (psi/in). Cv: flow coefficient.
* `Fk = k/1.40`. k: specific-heat ratio (natural gas 1.3, air 1.4).
* x: `dP/(P1+14.7)`. Xt: pressure-drop-ratio factor; critical when `Fk Xt <= x`. Y: expansion factor. Z1: upstream z.
* Pvo/PvoT: test-rack opening with P2 = 0, at 60 F or at T. Pvc/PvcT: closing with P2 = P1 and near-zero flow.
* Piod: injection pressure at the valve. Ppd: production pressure at the valve.
* LST: load-rate (maximum effective) stem travel. GST: geometric stem travel to full equivalent area.
  VST: valve stem travel.
* Standard conditions 14.73 psia and 60 F.

### 4.2 Probe test and load rate (Sec. 4, RP p.10-13; App. C, RP p.39-41)
* Test N2 and combination valves at Pvo 800, 1200 and the maximum recommended pressure; spring valves at the maximum.
* Raise pressure on the bellows in 10-25 psi steps, reading stem position by micrometer probe with ohmmeter contact.
  Record at least five points within the effective travel. Then come down; never reverse a step.
* Pressure vs travel shows **slope A**, the usable travel, and a steep **slope B** where the bellows stacks.
  **`B_lr` = slope of the best-fit line in region A** (psi/in). **Maximum effective travel (LST)** = the travel where
  A meets B.
* App. C algorithm, for at least 4 points rising monotonically:
  1. Sort by travel. Start line B from the last two points.
  2. Fit line A by regression to points 1..n-2.
  3. For point (n-2) compute its perpendicular distance to lines A and B
     (`x = (y_k - b_i)/m_i - x_k`, `y = m_i x_k + b_i - y_k`, `theta = atan(y/x)`, `d = |x| sin(theta)`).
  4. If it is closer to B, assign it to B. Re-fit.
  5. Repeat for n-3, ... until a point belongs to A.
  6. `LST = (bB - bA)/(mA - mB)`; `B_lr = mA`.
  * Hysteresis: analyse rising and falling sets separately and average.
* **Example (Table C.4.1)**. Travel (in) / pressure (psig): 0/450, .019/460, .032/470, .055/480, .070/490, .079/500,
  .102/550, .116/600, .126/650, .134/700, .147/800. First six points form line A, last five line B.
  The book gives **Load rate = 606.71 psi/in, LST = 0.098 in**. I reproduce 605.7 psi/in and 0.0976 in; use a 0.5 %
  tolerance.

### 4.3 Flow-coefficient test (Sec. 5, RP p.13-18)
* Hold the stem at a fixed travel (modified valve). Test at least 5 travels: one <= 10 % of LST, one at 100 %, and
  three in between. At each travel use at least 5 pressure ratios up to critical, with 3 inside 10-90 % of critical.
* Data reduction (5.4.2):
  ```
  Y*Cv = q[scf/hr] * sqrt( Sg (T1+460) Z1 / x ) / ( 1360 (P1 + 14.7) )
  ```
* Fit Y*Cv against x with a straight line, `YCv = A x + B` (slope M < 0).
  * **Cv = intercept at x = 0.**
  * **Xt = x where Y*Cv = 0.667 Cv**, that is `Xt = (0.667 Cv - Cv)/M`.
  * Re-test any point more than 5 % off the line. Low-x, small-travel points are suspect.
* Expansion factor (5.4.6):
  ```
  Y = 1 - x/(3 Fk Xt),   0.667 <= Y <= 1 ;   if x > Fk Xt use x = Fk Xt
  ```
* Record Cv(travel) (Fig. 12, ports 3/16-3/8) and Xt(travel) (Fig. 13). Maximum error is about 9 % on Cv and 11 % on Xt.
* **Rate equation (5.5)**:
  ```
  q_gi [Mscf/d] = 32.64 * Cv * (Piod + 14.7) * Y * sqrt( x / (Sg (Tv+460) Z1) ),
  x = (Piod - Ppd)/(Piod + 14.7)  (capped at Fk Xt)
  ```
  32.64 = 1360 x 24/1000.
* **Example 5.5.1 (RP p.18)**: Piod 1000, Ppd 850 psig, Sg 0.65, 150 F, stem travel 0.020 in, test gas k 1.4.
  x = **0.1478**; Xt(0.02) = **0.45**; Fk = 1.3/1.4 = **0.928**; Fk Xt = **0.417** (not critical);
  Y = 1 - 0.1478/(3 x 0.928 x 0.45) = **0.882**; Cv(0.02) = **0.40**; Z1 = 0.95; **q = 231 Mscf/d**
  (recomputed 231.5).

### 4.4 Dynamic performance tests (Sec. 6, RP p.20-26): what a model would be fitted to
* **CPPT**: hold P2 and raise P1 above Pvo.
  * Max dP above Pvo: `1.2 (probe dP)/(1 - As/Ab)` if VST = LST, or `1.2 VST B_lr` if VST = GST.
  * At least 4 dP steps (25/50/75/100 %).
  * P2 = 20/40/60/80 % of PvcT, where `PvcT = PvoT (1 - As/Ab)`.
* **CIPT**: hold P1 and let P2 fall from P1 to 0.9 P1 and then down (at least 6 rates).
  * P1 levels inside the throttling band: `PvoT - 0.1/0.25/0.5/0.65 (PvoT - PvcT)`.
  * Orifice regime: P1 = PvoT + max dP and PvoT + 0.5 max dP, with at least two P2 below P1/2.
* Fig. 16 is the 3-D (P1, P2, q) surface going from throttling, through transition, to orifice flow. A model needs
  tests at **at least three set pressures** (minimum, maximum, intermediate; at least 200 psi apart) (7.3).
* Equivalent flow area vs travel is the frustum of a right cone (6.2.4). **GST** = travel at which it equals the port
  area.
* Ramp method (App. D): first-order system response `P(t) = Pmax e^(-t/tau)`. tau is measured as the time to fall
  63.2 % (850 to 313 psig example). Ramps should last at least 5 tau (tau = 9 s gives a 45-s ramp).

### 4.5 Appendix A: simplified (static stem-position) model (RP p.27-28)
Assumptions: P2 acts on the ball/seat contact area As; the areas are constant; static force balance with load rate.
```
P_voT = Ptro * (temperature/z correction to Tv)          (use the manufacturer's chart if available)
P_vcT = P_voT (Ab - As)/Ab
P_vcT Ab + B_lr Ab dx = Piod (Ab - As) + Ppd As        (N2- or spring-loaded IPO valve)
dx = [ Piod (Ab - As) + Ppd As - P_vcT Ab ] / (B_lr Ab)     ,  dx <= LST
Cv = Cv(dx), Xt = Xt(dx) from the curves; then the 5.5 rate equation
```
Accuracy about +/-30 % for ports up to 3/16 in (limited data); the error grows with port size.

**Example A.5**: 1-in IPO valve, 3/16 port (Ab 0.31, As 0.0276), Piod 925, Ppd 450, Ptro 825 psig, Tv 150 F, Sg 0.65.
* P_voT = 825 x (610/520) x (z-ratio) = **919 psig**. The printed formula is
  "825*0.95*(150+460)/(60+460)/0.95" **[? OCR; 919 implies an effective z60/zT factor of about 0.95]**.
* P_vcT = 919(0.31 - 0.0276)/0.31 = **837 psig**.
* B_lr = **935 psi/in** at P_vcT; LST = **0.085 in** (manufacturer).
* **dx = 0.049 in** (recomputed 0.0489).
* Curve fits for Fig. 12 and Fig. 13:
  ```
  Cv = 219.3 dx³ - 149.04 dx² + 22.58 dx    (cap at the value for dx = 0.095)  → Cv = 0.774
  Xt = 734.4 dx³ - 178.3 dx² + 15.12 dx                                        → Xt = 0.40
  ```
* x = (925-450)/939.7 = **0.505** > Xt, so the flow is critical and x is set to 0.40. Y = 1 - 0.4/(3(1.3/1.4)0.4) =
  **0.641**. Z = 0.95. **q = 496 Mscf/d** (recomputed 496).
* **[Note: the example compares x with Xt, not Fk Xt as Sec. 5.5 says. Strictly, x = Fk Xt = 0.371 and Y = 0.667,
  which gives about 497 Mscf/d. Reproduce the book number with the book's rule and flag the option.]**

### 4.6 Appendix B: TUALP variable-area / orifice model (RP p.29-37)
Two categories exist, separated by the **transition injection pressure P_tran**:
* **Orifice flow** (Piod > P_tran): critical when Ppd < r_crit Piod, subcritical above that.
* **Variable-area (throttling) flow** (PvcT < Piod < P_tran): flow is zero at the production closing pressure Ppdc,
  rises linearly ("supermaximal" side) to q_max at Ppdmax, then falls to zero at Ppd = Piod ("submaximal" side).

Accuracy: orifice +/-5 %; variable area +/-30 % (N2-charged) or +/-15 % (spring). Basis: 3967 points on 1-in valves
(Camco BK, BK-1, McMurry JR-STD, Merla NM-16R) and 2590 points on the Camco R-20 1.5-in.

**B.3 Orifice regime** (P in psia):
```
q_gi[Mscf/d] = 1241 Ap Cd Y sqrt( Piod (Piod - Ppd) / (Tv[R] Zv Sg) )                 (B.3.1, B.6.5)
Cd Y = a (Piod - Ppd)/(Piod K) + c                                                     (B.3.3, B.6.4)
Ppd@rcrit = r_crit * Piod ; if Ppd < Ppd@rcrit use Ppd@rcrit (critical)                (B.3.4-B.3.6, B.6.1-B.6.3)
```
* Fit Cd Y from at least 2 orifice curves (one PvcT, two Piod > P_tran). r_crit comes from a plot of
  `(q_max - q)/(Piod - Ppd)` against Ppd/Piod.
* **0.25-in port example: a = -0.838, c = 0.844, r_crit = 0.67** (Fig. B-5, B-6). K is the specific-heat ratio
  ("Air = 1.4") **[? whether K is test gas or lift gas]**.
* Cross-check I computed: Ap 0.0491, Piod 1014.7, Ppd 614.7 psia, 610 R, Z 0.9, Sg 0.65 give Cd Y = 0.608 and
  **q = 1249 Mscf/d**, against Thornhill-Craver (Cd 0.865, k 1.27) **1257 Mscf/d** for the same case. The constant
  1241 is therefore consistent with Mscf/d.

**B.4 Variable-area regime** (at least 4 curves: 2 PvcT x 2 Piod between PvcT and P_tran):
* a. `Ppdc` = production pressure at which the extrapolated rising (supermaximal) line meets q = 0. It is **not** the
  force-balance closing pressure. Fig. B-7: **Ppdc = 250 psig**.
* b. `q_max` for each curve. Fig. B-7: **415 Mscf/d**.
* c. Normalise:
  ```
  N = (Ppd - Ppdc)/(Piod - Ppdc)                                      (B.4.1)
  ```
  q/q_max against N collapses all curves; **N_max** is the N at q_max. Fig. B-8: **N_max = 0.55**.
* d.
  ```
  Ppdmax = N_max (Piod - Ppdc) + Ppdc                                  (B.4.2)
  ```
  Example: **Ppdmax = 470 psig**, so the B-7 curve had Piod of about 650 psig.
* e.
  ```
  slope = q_gi/(Ppd - Ppdc)  on the rising side ;   slope = m PvcT + b              (B.4.3-B.4.5)
  ```
  Example (Fig. B-9): **m = 0.00127 Mscf/d/psi², b = 1.25 Mscf/d/psi**. The slope can be either sign.
* f. Dynamic tubing-sensitivity factor (not R):
  ```
  Ft = (Piod - PvcT)/(Piod - Ppdc)                                     (B.4.6)
  ```
  Average over the curves. Example **Ft = 0.09**.

**B.5 Transition**:
```
P_tran = PvcT (1 + Ft)                         (B.5.2, "recommended")
P_tran = PvoT = PvcT/(1 - R)                   (B.5.3, alternative)
```
**[? OCR of B.5.2. Setting Ppdc = 0 in B.7.1 would give P_tran = PvcT/(1 - Ft) instead; the two are close for small Ft.
Fig. B-4 (idealised) labels PvcT = 650 and P_tran of about 830 at valve temperature, which matches neither form
exactly.]**

**B.6 Orifice procedure**: if Piod > P_tran, cap Ppd at r_crit Piod, then Cd Y (B.6.4), then q (B.6.5).

**B.7 Variable-area procedure** (Piod < P_tran):
```
Ppdc   = Piod - (Piod - PvcT)/Ft                                         (B.7.1)
Ppdmax = N_max (Piod - Ppdc) + Ppdc                                      (B.7.2)
if Ppdc < Ppd <= Ppdmax:  q = (m PvcT + b)(Ppd - Ppdc)                   (B.7.3, B.7.4)
   gas/temperature correction:  q = q_test * sqrt( T_m Z_m S_gm / (Tv Zv Sg) )   (B.7.5)  [T_m, S_gm = test temperature, gas]
if Ppdmax < Ppd < Piod (submaximal):
   CF = q_varea(Ppd = Ppdmax) / q_orifice(Ppd = Ppdmax)                  (step h)
   q  = CF * 1241 Ap CdY sqrt(Piod(Piod-Ppd)/(Tv Zv Sg)) with CdY = a(Piod-Ppd)/(Piod K) + c   (B.7.6-B.7.8)
Ppd <= Ppdc: q = 0 (valve closed)
```

A self-consistent illustration built from the printed coefficients (my construction, use as a smoke test):
* PvcT = 614 psig, Piod = 650 psig, Ft = 0.09 give Ppdc = 650 - 36/0.09 = **250**, Ppdmax = **470**.
* slope = 0.00127(614) + 1.25 = 2.03, so q(Ppd = 470) = **447 Mscf/d**. The measured curve in Fig. B-7 gives 415.

### 4.7 Other 11V2 notes
* The flow-coefficient data are geometry-specific; valve-specific curves (Cv(travel), Xt(travel), B_lr, LST, Ab, As)
  must come from the manufacturer or from tests.
* The TUALP model does **not** use probe or load-rate data; the simplified model does **not** use Section 6 data.

---------------------------------------------------------------------------------------------------------------------

## 5. Relevance to fluxwell (`fluxwell/gaslift.py`)

### 5.1 What gaslift.py does now, against Brown and 11V2
| Topic | gaslift.py | Brown / 11V2 | Verdict |
|---|---|---|---|
| Gas column | `casing_pressure` / `gas_column`: real-gas density integration (blackoil), no annulus friction | Eq. 3.220. The chart (F) values equal ideal gas at 80 F, z = 1, with a linear T correction | Better than the charts. Brown's F numbers are only loose (+/-2-3 %) checks. Annulus/tubing-injection friction is missing (B p.145) |
| Valve force balance, IPO with N2 dome | `valve_settings`: `Pbt = Pvo(1-R) + Pt R`, `Pvc = Pbt`, `Pd60 = Ct·Pbt` (Winkler-Eads above 1200), `Ptro = Pd60/(1-R)`, closing CHP from the inverse column | Eqs. 3.27, 3.215, 3.222; Ct matches Table 3D.1 at 148 and 180 F | Same. No spring (St), fluid (PPO), combination or balanced valve variants |
| Port flow | `thornhill_craver`, Cd 0.865, k 1.27, Mscf/d | Brown: same form, k about 1.31. TUALP orifice gives the same rate in my check case (1249 vs 1257) | OK for orifice/critical flow. **No throttling** (variable-area) model |
| Operating point | `lifted_traverse`: injection where Pt + dp_valve = Pc (dp_valve default 100) | 3.4231: balance point less 100 psi ("operating differential"), Fig. 3.476 | Same concept |
| Design rate | Performance curve (`performance_curve`, `optimum`), max oil or max revenue | 3.4232 pressure-rate diagram and equilibrium curve; 3.425 economic slope `X = qG/(PI P)` (Mitchell/Pacheco) | Revenue maximum is equivalent to dq_o/dq_gi = G/P on the performance curve. Fine |
| Spacing | `space_valves`: first valve on the load-fluid line from top_p_unload to the kick-off casing less margin_top (50); next valves from the transfer pressure (design traverse, or a "spacing line" `top_p + SF(p_inj - top_p)`, SF 0.20/0.25/0.30 by PI) at load gradient to the casing pressure reduced by the drops, less margin (20) | 3.42922: design line from `Pwh + max(0.20 Pso, 200)`. 3.4293: 50-psi margin at every valve, transfer on or right of the design traverse. 3.4211: 40 psi back from kick-off. Analytical 3.42972 uses Gu | Same family. Brown's line is applied to Pso, not (Pso - Pwh), and has a 200-psi floor. No **bracketing envelope**, no **dP/Gs spacing near the injection point**, no "200-ft once spacing < 200" rule |
| Casing drop per valve | `dp_close` 20 psi, or the spread R(Pvo - Pt) if larger | 10-20 psi on the surface close; 15-25 per valve; 25 psi in the analytical and graphical examples | OK |
| Unloading gas per valve | `_unloading_gas_transfer`: gas that brings Pt at the valve to the transfer pressure while lifting the **design** rate; or `_unloading_gas`: gas at minimum Pt with the IPR rate | Brown pressure-valve designs (3.42922-3.42973): **no gas calculation for unloading valves; same port on all valves, or 1/4-in unloading ports**. Fluid valves (3.4296): `Q = GLR_transfer x q_design`, smallest port with **70 % of orifice capacity >= Q**. Merla (3.42911): `Q = q_t (R_T - Rs)` with **q_t from the IPR at Pwf = Ptp + (D - Dv) G** (or an assumed unloading rate with Rs = 0 when there is no inflow), corrected by sqrt(520/T) | PROSPER-like. Brown/Merla differ: q_t comes from the IPR at the transfer Pwf, not from the design rate |
| Port of unloading valves | First standard port that passes q_req at (Pc_depth → flowing Pt), with TC de-rate | Brown: same port everywhere, or 1/4-in for spread control. Fluid valves: 70 % capacity rule. Merla: trim chosen from `M (Ptp - Pt_min) >= Q_corr` and `Qm >= Q` | Add a "same port on all valves" option and a capacity-fraction (70 %) option |
| Rate reduced for IPR conformance | `design()` keeps `qgi_d` at the maximum and recomputes q with `solve_rate` at the fixed depth, `q_new = min(op.q, q_first)` | See 5.2(b) | **Gap**: operating gas is never reduced |
| Throttling or multipoint | None. Single injection point, orifice valves | Brown 3.26, 3.27, 3.42911; 11V2 App. A/B | **Gap**, see 5.2(c) |
| Directional | MD/TVD path, gas column in TVD, inclined traverse | 3.4210: project the traverse to TVD and design as vertical | Already better |
| Unloading simulation, troubleshooting | None (design and stability only) | 3.6 | Optional diagnostics |

### 5.2 Specific questions

**(a) How Brown chooses the gas rate per unloading valve and the port size (continuous flow).**
1. **Operating valve**: gas = (TGLR - Rs) q at the design point. The TGLR comes from the traverse joining Pwh to
   (Dinj, Pc - dP). The port is the next standard size passing that gas at Pc(depth) → Pt(depth), via the TC chart with
   `CF = 0.0544 sqrt(gamma T)` (Ex 9: 400 Mscf/d gives 12/64; universal design: 800 Mscf/d gives 5/16). The 100-psi
   operating differential is set by **spacing**, and the port only has to pass the gas at that differential (B p.226).
2. **Unloading valves, pressure-operated (all Brown pressure-valve examples)**: **no unloading gas is computed**.
   Practice is the **same port as the operating valve on every valve**, or **1/4-in on the unloading valves** (smaller
   R, smaller spread, more reliable closing) with 9/32-5/16 on the bottom 2-3. For the first transfer there is no
   feed-in, so the "port can be based on other considerations than gas passage" (B p.231). Spread and closing order are
   what limit port size: 3/8-in ports can keep upper valves open (B p.239).
3. **Unloading valves, fluid-operated (B p.233-236, Table 3.470)**: per valve, `Q_req = GLR at transfer x q_design`
   (120 → 480 Mscf/d down the string, operating 800). Port = **smallest port whose 70 % orifice capacity is at least
   Q_req**: 8, 8, 10, 10, 12, 16, 16, 16/64.
4. **Proportional response (Merla, Table 3.478)**: per valve, `q_t` comes from the IPR at `Pwf = Ptp + (D - Dv) G`
   (load gradient, later the static gradient), or an assumed unloading rate (200 B/D, Rs = 0) when Pwf >= Pr.
   `R_T` is the GLR joining Pwh to Ptp at Dv, and `Q = q_t (R_T - Rs) / sqrt(520/Tv)`. The trim (M) and Ptp are chosen
   so that `M (Ptp - Pt_min) >= Q`, checked against `Qm`. The **gas grows down the string** (63 → 342 Mscf/d) because
   inflow starts. In the computer version every unloading station has the same capacity (275 Mscf/d) and the
   requirements are 55 → 137 Mscf/d.
   * gaslift.py's `_unloading_gas_transfer` sits between methods 3 and 4. It uses the **design** rate rather than the
     IPR rate at the transfer Pwf, and it sizes the port against the flowing Pt, not the transfer Pt.

**(b) A rule that brings the operating gas below the maximum when the design rate is reduced for IPR conformance
(PROSPER: 3.28 MMscf/d of 4 available; oil 1486 → 1377 STB/d).**
Brown's continuous design never treats "maximum gas available" as the operating gas. The operating gas is always the
**output** of the operating-point construction (3.4231, steps 10-14; 3.4232 limited gas). Given the final injection
depth Dinj (fixed by the valves) and the **operating differential dP** (100 psi):
1. `Pt_inj = Pc(Dinj) - dP`.
2. Traverse **below** injection at the formation GLR from (Dinj, Pt_inj) to the perforations gives Pwf.
3. `q = IPR(Pwf)`.
4. Find the TGLR whose traverse joins Pwh to (Dinj, Pt_inj) at rate q.
5. `Qgi = (TGLR - Rs) q`, capped at the gas available. Only if the cap binds does the rate come from the limited-gas
   construction (`GLRinj = Qavail/q`) instead.

When the injection point ends up shallower than the depth of the unconstrained design point (the valves cannot unload
deeper), steps 1-3 give a lower q. Step 4 then usually needs **less** than the maximum gas, because the traverse only
has to reach a higher Pt_inj, which is the PROSPER pattern (q down 7 %, gas down 18 %).

In gaslift.py, `design()` keeps `qgi_d = qgi_max` and only recomputes q with `solve_rate(..., [qgi_d],
fixed_md=[inj_md])`. That solution has Pt_inj **below** Pc - dp_valve, so it consumes more casing-tubing differential
than designed, and the reported gas stays at the maximum.

**Suggested rule:** after the IPR check, solve the two-equation system for (q, qgi) at the fixed depth:
* `VLP_bottom(q, qgi, Dinj) = IPR(q)`;
* `Pt(Dinj; q, qgi) = Pc(Dinj) - dp_valve`, with `qgi <= qgi_max`.

Equivalently, run steps 1-5 above. This is a hypothesis about PROSPER's rule; test it against the manual's 3.28 / 1377.
Report "gas required" against "gas available" as Brown's tables do.

**(c) What is needed to model valve throttling and multipoint injection.**
1. **Per-valve flow law q_i(Pc_i, Pt_i, T_i)** with a closed state. In order of data demand:
   * Merla linear: `q = M (Pt - Ptc)` capped by `Qm`, `Ptc = Pvo - (Pvo - Pvc_adj)/Fs`
     **[derived from 3.431 rearranged; check]**. Simple and calibratable to vendor curves.
   * TUALP (11V2 App. B): needs a, c, r_crit (orifice) and N_max, m, b, Ft (variable area) per valve and port, plus
     PvcT from the dome setting with Ct or Winkler-Eads. It gives the full throttling, transition and orifice surface.
   * 11V2 App. A: needs Ab, As, B_lr (from the probe test or App. C fit), LST, and the Cv(dx), Xt(dx) curves.
     `dx` comes from the static force balance, then the Cv/Xt rate equation with Fk and Y.
   * Fallback: the existing TC orifice when Pc >= Pvo (open), q = 0 when Pc < Pvc, plus a hysteresis flag.
2. **Valve state logic** from the force balance: IPO opens if `Pc >= (Pd - Pt R)/(1-R)` and stays open until
   `Pc < Pd` (Brown's single-element valve), or throttles per the law above. PPO and fluid valves use Eqs. 3.235/3.242.
   Valve-specific Ab, Ap/As and R come from the `ValveType` table (add As, B_lr, LST, Cv/Xt or TUALP coefficients).
3. **Coupled solver**:
   * Unknowns: the casing surface pressure Pso (or the surface choke flow), q_liquid, and the tubing traverse with gas
     added at every open valve.
   * Equations: the tubing traverse top-down with a GLR step at each injecting valve; `Σ q_i = Q_supply` (choke-
     controlled), or Pso fixed (regulator); IPR at the bottom.
   * Pc(D) comes from the static column (plus friction if needed).
   * Iterate Pso/q. Expect multiple solutions and hysteresis (open/closed history), so solve from the unloading history
     (top valve first) or report all branches.
4. **Validation data**:
   * Brown's survey patterns (two breaks = multipoint, Fig. 3.611E, 3.612B), with the input GLR roughly doubling.
   * The Merla example (unloading-station capacities 275 Mscf/d against requirements 55-137).
   * 11V2 Figs. 14-16 for the shape.
   * Stability: pair with the Asheim/Alhanati criteria already in `papers_digest.md`. A throttling valve changes
     `dq/dPt` (a positive slope instead of the orifice's negative one), which is the stabilising effect Brown describes.

### 5.3 Ranked suggestions (what to add or change, which equations, which tests)
1. **Operating gas after the IPR reduction (5.2(b))**: replace "qgi stays at the maximum" with the Pt_inj = Pc - dP
   construction at the final depth. Report `gas required` and `gas available`.
   *Test*: PROSPER manual case (target about 3.28 MMscf/d and 1377 STB/d).
   *Brown sanity tests*: Example #9 (balance 4800, injection 4625, TGLR 600, 400 Mscf/d, 12/64 port), and the T3.481
   vertical case (inj 4500, TGLR 520, 220 Mscf/d). Both are chart-based, so use loose tolerance.
2. **Valve-mechanics unit tests and missing valve types** (pure algebra, exact):
   * Brown Ex 1-3 (722 / 56 / 778 / 22).
   * Ex 5-6 (845 / 811 / 716 / 682 / 758, using Brown's F and Ct).
   * Summary 3.2516 (955 / 879 / 76 / 737 / 63 / 70 / 940).
   * Spring/dome examples (St 1181; Pd 433, Pd60 344, Ptro 1062).
   * Fluid valve Ex 8 (1224 / 973 / 1614) and the 3.274 summaries (793 / 834; 795 / 663; 512 / 793 / 407 / 724).
   * Universal design Table 3.464 rows (Pvc 917, 949, ..., Ptro 886 ...) with R = 0.0996, from Pvo and Pt design.
   Extend `valve_settings` with `St` (Eqs. 3.226/3.230) and PPO/fluid valves (3.235/3.242/3.244).
3. **Analytical spacing as an exact test of the spacing kernel**: implement 3.421-3.426 (gas column neglected, Gu
   design line, drop per valve, Pso of the valve above). *Test*: **2200, 3630, 4497, 4998, 5261, 5370** ft (then
   200-ft steps). Add the **3.6211 unloading example: 1250, 2150, 2820, 3306 ft**, and differential valves
   (1800 + 300 n ft).
4. **Spacing options from Brown**:
   * (i) Universal design line `Pwh + max(0.20 Pso, 200)` to Pt_inj, alongside PROSPER's `top_p + SF(Pinj - top_p)`.
   * (ii) A **bracketing envelope** (Pwh +/-10-20 %, Pt_inj +/-10-20 %, crossed with the Pc - dP line) with valves at
     `dP/Gs` inside it, plus "spacing < 200 ft → 200 ft" and "at least one valve below the injection point".
   * (iii) The 3.4293 closing-pressure drop design with the transfer point on or right of the design traverse and a
     50-psi margin at every valve.
   *Test*: the universal design depths 2350/3350/4150/4750/5250/5650/5900 ft, envelope 4450-7850 ft (needs the
   example's traverse, so tolerance about 100 ft).
5. **Unloading-gas and port options (5.2(a))**:
   * (i) "Same port on all valves" and "1/4-in unloading ports, operating port on the bottom N" (Brown practice).
   * (ii) A capacity-fraction rule (70 %) for unloading ports.
   * (iii) Merla/Brown transfer gas: `q_t = IPR(Ptp + (D - Dv) G)`, or an unloading rate with Rs = 0 if no inflow;
     `R_T` from Pwh to Ptp at Dv; `Q = q_t (R_T - Rs)`. This replaces the design-rate assumption in
     `_unloading_gas_transfer`.
   *Test*: check the arithmetic chain of Table 3.478 (Pwf 2680 / 2225 / 1685 / 1540 / 1225; Q = R_inj q; TCF; 897 /
   962 / 1019 / 947 / 990 / 1030 adjustments).
6. **Valve flow models (11V2) as a new module**, independent of design:
   * (i) Cv/Xt rate equation. *Test*: **231 Mscf/d** (5.5.1).
   * (ii) App. A static stem position. *Test*: P_voT 919, P_vcT 837, **dx 0.049**, **Cv 0.774**, **Xt 0.40**,
     **Y 0.641**, **496 Mscf/d**.
   * (iii) App. C load-rate fit. *Test*: **606.7 psi/in, LST 0.098 in** (0.5 % tolerance).
   * (iv) TUALP orifice and variable area with the printed coefficients (a -0.838, c 0.844, r_crit 0.67, N_max 0.55,
     m 0.00127, b 1.25, Ft 0.09). *Smoke tests*: the orifice equals TC within 1 % at my check point (1249 vs 1257);
     Ppdc 250 / Ppdmax 470 for PvcT 614, Piod 650.
   Store the coefficients in `ValveType`. The Camco R-20 entry already has R values.
7. **Multipoint and throttling operating-point solver** (5.2(c)), using item 6. Show multipoint diagnostics (gas through
   more than one valve) and use them in `design_stability`.
8. **Performance-curve regression tests**, all chart-based, so +/-5-10 %:
   * Fonseca Table 3.419 (max 1412 B/D near 1 MMscf/d, injection about 4985 ft).
   * Pacheco Table 3.434 rows (variable Pwh, 4000-ft 2.5-in flowline, Psep 80).
   * Marin constant-Pwh maximum (HB 1500 B/D at 5350 ft; limited 600 Mscf/d gives 1420 / 1395 B/D).
   * Example #10 TGLR optimum at 600 scf/B (2365 B/D).
   The economic slope X = qG/(PI P) can be added as an alternative design-rate option. *Tests*: Ex 12 gives
   **X = 0.1**; Pacheco gives **0.0417**.
9. **Rules-of-thumb warnings in the UI**: Table 3.42 rate limits per tubing size; maximum continuous depth
   `(Pso - Pwh)/0.15`; 100 psi per 1000 ft; 150-250 scf/B per 1000 ft; valves no closer than about 90 ft; ports
   >= 3/8 in on intermittent-type valves warn of non-closing upper valves.
10. **Diagnostics (optional)**: from a measured flowing survey, locate the gradient breaks (operating valve, multipoint,
    tubing leak), the balance point against the next valve depth, and "valve should be open but is not". Brown 3.632
    has the rules. Recorder-chart heuristics (3.6372) could go into a troubleshooting help page.

### 5.4 Uncertainties to keep in mind
* Brown's examples were made with **gradient curves and charts**: Fig. 3A.1 gas F factors (ideal-gas-like), Fig. 3C.1
  choke chart (0.65 / 60 F basis), Kirkpatrick temperatures. Only the algebraic examples (Sections 1.2-1.9, 2.6.6, 3.1,
  4.2-4.6) are exact tests.
* Garbled or inconsistent items are marked [?] above:
  * Ex 9 temperature (166 vs 176 F);
  * Shiu Tsurf (130.3 vs printed 180.3);
  * Table 3.466 (pressure valves with closing-pressure drop);
  * Table 3.470 casing column (some values);
  * Table 3.478 valve 5 rate (575 vs 460);
  * Table 3.480 vertical Pwf;
  * TUALP Eq. B.5.2 form;
  * the App. A Z-ratio in P_voT and the critical check (Xt vs Fk Xt);
  * the annulus maximum-depth divisor (Eq. 3.48).
