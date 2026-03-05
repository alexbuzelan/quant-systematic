# Nelson–Siegel and Dynamic Nelson–Siegel Yield Curve Models

**1. Nelson–Siegel (NS) Model**

The **Nelson–Siegel model** provides a parsimonious representation of the yield curve using three latent factors.  
For a maturity $( \tau )$, the yield at time \( t \) is given by:

```math
y_t(\tau) =
L_t
+ S_t \frac{1 - e^{-\lambda \tau}}{\lambda \tau}
+ C_t \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right)
```

where:

- $L_t$ — **Level factor** (long-term interest rate component)
- $S_t$ — **Slope factor** (difference between short and long rates)
- $C_t$ — **Curvature factor** (medium-term hump in the yield curve)
- $\lambda$ — decay parameter controlling where the curvature loading peaks
- $\tau$ — maturity

**Interpretation of factors**

| Factor | Economic interpretation | Typical effect |
|------|------|------|
| Level | Long-run interest rate | Parallel shift of the curve |
| Slope | Short vs long rate spread | Steepening / flattening |
| Curvature | Medium-term hump | Changes around medium maturities |

In the **static Nelson–Siegel model**, the factors $(L_t, S_t, C_t)$ are estimated **independently for each date** using cross-sectional regressions across maturities.

---

**2. Dynamic Nelson–Siegel (DNS)**

The **Dynamic Nelson–Siegel model** introduces time-series dynamics for the factors.

Let

$$
\beta_t =
\begin{bmatrix}
L_t \\
S_t \\
C_t
\end{bmatrix}
$$

The factor dynamics are typically modeled as:

$$
\beta_t = \mu + \Phi \beta_{t-1} + \varepsilon_t
$$

where:

- $( \Phi )$ is a **3×3 transition matrix**
- $( \mu )$ is a constant vector
- $( \varepsilon_t )$ is a vector of shocks

Thus:

**Dynamic Nelson–Siegel = Nelson–Siegel representation + VAR dynamics for the factors**

<u>Personal Notes:</u> DNS = VAR for the factors

---

**3. State-Space Representation**

DNS can also be written as a **state-space model**.

**Measurement equation (yields)**

$$
y_t(\tau) = \Lambda(\tau) \beta_t + \epsilon_t
$$

**State equation (factor dynamics)**

$$
\beta_t = \mu + \Phi \beta_{t-1} + \eta_t
$$

This representation allows estimation using the **Kalman filter**.

<u>Pers Note:</u>
    
    This means:
    
    - factors are not estimated independently
    - estimation uses all time-series information

---

**4. Two-Step (Diebold–Li) Estimation**

A common empirical implementation proceeds in two steps:

1. **Cross-sectional estimation**

Estimate $(L_t, S_t, C_t)$ for each time period using OLS.

2. **Time-series estimation**

Fit a VAR model to the factor series:

$$
\beta_t = \mu + \Phi \beta_{t-1} + u_t
$$

---

**5. Macro-Augmented DNS**

The model can be extended by allowing macroeconomic variables to influence the factor dynamics:

$$
\beta_t = \mu + \Phi \beta_{t-1} + \Gamma X_t + u_t
$$

where $(X_t)$ may include variables such as:

- inflation
- GDP growth
- output gap
- policy rate
- unemployment

This extension is often called a **macro-augmented Nelson–Siegel model** or a **macro-finance term structure model**.

---

**Key Takeaway**

- **NS model:** cross-sectional representation of the yield curve  
- **DNS model:** NS + time-series dynamics for factors  
- **Macro DNS:** DNS + macroeconomic drivers
