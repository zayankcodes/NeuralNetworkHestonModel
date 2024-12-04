# NeuralNetworkHestonModel
A neural network based calibration method for the Heston model that performs the calibration task at high speeds for the full implied volatility surface


# __Introduction__

## The Black-Scholes Model

In the Black-Scholes model, the underlying asset follows a Geometric Brownian motion. That is, given a probability space (Ω, 𝓕, ℙ) supporting a one-dimensional Brownian motion (Wᵗ)ₜ≥₀, the asset price process (Sₜ)ₜ≥₀ is the unique strong solution to the following SDE:

### dSₜ = rSₜdt + σSₜdWₜ, S₀ > 0 


where r > 0 is a constant risk-free interest rate and σ > 0 is a constant instantaneous volatility.

---

## Implied Volatility

Option prices are often discussed with regard to their implied volatility. The implied volatility, or Black-Scholes implied volatility, is the unique value of the volatility parameter such that the Black-Scholes pricing formula equals the given price of a specific option.

The Black-Scholes formula for a European Call option is defined as:


### Cᴮˢ(σ) := Cᴮˢ(Sₜ, t, K, T, σ) := SₜN(d₊) - Ke⁻ʳ⁽ᵀ⁻ᵗ⁾N(d₋),

where

### d₊ = 1 / (σ√(T - t)) [ln(Sₜ / K) + (r + σ² / 2)(T - t)], 

### d₋ = d₊ - σ√(T - t).


Here, Sₜ is the price of the asset at time t, K is the options strike, T is the maturity of the option, t is the current time of pricing, and σ is the volatility parameter.

Given an observed call option price C(K, T), the implied volatility σᴮˢ(K, T) is defined as:


### C(K, T) = Cᴮˢ(K, T, σᴮˢ(K, T)).

This solution is unique because Cᴮˢ is strictly increasing in σ.



---

# __Stochastic Volatility__

## The Heston Model

The Heston model introduces stochastic volatility, which evolves over time according to a stochastic differential equation (SDE). It accounts for the volatility smile observed in markets.
