# NeuralNetworkHestonModel
A neural network based calibration method for the Heston model that performs the calibration task at high speeds for the full implied volatility surface


# __Introduction__

## The Black-Scholes model

In the Black-Scholes model, the underlying asset follows a Geometric Brownian motion. That is, given a probability space \( (\Omega, \mathcal{F}, \mathbb{P}) \) supporting a one-dimensional Brownian motion \( (W_t)_{t \geq 0} \), the asset price process \( (S_t)_{t \geq 0} \) is the unique strong solution to the following SDE:

\[
dS_t = r S_t dt + \sigma S_t dW_t, \quad S_0 > 0, \tag{3.1.1}
\]

where \( r > 0 \) is a constant risk-free interest rate and \( \sigma > 0 \) is a constant instantaneous volatility.


## Implied Volatility

Option prices are often discussed with regard to their implied volatility. The implied volatility, or Black-Scholes implied volatility, is the unique value of the volatility parameter such that the Black-Scholes pricing formula is equal to the given price of a specific option. The Black-Scholes formula for a European Call option is defined as follows:

\[
C^{BS}(\sigma) := C^{BS}(S_t, t, K, T, \sigma) := S_t \mathcal{N}(d_+) - K e^{-r(T-t)} \mathcal{N}(d_-), \tag{2.2.2}
\]

where

\[
d_+ = \frac{1}{\sigma \sqrt{T - t}} \left[ \ln \left( \frac{S_t}{K} \right) + \left( r + \frac{\sigma^2}{2} \right)(T - t) \right],
\]

\[
d_- = d_+ - \sigma \sqrt{T - t}.
\]

\( S_t \) is the price of the asset at time \( t \), \( K \) is the options strike, \( T \) is the maturity of the option, \( t \) is the current time of pricing, and \( \sigma \) is the volatility parameter.

Now, given an observed call option price \( C(K, T) \), the implied volatility for strike \( K \) and maturity \( T \) is defined as the value \( \sigma^{BS}(K, T) \) that solves [6, Equation 1.6 Page 3]:

\[
C(K, T) = C^{BS} \left( K, T, \sigma^{BS}(K, T) \right).
\]

This solution will be unique as the function \( C^{BS} \) is strictly increasing in \( \sigma \).


---

# __Stochastic Volatility__

## The Heston Model

The Heston model introduces stochastic volatility, which evolves over time according to a stochastic differential equation (SDE). It accounts for the volatility smile observed in markets.
