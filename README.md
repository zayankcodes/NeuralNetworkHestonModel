# NeuralNetworkHestonModel
A neural network based calibration method for the Heston model that performs the calibration task at high speeds for the full implied volatility surface


# __Introduction__

## The Black-Scholes model

The Black-Scholes model assumes constant volatility. However, market data shows that implied volatility changes with strike price and maturity, leading to the volatility smile.

## Implied Volatility

Implied volatility represents the market's expectation of future volatility. It is derived from option prices using the Black-Scholes formula.

---

# __Stochastic Volatility__

## The Heston Model

The Heston model introduces stochastic volatility, which evolves over time according to a stochastic differential equation (SDE). It accounts for the volatility smile observed in markets.
