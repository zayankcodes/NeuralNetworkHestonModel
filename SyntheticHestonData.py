import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from HestonAnalytical import heston_price, implied_volatility
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
from pyDOE import lhs
def generate_params_lhs(num_samples=2000):

    bounds = {
       'kappa': (0.5, 5.0),
       'theta': (0.01, 1.0),
       'sigma': (0.01, 1.0),
       'rho':   (-0.9, -0.1),
       'v0':    (0.01, 1.0),
       'r':     (0.0, 0.1),
       'q':     (0.0, 0.05)
    }
    

    dim = len(bounds)
    raw_lhs = lhs(dim, samples=num_samples)  
    

    param_list = []
    keys = list(bounds.keys())
    for row in raw_lhs:
 
        single_param = []
        for i, key in enumerate(keys):
            lo, hi = bounds[key]
            val = lo + (hi - lo)*row[i]
            single_param.append(val)
        

        kappa, theta, sigma, rho, v0, r, q = single_param
        

        if 2.0 * kappa * theta >= sigma**2:
            param_list.append((kappa, theta, sigma, rho, v0, r, q))
    
    return param_list


def param_distance(p, p_ref, scales=None):

    arr_p = np.array(p)
    arr_ref = np.array(p_ref)
    if scales is not None:
        return np.sqrt(np.sum(((arr_p - arr_ref)/scales)**2))
    else:
        return np.sqrt(np.sum((arr_p - arr_ref)**2))


p_ref = (2.0, 0.3, 0.2, -0.5, 0.2, 0.02, 0.01)

scales_ref = np.array([3.0, 0.5, 0.3, 0.4, 0.5, 0.05, 0.02]) 



strike_fine = np.linspace(0.8, 1.2, 40)     
maturity_fine = np.linspace(30/365.25, 2.0, 35)   

strike_coarse = np.linspace(0.8, 1.2, 15)  
maturity_coarse = np.linspace(30/365.25, 2.0, 10) 

DIST_THRESHOLD = 0.8

def process_param(param, S0=1.0):

    (kappa, theta, sigma, rho, v0, r, q) = param
    dist_ = param_distance(param, p_ref, scales_ref)
    if dist_ < DIST_THRESHOLD:
        strikes = strike_fine
        maturities = maturity_fine
    else:
        strikes = strike_coarse
        maturities = maturity_coarse
    data_points = []
    for T in maturities:
        for K in strikes:
            px = heston_price(S0, K, r, T, q, kappa, theta, sigma, rho, v0)
            iv = implied_volatility(px, S0, K, r, T, q)
            if 0 < iv < 3.0:
                data_points.append({
                    'kappa': kappa,
                    'theta': theta,
                    'sigma': sigma,
                    'rho': rho,
                    'v0': v0,
                    'r': r,
                    'q': q,
                    'strike': K,
                    'maturity': T,
                    'implied_volatility': iv
                })
    return data_points

if __name__ == "__main__":
    np.random.seed(42)
    num_samples = 10000
    param_list = generate_params_lhs(num_samples)
    print(f"Generated {len(param_list)} feasible parameter sets.")


    all_results = []
    with tqdm_joblib(tqdm(total=len(param_list), desc="Generating Heston Data")):
        all_results = Parallel(n_jobs=-1)(
            delayed(process_param)(p) for p in param_list
        )


    dataset = []
    for sub in all_results:
        dataset.extend(sub)
    df = pd.DataFrame(dataset)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("heston_dataset.csv", index=False)
    print(f"Final dataset size: {len(df)}")