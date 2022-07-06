import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    thermo_data = np.loadtxt("apr.thermo", skiprows=1)
    nb = np.loadtxt("apr.nb", skiprows=2)
    
    rho = nb * 10**39 * 1.67e-24 # ans in g/cm^3
    
    p = thermo_data[:, 3] * nb * 10**39 * 1.67e-24/939.6 # in g/cm^3

    e = (thermo_data[:, 9] + 1.0) * rho #  in g/cm^3

    out = pd.DataFrame()
    out["baryon_density"] = rho
    out["pressurec2"] = p
    
    out["energy_densityc2"] = e

    out.to_csv("apr.csv", index=False)
