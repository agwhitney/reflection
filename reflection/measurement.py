"""For analyzing monostatic reflection measurements made in the RF lab at NIST"""
# TODO maybe I should not have the index be the steps and have it be a column
# TODO Create object by pointing directly to data and/or results file.
# TODO I think I'm using gamma for something that is NOT gamma. Gamma is ratio and dB is just dB? Gain?
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.constants import c
from scipy.optimize import curve_fit


def fitting_function(x, a, b, phi, x0, beta) -> float:
    return np.abs(
        a - b * np.exp(1j*(2*beta*(x+x0) + phi)) / (x + x0)**2
    )


class Measurement:
    """Object representing the measurement.
    Holds the raw and fitted data."""
    def __init__(self, path: str, process: str = None, **kwargs) -> None:
        self.path = path
        self.usermeta = kwargs  # TODO clean this up. Combine with existing `meta` while keeping that isolated?

        self.nfreq : int
        self.nstep : int
        self.freqs : np.ndarray
        self.steps : np.ndarray
        self.meta : dict
        self._get_metadata()

        # Data from measurement z(x (index, cm), f (columns, GHz)) = complex(real, imag)
        # TODO I think it's better to have x as a column, not an index 
        self.data: pd.DataFrame = self._process(process)
        self.data.to_csv(os.path.join(self.path, "measurement_data.csv"))
        print("Saved measurement data!")

        self.results: pd.DataFrame = self.ripple_fit()
        self.results.to_csv(os.path.join(self.path, "results_data.csv"))
        print("Saved results!")

    
    @property
    def gamma(self):
        return 20*np.log10(self.results['b'])
    
    @property
    def steps_m(self):
        return self.steps / 100
    
    @property
    def freqs_hz(self):
        return self.freqs * 10**9


    def _get_metadata(self) -> None:
        try:
            p = os.path.join(self.path, 'meta.json')
            with open(p) as f:
                self.meta = json.load(f)
        except FileNotFoundError:
            print("No metadata file")
            self._force_metadata()
            return

        self.meta['long_suffix'] = self.meta['long_suffix'].replace('__', '_')  # Handles a typo
        
        self.nfreq = self.meta['frequency']['count']
        self.freqs = np.linspace(self.meta['frequency']['start'], self.meta['frequency']['end'], self.nfreq)
        self.nstep = self.meta['distance']['count']
        self.steps = np.linspace(self.meta['distance']['end'], self.meta['distance']['start'], self.nstep) # Sample moves closer to horn

    
    def _force_metadata(self) -> None:
        """Gets metadata by looping through the filenames in the given directory"""
        self.meta = {'short_suffix': 'cm.txt'}
        self.nfreq = 0
        self.nstep = 0
        steps = []
        freqs = []

        for f in os.listdir(os.path.join(self.path)):
            split = f.split('_')
            # step files {prefix}_{distance}.txt
            if len(split) == 2:
                self.meta['file_prefix'] = split[0] + '_'
                step = float(split[1].rstrip('cm.txt'))
                steps.append(step)
                self.nstep += 1
            # freq files {prefix}_{frequency}_{xspan}.txt
            elif len(split) == 3:
                self.meta['file_prefix'] = split[0] + '_'
                self.meta['long_suffix'] = 'GHz_' + split[2]
                freq = float(split[1].rstrip('GHz'))
                freqs.append(freq)
                self.nfreq += 1
            else:
                print("Skipped file '{}'".format(f))

        steps.sort()
        freqs.sort()
        self.steps = np.array(steps)
        self.freqs = np.array(freqs)
        

    def _process(self, process) -> pd.DataFrame:
        """Returns dataframe of complex data z(x, f)
        Uses the step files to gather rows
        """
        # TODO think about and correct the conditions. 
        if not process and os.path.exists(os.path.join(self.path, "measurement_data.csv")):
            print("Loading exisiting data csv")
            df = pd.read_csv(os.path.join(self.path, "measurement_data.csv"), index_col=0).apply(lambda x: np.complex128(x))
            df.columns = df.columns.astype(float)  # Have headers as floats, not strings
            return df     
        elif not process or process == 'step':
            rows = []
            for x in self.steps:
                filename = f"{self.meta['file_prefix']}{x:.3f}{self.meta['short_suffix']}"
                data = pd.read_csv(os.path.join(self.path, filename), delimiter='\t', header=None)
                rows.append(data[1] + 1j*data[2])
            df = pd.DataFrame(rows, dtype=complex, index=self.steps)
            df.columns = self.freqs  # Not sure why but this can't be set in declaration
            return df
        elif process == 'freq':
            cols = []
            for f in self.freqs:
                filename = f"{self.meta['file_prefix']}{f:.2f}{self.meta['long_suffix']}"
                data = pd.read_csv(os.path.join(self.path, filename), delimiter='\t', header=None).sort_values(0)
                cols.append(data[1] + 1j*data[2])
            df = pd.DataFrame(cols, dtype=complex).transpose()
            df.index = self.steps
            df.columns = self.freqs
            return df
            

    def ripple_fit(self, overwrite=False) -> pd.DataFrame:
        """Ripple method solves for the standing wave created in x between the antenna and target at each frequency
        Units are all in SI (meters, Hz).
        """
        # TODO add override flag
        if os.path.exists(os.path.join(self.path, "results_data.csv")):
            if overwrite: # Untested
                print("Overwriting an existing results file.")
            else:
                print("Loading existing results csv")
                df = pd.read_csv(os.path.join(self.path, "results_data.csv"), index_col=0)
                return df

        x0 = self.steps_m.min()
        travel_length = self.steps_m.max() - self.steps_m.min()
        stepsize = self.meta['distance'].get('interval', travel_length * 100 / self.nstep) / 100  # metadata file has data in cm

        rows = []
        for i, f in enumerate(self.freqs_hz):
            half_wavelength = (c / f) / 2
            wavenumber = (2*np.pi) / (2 * half_wavelength)

            # Restrict the measurement to an integer number of half-wavelengths.
            h_idx = int(np.round(
                np.floor(travel_length / half_wavelength) * (half_wavelength / stepsize)
            ))
            mmt = self.data.iloc[:h_idx, i]
            avg = np.abs(mmt.mean())
            pp = (mmt.abs().max() + mmt.abs().min()) / 2
            mm = (mmt.abs().max() - mmt.abs().min()) / 2
            if np.abs(avg - pp) < np.abs(avg - mm):
                a0, b0 = mm, pp
            else:
                a0, b0 = pp, mm

            func = lambda x, a, b, phi: fitting_function(x, a, b, phi, x0=x0, beta=wavenumber)
            x = self.steps_m[:h_idx] - x0
            y = mmt.abs()
            seeds = [avg, a0*x0**2, 0]
            bounds = ( [0.95 * i if i != 0 else -np.pi for i in seeds], [1.05 * i if i != 0 else np.pi for i in seeds] )
            fits, errs = curve_fit(
                func, x, y, p0=seeds, bounds=bounds
            )
            errs = np.sqrt(np.diag(errs))
            rows.append([i for j in zip(fits, errs) for i in j])

        cols = ['a', 'da', 'b', 'db', 'phi', 'dphi']
        results = pd.DataFrame(rows, columns=cols)
        results['f'] = self.freqs
        return results
    

    def plot_fit(self, frequency: float, savepath: str = None) -> None:
        # TODO slider
        wavenumber = (2*np.pi * frequency*10**9) / c
        params = self.results.loc[frequency, ['a', 'b', 'phi']]
        print(self.results.loc[frequency])  # TODO pretty it up
        mmt = self.data.loc[:, frequency]

        fig, ax = plt.subplots()
        ax.set(
            title=f"Measurement and fit at $f$ = {frequency} GHz",
            xlabel="Sample-Antenna separation (m)",
            ylabel="Voltage ratio (arb.)"
        )
        x = self.steps_m
        # ax.plot(x, mmt.apply(lambda x: np.real(x)), label="Real measurement data", marker='o')
        # ax.plot(x, mmt.apply(lambda x: np.imag(x)), label="Imag measurement data", marker='o')
        ax.plot(x, mmt.abs(), label="Measurement data (magnitude)", marker='o')
        ax.plot(x, fitting_function(x, *params, x0=x.min(), beta=wavenumber), label="Ripple fit result", linestyle='--')
        ax.grid()
        ax.legend()
        plt.show()

        if savepath:
            fig.savefig(savepath, dpi=200)
