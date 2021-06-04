"""Module to facilitate automatic nearest-neighbor search of most influetial
future data point."""

import torch


class NNProber():
    def __init__(self, *,
                 n_times=100, max_time_diff=10,
                 n_fluxes=100, max_flux_diff=10,
                 n_bands=5, n_classes=15
                 ):
        self.n_times = n_times
        self.max_time_diff = max_time_diff
        self.n_fluxes = n_fluxes
        self.max_flux_diff = max_flux_diff
        self.n_bands = n_bands
        self.n_classes = n_classes

    def probe(self, model, curve):
        """Probes the curve space for an optimal point using the given model"""
        baseline = model(curve)  # The baseline result to compare against

        times = torch.linspace(0, self.max_time_diff, self.n_times)
        fluxes = torch.linspace(-1 * self.max_flux_diff, self.max_flux_diff,
                                self.n_fluxes)
        search_grid = torch.zeros(self.n_times, self.n_bands)
        cache = torch.zeroes(self.n_fluxes)

        # loop over search grid (bands, then times, then fluxes)
        # calculate loss/diff per generated flux
        # aggregate for whole flux range
        # end loop
        # find best point in time+band (account for all possible fluxes in
        # aggregation)