import numpy as np
import torch
from gflownet.proxy.base import Proxy





class IsingEnergy(Proxy):
    def __init__(self, J: float = 1.0, h: float = 0.0, **kwargs):
        """
        Parameters
        ----------
        J : float
            Interaction strength between neighboring spins.
        h : float
            External magnetic field strength.
        """
        super().__init__(**kwargs)
        self.J = J
        self.h = h

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Ising model energy for a batch of spin configurations.

        Parameters
        ----------
        states : torch.Tensor
            A tensor of shape (batch_size, n_spins) where each element is either -1 or 1.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size,) containing the energy of each configuration.
        """

        #Ising 2D energy:
        batch_size, n, _ = states.shape
        

        """
        Compute energy for a batch of 2D Ising spin configurations.

        spins_batch: shape (B, L, L), values +1 or -1
        J: interaction strength
        h: external magnetic field
        Returns: energies, shape (B,)
        """

        # Horizontal neighbors (roll along columns)
        horiz = torch.sum(states * torch.roll(states, shifts=1, dims=2), dim=(1,2))
        # Vertical neighbors (roll along rows)
        vert = torch.sum(states * torch.roll(states, shifts=1, dims=1), dim=(1,2))

        energy = self.J * (horiz + vert) - self.h * torch.sum(states, dim=(1,2))
        return abs(energy)

