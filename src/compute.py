import numpy as np
from hoomd.custom import Action

class CylindricalFlowFieldProfiler(Action):
    """Measure average profiles in cylindrical coordinates (radial and axial).

    Computes radial profiles for number density, mass density, velocities, and temperature (kT).

    Args:
        num_bins (int): Number of bins in the radial direction.
        bin_ranges (2-tuple of tuples): Ranges for radial (r_min, r_max) and axial (z_min, z_max) directions.

    Example:
        flow_field = CylindricalFlowFieldProfiler(
            num_bins=100,
            bin_ranges=([r_min, r_max], [z_min, z_max])
        )
    """

    def __init__(self, num_bins, bin_ranges):
        super().__init__()
        self.num_bins = int(num_bins)
        self.bin_ranges = np.array(bin_ranges, dtype=float)

        # Verify the bin ranges
        if self.bin_ranges.shape != (2, 2):
            raise TypeError('bin_ranges must be a (2,2) array for radial and axial ranges in cylindrical coordinates')

        if self.bin_ranges[0, 1] <= self.bin_ranges[0, 0]:
            raise ValueError('Radial range must be increasing')

        # Initialize bin edges and centers for radial direction
        self.bin_edges_r = np.linspace(self.bin_ranges[0, 0], self.bin_ranges[0, 1], self.num_bins + 1)
        self.bin_centers_r = 0.5 * (self.bin_edges_r[:-1] + self.bin_edges_r[1:])
        self.bin_sizes_r = self.bin_edges_r[1:] - self.bin_edges_r[:-1]

        # Initialize arrays for storing profiles
        self._number_density = np.zeros(self.num_bins, dtype=float)
        self._mass_density = np.zeros(self.num_bins, dtype=float)
        self._number_velocity = np.zeros((self.num_bins, 3), dtype=float)  # Stores [v_r, v_theta, v_z]
        self._mass_velocity = np.zeros((self.num_bins, 3), dtype=float)    # Mass-averaged velocities
        self._kT = np.zeros(self.num_bins, dtype=float)  # Temperature profile
        self._counts = np.zeros(self.num_bins, dtype=int)
        self._num_samples = 0

    def act(self, timestep):
        """Compute flow profiles in cylindrical coordinates at the given timestep."""
        with self._state.cpu_local_snapshot as snap:
            pos = snap.particles.position
            vel = snap.particles.velocity
            mass = snap.particles.mass

            # Compute radial distance
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            r = np.sqrt(x**2 + y**2)

            # Filter particles within radial and axial bounds
            in_range_filter = (r >= self.bin_ranges[0, 0]) & (r < self.bin_ranges[0, 1]) & (z >= self.bin_ranges[1, 0]) & (z <= self.bin_ranges[1, 1])
            r_filtered = r[in_range_filter]
            vel_filtered = vel[in_range_filter]
            mass_filtered = mass[in_range_filter]
            x_filtered, y_filtered = x[in_range_filter], y[in_range_filter]

            # Radial velocity component (projected from Cartesian components)
            v_x, v_y, v_z = vel_filtered[:, 0], vel_filtered[:, 1], vel_filtered[:, 2]
            v_r = (x_filtered * v_x + y_filtered * v_y) / r_filtered
            v_r = np.where(r_filtered != 0, v_r, 0.0)  # Handle cases where r=0

            # Azimuthal velocity component
            v_theta = (-y_filtered * v_x + x_filtered * v_y) / r_filtered
            v_theta = np.where(r_filtered != 0, v_theta, 0.0)  # Handle cases where r=0

            # Bin particles by their radial distance
            bin_indices = np.digitize(r_filtered, self.bin_edges_r) - 1
            valid_bins = (bin_indices >= 0) & (bin_indices < self.num_bins)
            bin_indices = bin_indices[valid_bins]
            particles_in_bins = in_range_filter.nonzero()[0][valid_bins]

            # Accumulate number density, mass density, velocity, and temperature
            np.add.at(self._counts, bin_indices, 1)
            np.add.at(self._number_density, bin_indices, 1)  # Number density is count-based
            np.add.at(self._mass_density, bin_indices, mass_filtered)  # Sum of mass for mass density

            # Number-averaged velocity components
            np.add.at(self._number_velocity[:, 0], bin_indices, v_r[particles_in_bins])      # Radial
            np.add.at(self._number_velocity[:, 1], bin_indices, v_theta[particles_in_bins])  # Azimuthal
            np.add.at(self._number_velocity[:, 2], bin_indices, v_z[particles_in_bins])      # Axial

            # Mass-averaged velocity components (mass-weighted)
            np.add.at(self._mass_velocity[:, 0], bin_indices, v_r[particles_in_bins] * mass_filtered[valid_bins])
            np.add.at(self._mass_velocity[:, 1], bin_indices, v_theta[particles_in_bins] * mass_filtered[valid_bins])
            np.add.at(self._mass_velocity[:, 2], bin_indices, v_z[particles_in_bins] * mass_filtered[valid_bins])

            # Temperature profile (kT)
            np.add.at(self._kT, bin_indices, mass_filtered * (np.linalg.norm(vel_filtered, axis=1) ** 2) / 3)

            # Increment the sample count
            self._num_samples += 1

    def finalize_profiles(self):
        """Finalize the averaging of profiles after all sampling."""
        nonzero = self._counts > 0

        # Normalize densities
        bin_areas = 2 * np.pi * self.bin_centers_r * self.bin_sizes_r
        self._number_density[nonzero] /= (self._num_samples * bin_areas[nonzero])
        self._mass_density[nonzero] /= (self._num_samples * bin_areas[nonzero])

        # Average velocities
        self._number_velocity[nonzero] /= self._counts[nonzero, None]
        self._mass_velocity[nonzero] /= self._mass_density[nonzero, None]  # Mass-weighted average

        # Temperature (kT)
        self._kT[nonzero] /= self._counts[nonzero]

    @property
    def number_density(self):
        """Return the number density profile."""
        return self._number_density

    @property
    def mass_density(self):
        """Return the mass density profile."""
        return self._mass_density

    @property
    def number_velocity(self):
        """Return the number-averaged velocity profile as an array."""
        return self._number_velocity

    @property
    def mass_velocity(self):
        """Return the mass-averaged velocity profile as an array."""
        return self._mass_velocity

    @property
    def temperature(self):
        """Return the temperature profile."""
        return self._kT

    def write(self, filename):
        """Write the computed profiles to a file."""
        np.savez(
            filename,
            r=self.bin_centers_r,
            number_density=self.number_density,
            mass_density=self.mass_density,
            number_velocity=self.number_velocity,
            mass_velocity=self.mass_velocity,
            temperature=self.temperature,
        )
