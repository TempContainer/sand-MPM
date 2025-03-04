# sand-MPM

A taichi implementation of Drucker-Prager sand simulation using MLS-MPM. References:

- (SIGGRAPH 2016) Drucker-Prager Elastoplasticity for Sand Animation

with two volume correction methods featured in:

- (SIGGRAPH 2017) Multi-Species Simulation of Porous Sand and Water Mixtures
- (SIGGRAPH 2018) Animating Fluid Sediment Mixture in Particle-Laden Flows

Add an implementation of CPIC to handle two-way coupling with rigid body (only in 2D). Reference:

- (SIGGRAPH 2018) A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling

## Usage

- Directly run the python file
- Change `dim` in the code to change dimension
- Change `write_to_disk` to determine whether write to disk (for exportation)
- When in 3D, hold `RMB` to change perspective, then use `W`, `A`, `S`, `D`, `E`, `Q` to move
- Directly exit by `Ctrl-C`