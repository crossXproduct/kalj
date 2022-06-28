import itertools
import numpy
import math
import hoomd
import gsd.hoomd

#STEP 1: INITIALIZATION
# Set number of particles
m = 4
N_particles = 4 * m**3

# Set up lattice of positions
spacing = 1.3
K = math.ceil(N_particles**(1/3))
L = K * spacing
x = numpy.linspace(-L/2, L/2, endpoint=False)
position = list(itertools.product(x, repeat=3))

# Save snapshot of initial lattice
snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.typeid = [0] * N_particles
snapshot.configuration.box = [L,L,L,0,0,0]

# Define a list of particle types in the snapshot by their names
snapshot.particles.types = ['A']

# Save initial snapshot to a file
with gsd.hoomd.open(name='lattice.gsd', mode='xb') as f:
    f.append(snapshot)


###STEP 2: RANDOMIZATION
## This step requires running a simulation, so create a simulation object and assign it a device to run on
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=1) #NOTE: define a randomly generated seed for real runs

## Set up the initial condition from previous state
sim.create_state_from_gsd(filename='lattice.gsd')

## Set up an integrator to perform MD operations on the system state
integrator = hoomd.md.Integrator(dt=0.005) #the timestep used by the (Verlet?) MD algorithm (Kob & Andersen use 0.01 or 0.02)
#   Set up a pair force object (using LJ potential) to be used by the integrator
#   Set up neighbor list for force operator to use
cell = hoomd.md.nlist.Cell(buffer=0.4) #buffer calculates some neighbors outside cutoff to reduce number of times it has to calculate the list
#   Set up operator
lj = hoomd.md.pair.LJ(nlist=cell)
#   Set the LJ force object's parameters for interactions between 'A' and 'A' particles
#   (same logic applies to interactions between different types, just add another set of lines replacing one 'A' with a 'B'
#   to define A-B interactions, then a set to define B-B interactions, etc.)
lj.params[('A','A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A','A')] = 2.5 #the LJ potential cutoff radius (for numerical purposes)
#   Add the force object to the integrator
integrator.forces.append(lj)
#   Set up a 'method' to be used by the integrator (defines the type of ensemble) and add it to the integrator
nvt = hoomd.md.methods.NVT(kT=1.5, filter=hoomd.filter.All(), tau=1.0) #using NVT ensemble for all particles, with thermostat parameter 1.0 (Nosé-Hoover thermostat)
integrator.methods.append(nvt)
# Add the integrator to the simulation object
sim.operations.integrator = integrator

## Give particles random (nonzero) initial velocities for integrator to work with
#   Print velocity of system state to demonstrate velocities before assignment
snapshot = sim.state.get_snapshot()
print(snapshot.particles.velocity[0:5])
#   Assign random velocities
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)
#   Print new velocities
snapshot = sim.state.get_snapshot()
print(snapshot.particles.velocity[0:5])

## Set up and assign a ThermodynamicQuantities Compute to calculate (instantaneous) thermo properties of system state (i.e. of a single snapshot)
thermo_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermo_properties)
sim.run(0) #must call "run" for properties to be available
print(thermo_properties.degrees_of_freedom, thermo_properties.kinetic_energy, thermo_properties.kinetic_temperature)

## Run the simulation for some time to randomize
sim.run(10000)
#   Prints to check
print(thermo_properties.degrees_of_freedom, thermo_properties.kinetic_energy, thermo_properties.kinetic_temperature)
print(snapshot.particles.positions[0:5])
print(snapshot.particles.velocity[0:5])

## Save randomized system state to file
hoomd.write.GSD.write(state=sim.state, filename='random.gsd', mode='xb')


###STEP 3: COMPRESSION

cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu)
sim.create_state_from_gsd(filename='random.gsd')

integrator = hoomd.md.Integrator(dt=0.005) #the timestep used by the (Verlet?) MD algorithm (Kob & Andersen use 0.01 or 0.02)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A','A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A','A')] = 2.5
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=1.5, filter=hoomd.filter.All(), tau=1.0) #using NVT ensemble for all particles, with thermostat parameter 1.0 (Nosé-Hoover thermostat)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

# "Ramp up" the density of the system
ramp = hoomd.variant.Ramp(A=0,B=1, t_start=sim.timestep, t_ramp=20000) #t_start is timestep to start on, t_ramp is time over which to spread out compression
# Print initial density
rho = sim.state.N_particles / sim.state.box.volume
print(rho)
# Create new box at target density
initial_box = sim.state.box
final_box = hoomd.Box.from_box(initial_box)  # make a copy of initial_box
final_rho = 1.2
final_box.volume = sim.state.N_particles / final_rho
# Set up the "resize" operator and set to trigger frequently
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box,box2=final_box,variant=ramp,trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)
# Run the compression
sim.run(20000)
# Check compression success
print(sim.state.box == final_box)
# Remove "resize" operator
sim.operations.updaters.remove(box_resize)


###STEP 4: EQUILIBRATE
sim.run(5e5)


###STEP 5: RUN!!!
