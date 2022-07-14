import itertools
import numpy
import math
import hoomd
import gsd.hoomd
import sys
import random
import timeit

N_particles = int(sys.argv[1])
temp = float(sys.argv[2])
delta_t = float(sys.argv[3])
t_eq = int(sys.argv[4])
t_run = int(sys.argv[5])
t_write = float(sys.argv[6]) #THIS NEEDS TO STAY THE SAME FOR ALL RUNS
#final_rho = float(sys.argv[3])
random_seed = int(random.randrange(0,65535))
time_conversion = 1.0/delta_t

time1 = timeit.default_timer()
###SYSTEM SETUP
spacing = 1.3
K = math.ceil(N_particles**(1 / 3))
L = K * spacing
x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))

snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.typeid = [0] * int(0.5*N_particles) + [1] * int(0.5*N_particles) #particle 'B' is 20%
snapshot.configuration.box = [L, L, L, 0, 0, 0]
snapshot.particles.types = ['A','B']
snapshot.particles.mass[0:int(0.5*N_particles)] = 1
snapshot.particles.mass[1:int(0.5*N_particles)] = 2

with gsd.hoomd.open(name='lattice.gsd', mode='xb') as f:
    f.append(snapshot)
time2 = timeit.default_timer()
print("Setup complete.")
print("Setup time=",time2-time1)

###RANDOMIZE
## Initialize state
print("Randomizing...")
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=random_seed)
sim.create_state_from_gsd(filename='lattice.gsd')

## Setup LJ Integrator
integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell(buffer=0.1)
lj = hoomd.md.pair.LJ(nlist=cell)
#   Define pair potential
epsilon_AA = 1.0
sigma_AA = 1.0
lj.params[('A', 'A')] = dict(epsilon=epsilon_AA, sigma=sigma_AA)
lj.r_cut[('A', 'A')] = 2.5*sigma_AA
print("r_cut_AA=",lj.r_cut[('A', 'A')])
epsilon_AB = 1.0
sigma_AB = 1.1
lj.params[('A', 'B')] = dict(epsilon=epsilon_AB, sigma=sigma_AB)
lj.r_cut[('A', 'B')] = 2.5*sigma_AB
epsilon_BB = 1.0
sigma_BB = 1.2
lj.params[('B', 'B')] = dict(epsilon=epsilon_BB, sigma=epsilon_BB)
lj.r_cut[('B', 'B')] = 2.5*sigma_BB
r_buf = 0.3*max(lj.r_cut[('A', 'A')],lj.r_cut[('A', 'B')],lj.r_cut[('B', 'B')])
cell.buffer = r_buf
#   Assign force to integrator and integrator to simulation
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=temp, filter=hoomd.filter.All(), tau=time_conversion*delta_t)
integrator.methods.append(nvt)
sim.operations.integrator = integrator
snapshot = sim.state.get_snapshot()

## Randomize velocities
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)
#   Prints
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)

## Randomize positions
time3 = timeit.default_timer()
sim.run(100000)
time4 = timeit.default_timer()
## Write randomized state to file
hoomd.write.GSD.write(state=sim.state, filename='random.gsd', mode='xb')
print("RANDOMIZED")
print("Randomization time=",time4-time3)
print("Velocities:\n",sim.state.get_snapshot().particles.velocity[0:5])
print("Thermodynamic properties of snapshot:")
print("DOF=",thermodynamic_properties.degrees_of_freedom)
print("Particles=",thermodynamic_properties.num_particles)
print("KE=",thermodynamic_properties.kinetic_energy)
print("kT=",thermodynamic_properties.kinetic_temperature)
print("V=",thermodynamic_properties.volume**(1.0/3.0))

###COMPRESS
## Initialize state
print("Compressing...")
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=random_seed)
sim.create_state_from_gsd(filename='lattice.gsd')

## Setup LJ Integrator
integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell(buffer=r_buf)
lj = hoomd.md.pair.LJ(nlist=cell)
#   Define pair potential
epsilon_AA = 1.0
sigma_AA = 1.0
lj.params[('A', 'A')] = dict(epsilon=epsilon_AA, sigma=sigma_AA)
lj.r_cut[('A', 'A')] = 2.5*sigma_AA
epsilon_AB = 1.5
sigma_AB = 0.8
lj.params[('A', 'B')] = dict(epsilon=epsilon_AB, sigma=sigma_AB)
lj.r_cut[('A', 'B')] = 2.5*sigma_AB
epsilon_BB = 0.5
sigma_BB = 0.88
lj.params[('B', 'B')] = dict(epsilon=epsilon_BB, sigma=epsilon_BB)
lj.r_cut[('B', 'B')] = 2.5*sigma_BB
#   Assign force to integrator and integrator to simulation
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=temp, filter=hoomd.filter.All(), tau=time_conversion*delta_t)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

## Setup compressor
print(sim.timestep)
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=10000)
initial_rho = sim.state.N_particles / sim.state.box.volume
print("rho_i=",initial_rho)
initial_box = sim.state.box
print("V_i=",initial_box.volume**(1.0/3.0))
final_box = hoomd.Box.from_box(initial_box)  # make a copy of initial_box
final_rho = 0.75 #same as Kim & Saito / Wahnstrom
print("rho_f=",final_rho)
final_box.volume = sim.state.N_particles / final_rho
print("V_f=",final_box.volume**(1.0/3.0)) #Kob & Andersen's is 9.4 in units of sigma_AA
t_r = int(10e4*final_rho/initial_rho)
ramp.t_ramp = t_r
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)

## Run compressor
time5 = timeit.default_timer()
sim.run(t_r)
time6 = timeit.default_timer()
print("COMPRESSED")
print("Compression time=",time6-time5)
print("Compression successful?",abs(sim.state.box.volume - final_box.volume)/final_box.volume >= 0.01) #check compression success
print("Initial:\n",initial_box)
print("Final:\n",final_box)
print("Goal:\n",sim.state.box)
#sim.operations.updaters.remove(box_resize) #remove compressor, don't need it anymore
print("Thermodynamic properties of snapshot:")
print("DOF=",thermodynamic_properties.degrees_of_freedom)
print("Particles=",thermodynamic_properties.num_particles)
print("KE=",thermodynamic_properties.kinetic_energy)
print("kT=",thermodynamic_properties.kinetic_temperature)
print("V_therm=",thermodynamic_properties.volume**(1.0/3.0))
print("V_box=",sim.state.box.volume**(1.0/3.0))

###EQUILIBRATION
print("Equilibrating...")
time7 = timeit.default_timer()
sim.run(time_conversion*t_eq)
time8 = timeit.default_timer()
hoomd.write.GSD.write(state=sim.state, filename='equilibrated.gsd', mode='xb')
#   Prints
print("EQUILIBRATED")
print("Equilibration time=",time8-time7)
print(sim.state.get_snapshot().particles.velocity[0:5])
print("Thermodynamic properties of snapshot:")
print("DOF=",thermodynamic_properties.degrees_of_freedom)
print("Particles=",thermodynamic_properties.num_particles)
print("KE=",thermodynamic_properties.kinetic_energy)
print("kT=",thermodynamic_properties.kinetic_temperature)
print("V=",thermodynamic_properties.volume**(1.0/3.0))

###PRODUCTION
print("Production running...")
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=random_seed)
sim.create_state_from_gsd(filename='equilibrated.gsd')

## Setup LJ Integrator
integrator = hoomd.md.Integrator(dt=delta_t)
cell = hoomd.md.nlist.Cell(buffer=r_buf)
lj = hoomd.md.pair.LJ(nlist=cell)
#   Define pair potential
epsilon_AA = 1.0
sigma_AA = 1.0
lj.params[('A', 'A')] = dict(epsilon=epsilon_AA, sigma=sigma_AA)
lj.r_cut[('A', 'A')] = 2.5*sigma_AA
epsilon_AB = 1.5
sigma_AB = 0.8
lj.params[('A', 'B')] = dict(epsilon=epsilon_AB, sigma=sigma_AB)
lj.r_cut[('A', 'B')] = 2.5*sigma_AB
epsilon_BB = 0.5
sigma_BB = 0.88
lj.params[('B', 'B')] = dict(epsilon=epsilon_BB, sigma=epsilon_BB)
lj.r_cut[('B', 'B')] = 2.5*sigma_BB
#   Assign force to integrator and integrator to simulation
integrator.forces.append(lj)
nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
integrator.methods.append(nve)
sim.operations.integrator = integrator

## Setup logger and writers
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
logger = hoomd.logging.Logger()
logger.add(thermodynamic_properties,quantities=['kinetic_energy','kinetic_temperature'])
log_writer = hoomd.write.GSD(filename='log.gsd', trigger=hoomd.trigger.Periodic(int(time_conversion*t_write)),unwrap_full=True)
log_writer.log = logger
sim.operations.writers.append(log_writer)

traj_writer = hoomd.write.DCD(filename='trajectory.dcd', trigger=hoomd.trigger.Periodic(int(time_conversion*t_write)),unwrap_full=True)
sim.operations.writers.append(traj_writer)

## Run
starttime = timeit.default_timer()
sim.run(time_conversion*t_run)
stoptime = timeit.default_timer()
print("DONE")
print("Production time=",stoptime-starttime)
print("Thermodynamic properties of snapshot:")
print(sim.state.get_snapshot().particles.velocity[0:5])
print("DOF=",thermodynamic_properties.degrees_of_freedom)
print("Particles=",thermodynamic_properties.num_particles)
print("KE=",thermodynamic_properties.kinetic_energy)
print("kT=",thermodynamic_properties.kinetic_temperature)
print("V=",thermodynamic_properties.volume**(1.0/3.0))