using Oceananigans
using Oceananigans.Units
using GLMakie

arch = GPU()
Lx = 1000kilometers # east-west extent [m]
Ly = 1000kilometers # north-south extent [m]
Lz = 400 # depth [m]

Nx = 256
Ny = 256
Nz = 128

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       topology = (Periodic, Bounded, Bounded))

using Oceananigans.BuoyancyModels: g_Earth

g = 9.81
a = 0.8 # m
λ = 60  # m
k = 2π / λ# m⁻¹
σ = sqrt(g * k) # s⁻¹
Uˢ = a^2 * k * σ # m s⁻¹

parameters = (; k, Uˢ)
∂z_uˢ(z, t, p) = 1 / (2 * p.k) * p.Uˢ * exp(2 * p.k * z)
stokes_drift = UniformStokesDrift(; ∂z_uˢ, parameters)

model = NonhydrostaticModel(; grid, stokes_drift,
                            coriolis = BetaPlane(latitude = -45),
                            buoyancy = BuoyancyTracer(),
                            tracers = (:b, :c),
                            advection = WENO())

ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

N² = 1e-5 # [s⁻²] buoyancy frequency / stratification
M² = 1e-7 # [s⁻²] horizontal buoyancy gradient

Δy = 100kilometers # width of the region of the front
Δb = Δy * M²       # buoyancy jump associated with the front
ϵb = 1e-2 * Δb     # noise amplitude
hc = Lz / 10       # tracer decay scale

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()
cᵢ(x, y, z) = exp(-y^2 / 2Δy^2) * exp(z / hc)

set!(model, b=bᵢ)

simulation = Simulation(model, Δt=20minutes, stop_time=20days)
conjure_time_step_wizard!(simulation, IterationInterval(20), cfl=0.2, max_Δt=20minutes)

using Printf

wall_clock = Ref(time_ns())

function print_progress(sim)
    u, v, w = model.velocities
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(elapsed),
            maximum(abs, interior(u)),
            maximum(abs, interior(v)),
            maximum(abs, interior(w)),
            prettytime(sim.Δt))

    wall_clock[] = time_ns()
    
    return nothing
end

add_callback!(simulation, print_progress, IterationInterval(100))

# ## Diagnostics/Output
#
# Here, we save the buoyancy, ``b``, at the edges of our domain as well as
# the zonal (``x``) average of buoyancy.

b = model.tracers.b
u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)
B = Average(b, dims=1)
U = Average(u, dims=1)
V = Average(v, dims=1)

filename = "baroclinic_adjustment"
save_fields_interval = 0.5day

slicers = (east = (grid.Nx, :, :),
           north = (:, grid.Ny, :),
           bottom = (:, :, 1),
           top = (:, :, grid.Nz))

for side in keys(slicers)
    indices = slicers[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, (; b, ζ, w);
                                                       filename = filename * "_$(side)_slice",
                                                       schedule = TimeInterval(save_fields_interval),
                                                       overwrite_existing = true,
                                                       indices)
end

simulation.output_writers[:zonal] = JLD2OutputWriter(model, (; b=B, u=U, v=V);
                                                     filename = filename * "_zonal_average",
                                                     schedule = TimeInterval(save_fields_interval),
                                                     overwrite_existing = true)

# Now we're ready to _run_.

@info "Running the simulation..."

run!(simulation)

top_slice_filename = filename * "_top_slice.jld2"
zonal_average_filename = filename * "_zonal_average.jld2"

bt_top = FieldTimeSeries(top_slice_filename, "b")
ζt_top = FieldTimeSeries(top_slice_filename, "ζ")
wt_top = FieldTimeSeries(top_slice_filename, "w")
Ut = FieldTimeSeries(zonal_average_filename, "u")
Vt = FieldTimeSeries(zonal_average_filename, "v")
Nt = length(Vt)

# Next, we set up a plot with 4 panels. The top panels are large and square, while
# the bottom panels get a reduced aspect ratio through `rowsize!`.

set_theme!(Theme(fontsize=24))

fig = Figure(size=(1800, 1000))

axb = Axis(fig[1, 1], xlabel="x (km)", ylabel="y (km)", aspect=1)
axw = Axis(fig[1, 2], xlabel="x (km)", ylabel="y (km)", aspect=1, yaxisposition=:right)
axu = Axis(fig[2, 1], xlabel="y (km)", ylabel="z (m)")
axv = Axis(fig[2, 2], xlabel="y (km)", ylabel="z (m)", yaxisposition=:right)

rowsize!(fig.layout, 2, Relative(0.3))

# To prepare a plot for animation, we index the timeseries with an `Observable`,

slider = Slider(fig[3, 1:2], range=1:Nt, startvalue=1)
n = slider.value

b_top = @lift interior(bt_top[$n], :, :, 1)
ζ_top = @lift interior(ζt_top[$n], :, :, 1)
w_top = @lift interior(wt_top[$n], :, :, 1)
U = @lift interior(Ut[$n], 1, :, :)
V = @lift interior(Vt[$n], 1, :, :)

# and then build our plot:

hm = heatmap!(axb, b_top, colorrange=(0, Δb), colormap=:thermal)
hm = heatmap!(axw, w_top, colorrange=(-1e-4, 1e-4), colormap=:balance)
hm = heatmap!(axu, U; colorrange=(-5e-1, 5e-1), colormap=:balance)
hm = heatmap!(axv, V; colorrange=(-1e-1, 1e-1), colormap=:balance)

display(fig)

