using Oceananigans
using Oceananigans.Units
using Random
using Printf

Random.seed!(42)

f = 1e-4
N² = 1e-6 # [s⁻²] buoyancy frequency / stratification
Ri = 1
H = 200
h = 100
T = 14 # 6 # 10, 14, 18, 22
ϵ = 0.3
g = 9.81

L = sqrt(N²) * h / f # Rossby radius of deformation
Δy = L # width of the region of the front

M² = sqrt(f^2 * N² / Ri)
Δb = M² * Δy
σ = 2π / T
k = σ^2 / g # m⁻¹
Uˢ = a * ϵ * σ # m s⁻¹

arch = GPU()
Lx = 20L  # east-west extent [m]
Ly = 20L  # north-south extent [m]
Lz = H    # depth [m]

Nx = 192
Ny = 192
Nz = 48

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       halo = (5, 5, 5),
                       topology = (Periodic, Bounded, Bounded))

@show grid

filename = @sprintf("baroclinic_adjustment_Nx%d_H%d_Ri%d_T%d_ep%d", Nx, H, 10Ri, T, 10ϵ)

@inline ∂z_uˢ(z, t, p) = 2 * p.σ * p.ϵ^2 * exp(2 * p.k * z)
stokes_drift = UniformStokesDrift(; ∂z_uˢ, parameters=(; k, Uˢ, ϵ, σ))

model = NonhydrostaticModel(; grid, stokes_drift,
                            coriolis = FPlane(; f),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            advection = WENO(order=9))

@show model
@show model.stokes_drift

step(y, Δy) = min(max(0, y/Δy + 1/2), 1)
ramp(y, Δy=1) = max(0, y / Δy)

ϵb = 1e-2 * Δb # noise amplitude
bᵢ(x, y, z) = N² * z + Δb * step(y, Δy) * ramp(1 + z/h) + ϵb * randn()
set!(model, b=bᵢ)

simulation = Simulation(model, Δt=1minutes, stop_time=30days)
conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.7, max_Δt=2minutes)

wall_clock = Ref(time_ns())

function print_progress(sim)
    u, v, w = model.velocities
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] n: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
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
ξ = ∂z(u) - ∂x(w)
s = @at (Center, Center, Center) sqrt(u^2 + v^2)
B = Average(b, dims=1)
U = Average(u, dims=1)
V = Average(v, dims=1)
w² = Average(w^2, dims=1)
save_fields_interval = 1day

slicers = (east = (grid.Nx, :, :),
           north = (:, grid.Ny, :),
           bottom = (:, :, 1),
           top = (:, :, grid.Nz))

for side in keys(slicers)
    indices = slicers[side]

    simulation.output_writers[side] = JLD2OutputWriter(model, (; b, ζ, ξ, s, w);
                                                       filename = filename * "_$(side)_slice",
                                                       schedule = TimeInterval(save_fields_interval),
                                                       overwrite_existing = true,
                                                       indices)
end

simulation.output_writers[:zonal] = JLD2OutputWriter(model, (; b=B, u=U, v=V, w²);
                                                     filename = filename * "_zonal_average",
                                                     schedule = TimeInterval(save_fields_interval),
                                                     overwrite_existing = true)

# Now we're ready to _run_.

@info "Running the simulation..."

run!(simulation)

#=
using GLMakie

top_slice_filename = filename * "_top_slice.jld2"
zonal_average_filename = filename * "_zonal_average.jld2"

bt_top = FieldTimeSeries(top_slice_filename, "b")
ζt_top = FieldTimeSeries(top_slice_filename, "ζ")
wt_top = FieldTimeSeries(top_slice_filename, "w")
st_top = FieldTimeSeries(top_slice_filename, "s")
ξt_top = FieldTimeSeries(top_slice_filename, "ξ")
Nt = length(bt_top)

# Next, we set up a plot with 4 panels. The top panels are large and square, while
# the bottom panels get a reduced aspect ratio through `rowsize!`.

set_theme!(Theme(fontsize=24))

fig = Figure(size=(1000, 1000))

axb = Axis(fig[1, 1], xlabel="x (km)", ylabel="y (km)", aspect=1, xaxisposition=:top)
axξ = Axis(fig[1, 2], xlabel="x (km)", ylabel="y (km)", aspect=1, xaxisposition=:top, yaxisposition=:right)
axs = Axis(fig[2, 1], xlabel="x (km)", ylabel="y (km)", aspect=1)
axζ = Axis(fig[2, 2], xlabel="x (km)", ylabel="y (km)", aspect=1, yaxisposition=:right)

# To prepare a plot for animation, we index the timeseries with an `Observable`,

slider = Slider(fig[3, 1:2], range=1:Nt, startvalue=1)
n = slider.value

b_top = @lift bt_top[$n]
ζ_top = @lift ζt_top[$n]
ξ_top = @lift ξt_top[$n]
s_top = @lift st_top[$n]

# and then build our plot:

hm = heatmap!(axb, b_top, colorrange=(0, Δb), colormap=:thermal)
hm = heatmap!(axξ, ξ_top, colorrange=(-1e-2, 1e-2), colormap=:balance)
hm = heatmap!(axζ, ζ_top, colorrange=(-1e-4, 1e-4), colormap=:balance)
hm = heatmap!(axs, s_top, colorrange=(0, 1e0), colormap=:magma)

display(fig)
=#
