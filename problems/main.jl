import Pkg;
Pkg.activate(joinpath(@__DIR__, ".."));
Pkg.instantiate()

using Random, Plots
using BasicRL

# env = GridWorld()
# reset!(env)
# policy = random_policy(env)
# render_grid_policy(env, policy)



env = GridWorld()
V, policy = value_iteration(env)
# render_grid_policy(env, policy)
render_grid_policy_value(env, policy, V)



