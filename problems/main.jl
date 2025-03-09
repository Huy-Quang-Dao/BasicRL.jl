import Pkg;
Pkg.activate(joinpath(@__DIR__, ".."));
Pkg.instantiate()

using Random, Plots
using BasicRL

### Test
# env = GridWorld()
# reset!(env)
# policy = random_policy(env)
# render_grid_policy(env, policy)


### Main
# Create environment
env = GridWorld()
reset!(env)

### Method
# # Value Iteration
# V, policy = value_iteration(env)
# Policy Iteration
V, policy = policy_iteration(env)

### Render
render_grid_policy_value(env, policy, V)



