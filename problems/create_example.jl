import Pkg;
Pkg.activate(joinpath(@__DIR__, ".."));
Pkg.instantiate()

using Random, Plots
using BasicRL



# env = GridWorld()
# reset!(env)
# for t in 1:50
#     action = rand(env.action_space)
#     next_state, reward, done = step!(env, action)
#     println("Step: $t, Action: $action, State: $next_state, Reward: $reward, Done: $done")
#     if done
#         break
#     end
# end
# render_grid(env)

env = GridWorld()
reset!(env)
policy = random_policy(env)
render_grid_policy(env, policy)

