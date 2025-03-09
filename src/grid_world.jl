using Random, Plots

mutable struct GridWorld
    env_size::Tuple{Int, Int}
    num_states::Int
    start_state::Tuple{Int, Int}
    target_state::Tuple{Int, Int}
    forbidden_states::Vector{Tuple{Int, Int}}
    agent_state::Tuple{Int, Int}
    action_space::Vector{Tuple{Int, Int}}
    reward_target::Float64
    reward_forbidden::Float64
    reward_step::Float64
    traj::Vector{Tuple{Float64, Float64}}
    
    function GridWorld(env_size::Tuple{Int, Int} = (5,5),
                       start_state::Tuple{Int, Int} = (1,1),
                       target_state::Tuple{Int, Int} = (5,5),
                       forbidden_states::Vector{Tuple{Int, Int}} = [(3,2),(4,3),(2,4),(4,5)],
                       action_space::Vector{Tuple{Int, Int}} = [(0,1), (1,0), (0,-1), (-1,0),(0,0)], # up , right , down, left,stay
                       reward_target::Float64 = 10.0,
                       reward_forbidden::Float64 = -5.0,
                       reward_step::Float64 = -1.0)
        new(env_size, env_size[1] * env_size[2], start_state, target_state, forbidden_states,
            start_state, action_space, reward_target, reward_forbidden, reward_step, [])
    end
end

function reset!(env::GridWorld)
    env.agent_state = env.start_state
    env.traj = [env.agent_state]
    return env.agent_state
end

function step!(env::GridWorld, action::Tuple{Int, Int})
    @assert action in env.action_space "Invalid action"
    next_state, reward = get_next_state_and_reward(env, env.agent_state, action)
    done = is_done(env, next_state)
    push!(env.traj, next_state)
    env.agent_state = next_state
    return next_state, reward, done
end

function get_next_state_and_reward(env::GridWorld, state::Tuple{Int, Int}, action::Tuple{Int, Int})
    x, y = state
    new_state = (x + action[1], y + action[2])
    
    if new_state[2] > env.env_size[2] || new_state[1] > env.env_size[1] ||
       new_state[2] < 1 || new_state[1] < 1
        return state, env.reward_forbidden
    elseif new_state == env.target_state
        return new_state, env.reward_target
    elseif new_state in env.forbidden_states
        return state, env.reward_forbidden
    else
        return new_state, env.reward_step
    end
end

is_done(env::GridWorld, state::Tuple{Int, Int}) = state == env.target_state


function render_grid(env::GridWorld)
    grid = fill(0, env.env_size)  

    for (i, j) in env.forbidden_states
        grid[i, j] = -1  
    end
    grid[env.target_state...] = 1  


    heatmap(1:env.env_size[1], 1:env.env_size[2], grid', c=:blues, aspect_ratio=1, clims=(-1, 1), legend=false)

 
    for i in 0.5:env.env_size[1]+0.5
        plot!([i, i], [0.5, env.env_size[2]+0.5], color=:black, lw=1, label=false)
    end
    for j in 0.5:env.env_size[2]+0.5
        plot!([0.5, env.env_size[1]+0.5], [j, j], color=:black, lw=1, label=false)
    end


    x_traj = [p[1] for p in env.traj]
    y_traj = [p[2] for p in env.traj]
    scatter!(x_traj, y_traj, marker=:circle, color=:red, markersize=5, label=false)


    scatter!([env.start_state[1]], [env.start_state[2]], marker=:star5, color=:blue, markersize=10, label=false)
    scatter!([env.target_state[1]], [env.target_state[2]], marker=:rect, color=:green, markersize=10, label=false)
end


function render_grid_policy(env::GridWorld, policy::Dict{Tuple{Int, Int}, Tuple{Int, Int}})
    grid = fill(0, env.env_size)  

    for (i, j) in env.forbidden_states
        grid[i, j] = -1  
    end
    grid[env.target_state...] = 1  

    heatmap(1:env.env_size[1], 1:env.env_size[2], grid', c=:blues, aspect_ratio=1, clims=(-1, 1), legend=false)

    for i in 0.5:env.env_size[1]+0.5
        plot!([i, i], [0.5, env.env_size[2]+0.5], color=:black, lw=1, label=false)
    end
    for j in 0.5:env.env_size[2]+0.5
        plot!([0.5, env.env_size[1]+0.5], [j, j], color=:black, lw=1, label=false)
    end

    scatter!([env.start_state[1]], [env.start_state[2]], marker=:star5, color=:yellow, markersize=10, label=false)
    scatter!([env.target_state[1]], [env.target_state[2]], marker=:rect, color=:green, markersize=10, label=false)

    arrow_scale = 0.3  
    x_vals = []
    y_vals = []
    dx_vals = []
    dy_vals = []

    for (state, action) in policy
        if state != env.target_state && state ∉ env.forbidden_states
            push!(x_vals, state[1])
            push!(y_vals, state[2])
            push!(dx_vals, action[1] * arrow_scale)
            push!(dy_vals, action[2] * arrow_scale)
        end
    end

    quiver!(x_vals, y_vals, quiver=(dx_vals, dy_vals), color=:black, label=false)
end

function render_grid_policy_value(env::GridWorld, policy::Dict{Tuple{Int, Int}, Tuple{Int, Int}}, V::Dict{Tuple{Int, Int}, Float64})
    grid = fill(0, env.env_size)  

    for (i, j) in env.forbidden_states
        grid[i, j] = -1  
    end
    grid[env.target_state...] = 1  

    p = heatmap(1:env.env_size[1], 1:env.env_size[2], grid', c=:blues, aspect_ratio=1, clims=(-1, 1), legend=false)

    for i in 0.5:env.env_size[1]+0.5
        plot!(p, [i, i], [0.5, env.env_size[2]+0.5], color=:black, lw=1, label=false)
    end
    for j in 0.5:env.env_size[2]+0.5
        plot!(p, [0.5, env.env_size[1]+0.5], [j, j], color=:black, lw=1, label=false)
    end

    scatter!(p, [env.start_state[1]], [env.start_state[2]], marker=:star5, color=:yellow, markersize=10, label=false)
    scatter!(p, [env.target_state[1]], [env.target_state[2]], marker=:rect, color=:green, markersize=10, label=false)

    arrow_scale = 0.4  
    x_vals = []
    y_vals = []
    dx_vals = []
    dy_vals = []

    for (state, action) in policy
        if state != env.target_state && state ∉ env.forbidden_states
            push!(x_vals, state[1])
            push!(y_vals, state[2])
            push!(dx_vals, action[1] * arrow_scale)
            push!(dy_vals, action[2] * arrow_scale)
        end
    end

    quiver!(p, x_vals, y_vals, quiver=(dx_vals, dy_vals), color=:black, label=false)

    
    for x in 1:env.env_size[1], y in 1:env.env_size[2]
        state = (x, y)
        if state ∉ env.forbidden_states
            value = get(V, state, 0.0)  
            annotate!(p, x, y, text(round(value, digits=2), :white, :center, 7))
        end
    end

    display(p)  
end


function random_policy(env::GridWorld)
    policy = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    for x in 1:env.env_size[1], y in 1:env.env_size[2]
        state = (x, y)
        if state != env.target_state && state ∉ env.forbidden_states
            policy[state] = rand(env.action_space)  
        end
    end
    return policy
end

# policy = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
# V = Dict{Tuple{Int, Int}, Float64}()

# for x in 1:5, y in 1:5
#     state = (x, y)
#     if state != (5,5) && state ∉ [(3,2), (4,3), (2,4), (4,5)]
#         policy[state] = rand([(0,1), (1,0), (0,-1), (-1,0)])
#         V[state] = rand() * 10  # Giả sử V có giá trị từ 0 đến 10
#     end
# end

# env = GridWorld()
# render_grid_policy_value(env, policy, V)

# env = GridWorld()
# reset!(env)
# policy = random_policy(env)
# render_grid_policy(env, policy)


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