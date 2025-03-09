function value_iteration(env::GridWorld, gamma::Float64=0.9, theta::Float64=1e-10)
    """
    Value iteration for solving the Bellman Optimality Equation (BOE)
    :param env: instance of the environment
    :param gamma: discount factor
    :param theta: threshold for convergence
    :return: value function and policy
    """
    V = Dict{Tuple{Int, Int}, Float64}()
    policy = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    
    # Initialize value function
    for x in 1:env.env_size[1], y in 1:env.env_size[2]
        state = (x, y)
        if state in env.forbidden_states
            V[state] = env.reward_forbidden
        elseif state == env.target_state
            V[state] = env.reward_target
        else
            V[state] = rand(env.reward_forbidden:env.reward_target)
        end
    end
    
    iter_count = 0
    while true
        iter_count += 1
        delta = 0
        
        for x in 1:env.env_size[1], y in 1:env.env_size[2]
            state = (x, y)
            if state == env.target_state || state in env.forbidden_states
                continue
            end
            
            v = V[state]
            q_values = Dict{Tuple{Int, Int}, Float64}()
            
            for action in env.action_space
                next_state, reward = get_next_state_and_reward(env, state, action)
                q_values[action] = reward + gamma * V[next_state]
            end
            
            best_action = argmax(q_values)
            V[state] = q_values[best_action]
            policy[state] = best_action
            
            delta = max(delta, abs(v - V[state]))
        end
        
        println("Iteration $iter_count, delta: $delta")
        if delta < theta
            break
        end
        if iter_count > 10000
            break
        end
    end
    return V, policy
end


# env = GridWorld()
# reset!(env)
# V, policy = value_iteration(env)
# # render_grid_policy(env, policy)
# render_grid_policy_value(env, policy, V)









