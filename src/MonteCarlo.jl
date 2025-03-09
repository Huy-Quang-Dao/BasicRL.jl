using Random

function basic_MC(env::GridWorld, gamma::Float64=0.9, theta::Float64=1e-10, max_iter::Int=1000, epochs::Int=100)
    """
    Basic Monte-Carlo for solving the Bellman Optimality Equation (BOE)
    :param env: instance of the environment
    :param gamma: discount factor
    :param theta: threshold for convergence
    :param max_iter: number of iterations
    :param epochs: length of episodes
    :return: value function and policy
    """
    V = Dict{Tuple{Int, Int}, Float64}()
    policy = Dict((x, y) => (0, 0) for x in 1:env.env_size[1], y in 1:env.env_size[2])
    # policy = Dict((x, y) => rand(env.action_space) for x in 1:env.env_size[1], y in 1:env.env_size[2])
    
    # Initialize value function
    for x in 1:env.env_size[1], y in 1:env.env_size[2]
        state = (x, y)
        if state in env.forbidden_states
            V[state] = env.reward_forbidden
        elseif state == env.target_state
            V[state] = env.reward_target
        else
            V[state] = 0.0
        end
    end
    
    for j in 1:max_iter
        delta = 0.0
        
        for x in 1:env.env_size[1], y in 1:env.env_size[2]
            state = (x, y)
            if state == env.target_state || state in env.forbidden_states
                continue
            end
            
            q_values = Dict{Tuple{Int, Int}, Float64}()
            
            for action in env.action_space
                # policy evaluation
                total_return = 0.0
                state_temp = state
                action_temp = action
                discount = 1.0
                
                for _ in 1:epochs
                    next_state, reward = get_next_state_and_reward(env, state_temp, action_temp)
                    total_return += discount * reward
                    discount *= gamma
                    state_temp = next_state
                    action_temp = policy[next_state]
                end
                
                q_values[action] = total_return
            end
            
            # Policy improvement
            best_action = argmax(q_values)
            delta = max(delta, abs(V[state] - q_values[best_action]))
            V[state] = q_values[best_action]
            policy[state] = best_action
        end
        
        println("Iteration $j, delta: $delta")
        if delta < theta
            break
        end
    end
    
    return V, policy
end



env = GridWorld()
reset!(env)
V, policy = basic_MC(env)
render_grid_policy_value(env, policy, V)









