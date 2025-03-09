using Random

function policy_iteration(env::GridWorld, gamma::Float64=0.9, theta::Float64=1e-10,max_iter = 1000, epochs=100, continuing=false)
    """
    Policy iteration for solving the Bellman Optimality Equation (BOE)
    :param env: instance of the environment
    :param gamma: discount factor
    :param theta: threshold for convergence
    :param epochs: number of iterations for policy evaluation
    :return: value function and policy
    """
    V = Dict{Tuple{Int, Int}, Float64}()
    policy = Dict((x, y) => rand(env.action_space) for x in 1:env.env_size[1], y in 1:env.env_size[2])
    
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
    for j in 1:max_iter
        # Policy Evaluation
        epoch = 0
        while true
            epoch += 1
            delta = 0
            for state in keys(V)
                if state == env.target_state || state in env.forbidden_states
                    continue
                end
                v_old = V[state]
                action = policy[state]
                next_state, reward = get_next_state_and_reward(env, state, action)
                V[state] = reward + gamma * V[next_state]
                delta = max(delta, abs(v_old - V[state]))
            end
            if delta < theta
                break
            end
            if epoch > epochs
                break
            end
        end

        # Policy Improvement
        policy_stable = true
        for state in keys(V)
            if state == env.target_state || state in env.forbidden_states
                continue
            end
            old_action = policy[state]
            action_values = Dict(a => begin
                next_state, reward = get_next_state_and_reward(env, state, a)
                reward + gamma * V[next_state]
            end for a in env.action_space)

            best_action = argmax(action_values)
            policy[state] = best_action

            if best_action != old_action
                policy_stable = false
            end
        end

        if policy_stable
            break
        end
    end


    return V, policy
end


env = GridWorld()
reset!(env)
V, policy = policy_iteration(env)
render_grid_policy_value(env, policy, V)









