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


function exploringStarts_MC(env::GridWorld, gamma::Float64=0.9, max_iter::Int=1000, epochs::Int=1000)
    """
    Exploring Start Monte-Carlo for solving the Bellman Optimality Equation (BOE)
    :param env: instance of the environment
    :param gamma: discount factor
    :param theta: threshold for convergence
    :param max_iter: number of iterations
    :param epochs: length of episodes
    :return: value function and policy
    """
    # num_actions = length(env.action_space)
    V = Dict{Tuple{Int, Int}, Float64}()
    # policy = Dict((x, y) => (0, 0) for x in 1:env.env_size[1], y in 1:env.env_size[2])
    policy = Dict((x, y) => rand(env.action_space) for x in 1:env.env_size[1], y in 1:env.env_size[2])
    Q = Dict(((x, y), a) => 0.0 for x in 1:env.env_size[1], y in 1:env.env_size[2], a in env.action_space)
    return_temp = Dict(((x, y), a) => 0.0 for x in 1:env.env_size[1], y in 1:env.env_size[2], a in env.action_space)
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
    
    for i in 1:max_iter 
        state_action_pairs = Tuple{Tuple{Int, Int}, Tuple{Int, Int}}[]
        rewards = Float64[]
        # pair_idx = (i - 1) % (env.num_states * num_actions) + 1
        # s_idx = (pair_idx - 1) ÷ env.num_states + 1
        # a_idx = (pair_idx - 1) % num_actions + 1
        # s = (s_idx % env.env_size[1] + 1, s_idx ÷ env.env_size[1] + 1)
        # a = env.action_space[a_idx]
        s = rand(collect(keys(V)))
        a = rand(env.action_space)
        push!(state_action_pairs, (s, a))
        next_state, reward = get_next_state_and_reward(env, s, a)
        push!(rewards, reward)

        for _ in 1:epochs
            s_temp = next_state
            action = policy[s_temp]
            push!(state_action_pairs, (s_temp, action))
            next_state, reward = get_next_state_and_reward(env, s_temp, action)
            push!(rewards, reward)
        end

        g = 0.0
        for w in length(state_action_pairs):-1:1
            g = gamma * g + rewards[w]
            if state_action_pairs[w] in state_action_pairs[1:w-1]
                continue
            else
                s, a = state_action_pairs[w]
                return_temp[s, a] = g
            end
        end

        for x in 1:env.env_size[1], y in 1:env.env_size[2]
            s = (x,y)
            for a in env.action_space
                Q[s, a] = return_temp[s, a]
            end
            best_action = argmax(a -> Q[s, a], env.action_space)
            policy[s] = best_action
            V[s] = maximum([Q[s, a] for a in env.action_space])
        end
    end
    
    return V, policy
end



env = GridWorld()
reset!(env)
# V, policy = basic_MC(env)
V, policy = exploringStarts_MC(env)
render_grid_policy_value(env, policy, V)









