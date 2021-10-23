"""Run the Q-network on the environment for fixed steps.

Complete the code marked TODO."""
import numpy as np # pylint: disable=unused-import
import torch # pylint: disable=unused-import


def run_episode(
    env,
    q_net, # pylint: disable=unused-argument
    steps_per_episode,
):
    """Runs the current policy on the given environment.

    Args:
        env (gym): environment to generate the state transition
        q_net (QNetwork): Q-Network used for computing the next action
        steps_per_episode (int): number of steps to run the policy for

    Returns:
        episode_experience (list): list containing the transitions
                        (state, action, reward, next_state, goal_state)
        episodic_return (float): reward collected during the episode
        succeeded (bool): DQN succeeded to reach the goal state or not
    """

    # list for recording what happened in the episode
    episode_experience = []
    succeeded = False
    episodic_return = 0.0

    # reset the environment to get the initial state
    state, goal_state = env.reset() # pylint: disable=unused-variable
    #state = state.astype(np.float32)
    #goal_state = goal_state.astype(np.float32)
    #print(type(state))
    #print(type(goal_state))
    #print(state.dtype) # torch.float32 for bit flip
    #print('????')

    for _ in range(steps_per_episode):

        # ======================== TODO modify code ========================
        #pass
        #print('br')
        #print(q_net)
        #print(goal_state)
        #print(type(goal_state))
        #print('before')

        #print(state.dtype) # state got change to float 64
        #state.clone().to(torch.float32)
        #goal_state.clone().to(torch.float32)
        state = state.astype(np.float32)
        goal_state = goal_state.astype(np.float32)


        # append goal state to input, and prepare for feeding to the q-network
        _input =  np.concatenate((state, goal_state), axis = 0)
        _input = torch.tensor(_input)
        #print(_input.dtype) # torch.float64 for draw sewer, torch.float32 for bit flip

        # forward pass to find action
        #q_net = q_net.double()
        action = q_net(_input)
        #print('after action')
        #print(action)
        # action
        action = torch.argmax(action, keepdim=True)
        #print(_output)
        #print(action)

        # take action, use env.step
        next_state, reward, done, info = env.step(action.item())
        #print(done)
        #print(len(next_state))
        #print(state)
        #print(reward)
        #print(done)
        #print(info)
        # state, reward, done, info

        # add transition to episode_experience as a tuple of
        # (state, action, reward, next_state, goal)
        eps_exp = (state, action, reward, next_state, goal_state)
        episode_experience.append(eps_exp)

        # update episodic return
        episodic_return += reward

        # update state
        state = next_state

        # update succeeded bool from the info returned by env.step
        succeeded = info['successful_this_state']

        # break the episode if done=True
        if done == True:
            #print('info', info)
            break

        # ========================      END TODO       ========================

    return episode_experience, episodic_return, succeeded
