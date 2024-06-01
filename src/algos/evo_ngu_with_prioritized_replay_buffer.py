import math
import heapq

from src.algos.utils import *


# NGU
class EvoNGUWithPrioritizedReplayBuffer:

    def __init__(self, environment_generator,
                 beta=0.3,
                 N=1,
                 k=6,
                 epsilon=0.001,
                 c=0.001,
                 batch_size=500,
                 replay_buffer_capacity=10000,
                 replay_buffer_alpha=0.6,
                 replay_buffer_beta=0.4,
                 actor_epsilon_greedy_alpha=8):
        self.environment_generator = environment_generator
        self.env = self.environment_generator()
        self.beta = beta
        self.N = N
        self.k = k
        self.epsilon = epsilon
        self.c = c

        self.running_average_kth_nearest_distance_accumulator = 0
        self.running_average_kth_nearest_distance_count = 0

        self.list_of_all_states_visited_during_training = []

        self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_capacity, replay_buffer_alpha, replay_buffer_beta)

        self.batch_size = batch_size

        self.num_actions = self.env.get_num_of_actions()

        self.actor_epsilon_greedy_alpha = actor_epsilon_greedy_alpha

    def state_visited_during_training(self, state):
        if state not in self.list_of_all_states_visited_during_training:
            self.list_of_all_states_visited_during_training.append(state)

    def add_to_memory(self, M_memory, state):
        M_memory.append(state)

    def kernel_function(self, squared_distance, running_average):
        if squared_distance == running_average:
            return 1
        return self.epsilon / ((squared_distance / running_average) + self.epsilon)

    def get_intrinsic_reward(self, M_matrix, state):
        distances = [self.env.get_squared_euclidean_distance(comp_state, state) for comp_state in M_matrix]

        if len(distances) >= self.k:
            kth_nearest_distances = heapq.nsmallest(self.k, distances)
            kth_nearest_distance = kth_nearest_distances[-1]
        else:
            kth_nearest_distances = distances
            if len(kth_nearest_distances) == 0:
                kth_nearest_distance = 0
            else:
                kth_nearest_distance = max(kth_nearest_distances)

        if len(distances) != 0:
            self.running_average_kth_nearest_distance_accumulator += kth_nearest_distance
            self.running_average_kth_nearest_distance_count += 1
        running_average = self.running_average_kth_nearest_distance_accumulator / self.running_average_kth_nearest_distance_count

        sum_of_K = 0
        for dist in kth_nearest_distances:
            sum_of_K += self.kernel_function(dist, running_average)

        reward = 1 / math.sqrt(sum_of_K + self.c)
        return reward

    def get_priority_learner(self, current_score, goal_score):
        return abs(current_score - goal_score)

    def add_q(self, Q, state):
        if state not in Q:
            Q[state] = [0 for _ in range(self.env.get_num_of_actions())]
        return Q

    def get_priority_actor(self, actor, beta, GENERAL_EXTRINSIC_Q_TABLE, GENERAL_INTRINSIC_Q_TABLE,
                           Q_TABLES_FOR_EACH_AGENT,
                           state, action):

        self.add_q(Q_TABLES_FOR_EACH_AGENT[actor], state)
        self.add_q(GENERAL_EXTRINSIC_Q_TABLE, state)
        self.add_q(GENERAL_INTRINSIC_Q_TABLE, state)

        val_from_actor_q_table = Q_TABLES_FOR_EACH_AGENT[actor][state][action]

        val_from_general_q_table = GENERAL_EXTRINSIC_Q_TABLE[state][action] + beta * GENERAL_INTRINSIC_Q_TABLE[state][
            action]

        return abs(val_from_actor_q_table - val_from_general_q_table)

    def f(self, Q, D, learning_rate, discount_factor):
        for s, a, r, s_next in D:
            self.add_q(Q, s_next)
            self.add_q(Q, s)
            best_next_action = np.argmax(Q[s_next])
            td_target = r + discount_factor * Q[s_next][best_next_action]
            td_error = td_target - Q[s][a]
            Q[s][a] += learning_rate * td_error
        return Q

    def epsilon_greedy(self, state, expl, Q):
        if np.random.rand() < expl:
            return np.random.choice(self.env.get_num_of_actions())
        else:
            return np.argmax(Q[state])

    def run(self, learning_rate=0.1, discount_factor=0.95, max_num_of_episodes=None, max_num_of_environment_calls=None,
            max_num_of_steps_per_episode=1000, render_train=False, render_val=False, validation_mean_over_x_episodes=1,
            retrieve_stats=False, retrieve_stats_every_x_steps=10, stop_training_when_min_score_obtained=False,
            min_score=0.0):
        """
        Main function to initiate the training of the agent and evaluate its performance.

        Parameters:
        - learning_rate (float): The learning rate for Q-learning updates.
        - discount_factor (float): The discount factor for future rewards.
        - max_episodes (int): Maximum number of training episodes.
        - max_env_calls (int): Maximum number of environment interactions.
        - max_steps_per_episode (int): Maximum steps per episode.
        - render_train (bool): Render training process.
        - render_val (bool): Render validation process.
        - validation_episodes (int): Number of episodes for validation.
        - retrieve_stats (bool): Whether to retrieve and store training statistics.
        - stats_interval (int): Interval for retrieving statistics.
        - stop_at_min_score (bool): Stop training when minimum score is achieved.
        - min_score (float): Minimum score to stop training.

        Returns:
        - Dictionary containing training statistics and results.
        """

        num_of_environment_calls = 0
        average_reward = 0
        achieved_min_score = False
        achieved_min_score_with_num_of_environment_calls = 0
        achieved_min_score_with_num_of_episodes = 0
        achieved_min_score_value = 0
        all_scores_store = []

        betas_i = []

        if self.N > 2:
            betas_i.append(0)
            for i in range(1, self.N - 1):
                betas_i.append(self.beta * (1 / (1 + np.exp(-(10 * (2 * i - (self.N - 2)) / (self.N - 2))))))
            betas_i.append(self.beta)
        else:
            betas_i = [0, self.beta]

        # Environments
        envs = [self.environment_generator() for _ in range(self.N)]

        GENERAL_EXTRINSIC_Q_TABLE = {}
        GENERAL_INTRINSIC_Q_TABLE = {}

        num_actions = self.env.get_num_of_actions()

        mu_t_expl = 0.5
        sigma_expl = 0.05
        evo_N = 3
        evo_beta = 0.2

        total_reward_list = [0 for _ in range(evo_N)]

        epsilon_greedy_sample = sample_n(mu_t_expl, sigma_expl, evo_N)

        episode = 0
        num_of_environment_calls_registered = 0
        # TRAINING PART (include validation to compute stats and run Validation after last training step)
        while True:

            if max_num_of_episodes != None:
                if episode >= max_num_of_episodes:
                    break

            if max_num_of_environment_calls != None:
                if num_of_environment_calls_registered >= max_num_of_environment_calls:
                    break

            for evo_n in range(evo_N):

                # Update weights in actors
                Q_TABLES_FOR_EACH_AGENT = {n: {} for n in range(self.N)}
                for state, Q_table_states in GENERAL_EXTRINSIC_Q_TABLE.items():
                    for n in range(self.N):
                        val = np.array(Q_table_states) + betas_i[n] * np.array(GENERAL_INTRINSIC_Q_TABLE[state])
                        Q_TABLES_FOR_EACH_AGENT[n][state] = val.tolist()

                epsilon_greedy_epsilons_i = []
                if self.N > 1:
                    for i in range(self.N):
                        epsilon_greedy_epsilons_i.append(
                            math.pow(epsilon_greedy_sample[evo_n],
                                     1 + ((self.N -1 -i) / (self.N - 1)) * self.actor_epsilon_greedy_alpha))
                else:
                    epsilon_greedy_epsilons_i = [epsilon_greedy_sample[evo_n], epsilon_greedy_sample[evo_n]]

                # epsilon_greedy_epsilons_i = []
                # if self.N > 1:
                #     epsilon_greedy_epsilons_i = [epsilon_greedy_sample[evo_n] for _ in range(self.N)]
                # else:
                #     epsilon_greedy_epsilons_i = [epsilon_greedy_sample[evo_n], epsilon_greedy_sample[evo_n]]

                total_current_reward = 0
                total_num_of_steps = 0

                # Actors
                for n in range(self.N):

                    M_Memory = []

                    beta_i = betas_i[len(betas_i) - self.N + n]
                    env = envs[n]
                    state = self.env.get_comparable_representation_of_state(env.reset())
                    done = False
                    truncated = False

                    num_of_steps = 0

                    # TRAINING PART
                    while not done and not truncated and num_of_steps < max_num_of_steps_per_episode:

                        self.add_q(Q_TABLES_FOR_EACH_AGENT[n], state)

                        self.add_to_memory(M_Memory, state)
                        self.state_visited_during_training(state)

                        epsilon_greedy_epsilon = epsilon_greedy_epsilons_i[n]

                        action = self.epsilon_greedy(state, epsilon_greedy_epsilon, Q_TABLES_FOR_EACH_AGENT[n])

                        next_state, extrinsic_reward, done, truncated = env.make_step(action)
                        next_state = self.env.get_comparable_representation_of_state(next_state)
                        num_of_environment_calls += 1

                        if n == self.N - 1 and episode % 10 == 0 and render_train:
                            env.render()

                        self.add_q(Q_TABLES_FOR_EACH_AGENT[n], next_state)

                        intrinsic_reward = self.get_intrinsic_reward(M_Memory, next_state)

                        augmented_reward = extrinsic_reward + beta_i * intrinsic_reward

                        self.add_q(GENERAL_EXTRINSIC_Q_TABLE, state)
                        self.add_q(GENERAL_EXTRINSIC_Q_TABLE, next_state)
                        self.add_q(GENERAL_INTRINSIC_Q_TABLE, state)
                        self.add_q(GENERAL_INTRINSIC_Q_TABLE, next_state)

                        total_current_reward += get_loss(GENERAL_EXTRINSIC_Q_TABLE, state, next_state, action,
                                                         extrinsic_reward, discount_factor)

                        transition = (state, action, float(augmented_reward), next_state)
                        Q_TABLES_FOR_EACH_AGENT[n] = self.f(Q_TABLES_FOR_EACH_AGENT[n], [transition], learning_rate,
                                                            discount_factor)

                        # Adding to Prioritized Replay Buffer

                        priority = get_loss(GENERAL_EXTRINSIC_Q_TABLE, state, next_state, action,
                                                         extrinsic_reward, discount_factor)
                        self.replay_buffer.add(n, state, action, next_state, extrinsic_reward,
                                               intrinsic_reward, priority)

                        state = next_state
                        num_of_steps += 1
                    total_num_of_steps += num_of_steps

                total_reward_list[evo_n] += total_current_reward

            # Learning
            experiences, indices, weights = self.replay_buffer.sample(
                self.batch_size)
            for i in range(len(experiences)):  # for each experience
                actor, state, action, next_state, extrinsic_reward, intrinsic_reward = experiences[i]
                index = indices[i]
                weight = weights[i]

                self.add_q(GENERAL_EXTRINSIC_Q_TABLE, state)
                self.add_q(GENERAL_INTRINSIC_Q_TABLE, state)

                self.add_q(GENERAL_EXTRINSIC_Q_TABLE, next_state)
                self.add_q(GENERAL_INTRINSIC_Q_TABLE, next_state)

                transition = (state, action, float(extrinsic_reward), next_state)
                GENERAL_EXTRINSIC_Q_TABLE = self.f(GENERAL_EXTRINSIC_Q_TABLE, [transition], weight * learning_rate,
                                                   discount_factor)

                transition = (state, action, float(intrinsic_reward), next_state)
                GENERAL_INTRINSIC_Q_TABLE = self.f(GENERAL_INTRINSIC_Q_TABLE, [transition], weight * learning_rate,
                                                   discount_factor)

                # Update priority with new TD error
                self.replay_buffer.update_priorities([index], [
                    get_loss(GENERAL_EXTRINSIC_Q_TABLE, state, next_state, action,
                             extrinsic_reward, discount_factor)])

            # Update Epsilon Based on Evolutionary Algorithms
            if episode % 10 == 0:

                meta_obj = [0 for _ in range(evo_N)]

                for evo_n in range(evo_N):
                    if np.sum(total_reward_list) == 0:
                        meta_obj[evo_n] = 0
                    else:
                        meta_obj[evo_n] = total_reward_list[evo_n] / np.sum(total_reward_list)

                cumulative_learn = 0

                for i in range(evo_N):
                    cumulative_learn += meta_obj[i] * (epsilon_greedy_sample[i] - mu_t_expl)

                mu_t_expl = mu_t_expl + evo_beta * (1 / (sigma_expl * evo_N)) * cumulative_learn
                mu_t_expl = np.clip(mu_t_expl, 0, 1)

                total_reward_list = [0 for _ in range(evo_N)]
                epsilon_greedy_sample = sample_n(mu_t_expl, sigma_expl, evo_N)

            # VALIDATION PART
            if (retrieve_stats and episode % retrieve_stats_every_x_steps == 0) or (max_num_of_episodes !=None and episode == max_num_of_episodes - 1):
                total_reward = 0
                num_of_environment_calls_registered = num_of_environment_calls
                for _ in range(validation_mean_over_x_episodes):
                    state = self.env.reset()
                    done = False
                    truncated = False
                    while not done and not truncated:
                        comparable_state = self.env.get_comparable_representation_of_state(state)

                        if comparable_state not in GENERAL_EXTRINSIC_Q_TABLE:
                            GENERAL_EXTRINSIC_Q_TABLE[comparable_state] = [0 for _ in range(num_actions)]
                            GENERAL_INTRINSIC_Q_TABLE[comparable_state] = [0 for _ in range(num_actions)]

                        action = np.argmax(GENERAL_EXTRINSIC_Q_TABLE[comparable_state])
                        next_state, reward, done, truncated = self.env.make_step(action)

                        if render_val:
                            self.env.render()

                        total_reward += reward
                        state = next_state

                average_reward = total_reward / validation_mean_over_x_episodes

                all_scores_store.append({
                    "num_of_environment_calls": num_of_environment_calls,
                    "average_reward": average_reward,
                    "episode_num": episode,
                    "stats": {
                        "num_of_different_states_visited": len(self.list_of_all_states_visited_during_training),
                        "mu_t_expl": mu_t_expl
                    }
                })

                if average_reward >= min_score and not achieved_min_score:
                    achieved_min_score = True
                    achieved_min_score_value = average_reward
                    achieved_min_score_with_num_of_environment_calls = num_of_environment_calls
                    achieved_min_score_with_num_of_episodes = episode
                    if retrieve_stats:
                        print(
                            f"Achieved min score ! Reward: {average_reward}, Num of Environment calls: {num_of_environment_calls}")

                    if stop_training_when_min_score_obtained:
                        break
            episode += 1

        print(f"Average reward over {validation_mean_over_x_episodes} evaluation episodes: {average_reward}")
        print(f"number of different states : {len(self.list_of_all_states_visited_during_training)}")
        self.env.close()
        return {
            "all_scores_store": all_scores_store,
            "stats": {
                "num_of_different_states_visited": len(self.list_of_all_states_visited_during_training),
            },

            "achieved_min_score": {
                "achieved_min_score": achieved_min_score,
                "after_num_of_episodes": achieved_min_score_with_num_of_episodes,
                "after_num_of_environment_calls": achieved_min_score_with_num_of_environment_calls,
                "average_reward": achieved_min_score_value

            },
            "at_end": {
                "num_of_environment_calls": num_of_environment_calls,
                "average_reward": average_reward
            },

        }


if __name__ == '__main__':
    import pprint
    import src.environments.WumpusEnv.wumpus_env as env

    environment_generator = lambda: env.WumpusEnv()

    algo = EvoNGUWithPrioritizedReplayBuffer(environment_generator)

    results = algo.run(0.9, 0.99, 1000, min_score=0.5, retrieve_stats=True, validation_mean_over_x_episodes=1)

    pprint.pprint(results)
