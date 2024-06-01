import numpy as np

import src.utils.epsilon_basic_functions as EpsilonBasicFunctions


# import src.environments.FrozenLakeEnv.frozen_lake_env as FrozenLakeEnv
# import src.environments.TaxiEnv.taxi_env as TaxiEnv
# import src.environments.WumpusEnv.wumpus_env as WumpusEnv
class DefaultQLearning:

    def __init__(self, environment_generator, exploration_ratio_func):
        self.environment_generator = environment_generator
        self.env = self.environment_generator()
        self.exploration_ratio_func = exploration_ratio_func
        self.list_of_all_states_visited_during_training = []

    def state_visited_during_training(self, state):
        if state not in self.list_of_all_states_visited_during_training:
            self.list_of_all_states_visited_during_training.append(state)

    def add_q(self, Q, state):
        if state not in Q:
            Q[state] = [0 for _ in range(self.env.get_num_of_actions())]
        return Q

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

        Q = {}

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

            state = self.env.reset()
            done = False
            truncated = False
            num_of_steps = 0

            # TRAINING PART
            while not done and not truncated and num_of_steps < max_num_of_steps_per_episode:
                comparable_state = self.env.get_comparable_representation_of_state(state)
                self.add_q(Q, comparable_state)

                self.state_visited_during_training(comparable_state)

                action = self.epsilon_greedy(comparable_state, self.exploration_ratio_func(comparable_state, episode),
                                             Q)
                next_state, reward, done, truncated = self.env.make_step(action)
                num_of_environment_calls += 1

                if episode % 10 == 0 and render_train:
                    self.env.render()

                comparable_next_state = self.env.get_comparable_representation_of_state(next_state)
                self.add_q(Q, comparable_next_state)

                transition = (comparable_state, action, float(reward), comparable_next_state)
                Q = self.f(Q, [transition], learning_rate, discount_factor)

                state = next_state
                num_of_steps += 1

            # VALIDATION PART
            if (retrieve_stats and episode % retrieve_stats_every_x_steps == 0) or (max_num_of_episodes !=None and episode == max_num_of_episodes - 1):
                num_of_environment_calls_registered = num_of_environment_calls
                total_reward = 0
                for _ in range(validation_mean_over_x_episodes):
                    state = self.env.reset()
                    done = False
                    truncated = False
                    while not done and not truncated:
                        comparable_state = self.env.get_comparable_representation_of_state(state)
                        self.add_q(Q, comparable_state)

                        action = np.argmax(Q[comparable_state])
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
                        "mu_t_expl": self.exploration_ratio_func(None, episode)
                    }
                })

                if average_reward >= min_score and not achieved_min_score:
                    achieved_min_score = True
                    achieved_min_score_value = average_reward
                    achieved_min_score_with_num_of_environment_calls = num_of_environment_calls
                    achieved_min_score_with_num_of_episodes = episode

                    if retrieve_stats:
                        print(
                            f"Achieved min score ! Reward: {average_reward}, Num of Environment calls:"
                            f" {num_of_environment_calls}")

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
    import src.environments.TaxiEnv.taxi_env as fr

    environment_generator = lambda: fr.TaxiEnv()
    for i in range(10):
        algo = DefaultQLearning(environment_generator, EpsilonBasicFunctions.EpsilonBasicFuncs().constant_x(0.3))

        results = algo.run(0.1, 0.99, 1500, min_score=0.5, retrieve_stats=True, validation_mean_over_x_episodes=5)

    # pprint.pprint(results)
