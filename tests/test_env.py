from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
import unittest


class test_env(unittest.TestCase):

    def test_env(self,):
        # env = gym.make('CartPole-v1')
        config = (
            PPOConfig()
            .environment("Pendulum-v1")
        )
        config.env_runners(num_env_runners=2)
        config.training(
            lr=0.0002,
            train_batch_size_per_learner=2000,
            num_epochs=10,
        )
        # Build the Algorithm (PPO).
        ppo = config.build_algo()
        from pprint import pprint

        for _ in range(4):
            pprint(ppo.train())
        checkpoint_path = ppo.save_to_path()
        config.evaluation(
            # Run one evaluation round every iteration.
            evaluation_interval=1,

            # Create 2 eval EnvRunners in the extra EnvRunnerGroup.
            evaluation_num_env_runners=2,

            # Run evaluation for exactly 10 episodes. Note that because you have
            # 2 EnvRunners, each one runs through 5 episodes.
            evaluation_duration_unit="episodes",
            evaluation_duration=10,
        )

        # Rebuild the PPO, but with the extra evaluation EnvRunnerGroup
        ppo_with_evaluation = config.build_algo()

        for _ in range(3):
            pprint(ppo_with_evaluation.train())



        # config = (
        #     PPOConfig()
        #     .environment("Pendulum-v1")
        #     # Specify a simple tune hyperparameter sweep.
        #     .training(
        #         lr=tune.grid_search([0.001, 0.0005, 0.0001]),
        #     )
        # )

        # # Create a Tuner instance to manage the trials.
        # tuner = tune.Tuner(
        #     config.algo_class,
        #     param_space=config,
        #     # Specify a stopping criterion. Note that the criterion has to match one of the
        #     # pretty printed result metrics from the results returned previously by
        #     # ``.train()``. Also note that -1100 is not a good episode return for
        #     # Pendulum-v1, we are using it here to shorten the experiment time.
        #     run_config=train.RunConfig(
        #         stop={"env_runners/episode_return_mean": -1100.0},
        #     ),
        # )
        # # Run the Tuner and capture the results.
        # results = tuner.fit()

        # env = gym.make("LunarLander-v3", render_mode="human")
        # obs, info = env.reset(seed=42)
        # terminated, truncated, = False,False
        # while not terminated or not truncated:
        #     action = env.action_space.sample()
        #     next_obs, reward, terminated, truncated, info = env.step(action)
        #     if terminated or truncated:
        #         break

if __name__ == "__main__":
    unittest.main()