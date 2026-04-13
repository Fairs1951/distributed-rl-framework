import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

class RLTrainer:
    def __init__(self, env_name="CartPole-v1"):
        self.env_name = env_name
        ray.init(ignore_reinit_error=True)

    def train_ppo(self, stop_iters=50):
        config = (
            PPOConfig()
            .environment(env=self.env_name)
            .framework("torch")
            .rollouts(num_rollout_workers=4)
            .training(
                lr=1e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                num_sgd_iter=10,
                sgd_minibatch_size=64,
                train_batch_size=4000
            )
        )
        
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=tune.RunConfig(stop={"training_iteration": stop_iters}),
        )
        results = tuner.fit()
        return results

    def shutdown(self):
        ray.shutdown()

if __name__ == "__main__":
    trainer = RLTrainer()
    # results = trainer.train_ppo()
