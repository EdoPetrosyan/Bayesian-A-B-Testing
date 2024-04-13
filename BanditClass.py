from abc import ABC, abstractmethod
from logs import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

logging.basicConfig()
logger = logging.getLogger("Multi-Armed Bandit Application")

# Adding a console handler with a more verbose log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

console_handler.setFormatter(CustomFormatter())

logger.addHandler(console_handler)


class Bandit(ABC):
    """
    A base class to model the behavior of bandit arms.
    
    Parameters:
    p (float): Actual probability of winning.

    Attributes:
    p (float): Actual probability of winning.
    p_estimate (float): Estimated probability of winning.
    N (int): Total number of trials.

    Methods:
    pull(): Simulate pulling the arm, returns a reward.
    update(): Update the estimate of the win rate after each trial.
    experiment(): Conducts the multi-arm bandit experiment.
    report(): Provides a summary of the experiment outcomes.
    """
    
    def __init__(self, p):
        """
        Constructs an arm for the bandit with a specific win rate.
        
        Parameters:
        p (float): Win rate of the arm.
        """
        self.p = p
        self.p_estimate = 0  # Initial estimate of win rate
        self.N = 0  # Initial number of trials
        self.r_estimate = 0  # Initial estimate of regret

    def __repr__(self):
        """
        Returns a string representation of the bandit arm.

        Returns:
        str: Description of the bandit arm.
        """
        return f'Bandit Arm with a Win Rate of {self.p}'

    @abstractmethod
    def pull(self):
        """Abstract method to pull the bandit arm."""
        pass

    @abstractmethod
    def update(self):
        """Abstract method to update estimates based on the observed reward."""
        pass

    @abstractmethod
    def experiment(self):
        """Abstract method to run the bandit experiment."""
        pass

    def report(self, N, bandits, chosen_bandit, reward, cumulative_regret, count_suboptimal=None, algorithm="Epsilon Greedy"):
        """
        Outputs the experiment results and saves them to CSV files.

        Parameters:
        N (int): Number of experiments.
        bandits (List[Bandit]): List of bandit instances.
        chosen_bandit (np.array): Indices of chosen bandits.
        reward (np.array): Recorded rewards.
        cumulative_regret (np.array): Cumulative regret data.
        count_suboptimal (int, optional): Number of times a non-optimal bandit was chosen.
        algorithm (str): Algorithm used in the experiment.

        Saves data to CSV and prints the summary statistics.
        """
        # Data for ongoing experiment
        ongoing_data = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })
        ongoing_data.to_csv(f'{algorithm}_Experiment.csv', index=False)

        # Data for the final results
        final_results = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })
        final_results.to_csv(f'{algorithm}_Final.csv', index=False)

        for b in range(len(bandits)):
            print(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward: {round(bandits[b].p_estimate, 4)} - Estimated average regret: {round(bandits[b].r_estimate, 4)}')
            print("--------------------------------------------------")
        
        print(f"Cumulative Reward: {sum(reward)}")
        print(f"Cumulative Regret: {cumulative_regret[-1]}")

        if count_suboptimal is not None:
            print(f"Percentage of suboptimal pulls: {round((float(count_suboptimal) / N) * 100, 2)}%")


class Visualization:
    def plot_cumulative_reward(self, num_trials, avg_reward, bandits, algo='EpsilonGreedy'):
        """
        Visualizes algorithm performance in terms of cumulative average reward.
        
        Args:
            num_trials (int): Number of trials in the experiment.
            avg_reward (np.array): Cumulative average reward.
            bandits (list[Bandits]): List of Bandit class objects.
            algo (str): Name of the algorithm used, defaults to 'EpsilonGreedy'.
            
        Prints:
            Linear and log scale plots of cumulative average reward and optimal reward.
        """
        plt.plot(avg_reward, label='Cumulative Average Reward')
        plt.plot(np.ones(num_trials) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Algorithm Performance - {algo} (Linear Scale)")
        plt.xlabel("# of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

        plt.plot(avg_reward, label='Cumulative Average Reward')
        plt.plot(np.ones(num_trials) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Algorithm Performance - {algo} (Log Scale)")
        plt.xlabel("# of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        plt.show()

    def compare_algorithms(self, cumulative_rewards_e, cumulative_rewards_t, cumulative_regret_e, cumulative_regret_t):
        """
        Compares Epsilon-Greedy and Thompson Sampling algorithms in terms of cumulative rewards and regrets.
        
        Args:
            cumulative_rewards_1 (np.array): Cumulative rewards for Epsilon-Greedy.
            cumulative_rewards_2 (np.array): Cumulative rewards for Thompson Sampling.
            cumulative_regret_1 (np.array): Cumulative regret for Epsilon-Greedy.
            cumulative_regret_2 (np.array): Cumulative regret for Thompson Sampling.
            
        Prints:
            Plots comparing cumulative rewards and regrets for both algorithms.
        """
        plt.plot(cumulative_rewards_e, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_t, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        plt.plot(cumulative_regret_e, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_t, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()


class EpsilonGreedy(Bandit):
    
    """
    Epsilon-Greedy multi-armed bandit algorithm.

    This class implements the Epsilon-Greedy algorithm for solving the multi-armed bandit problem.

    Parameters:
    p (float): The true win rate of the arm.

    Attributes:
    p (float): The true win rate of the arm.
    p_estimate (float): The estimated win rate.
    N (int): The number of pulls.

    Methods:
    pull(): Pulls the arm and returns the sampled reward.
    update(x): Updates the estimated win rate with a new reward value.
    run_experiment(true_win_rates, trial_count, time_step=1): Runs the Epsilon Greedy algorithm experiment.
    generate_report(N, results): Generates a report with statistics about the experiment.
    """

    def __init__(self, win_rate):
        
        """
        Initializes the EpsilonGreedy arm.

        Parameters:
        win_rate (float): The win rate of the arm.
        """
        super().__init__(win_rate)

    def pull(self):
        
        """
        Pulls the arm and returns the sampled reward.

        Returns:
        float: The sampled reward from the arm.
        """
        return np.random.randn() + self.p

    def update(self, reward):
        
        """
        Updates the estimated win rate with a new reward value.

        Parameters:
        reward (float): The observed reward.
        """
        self.N += 1.
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + 1.0/ self.N * reward
        self.regret_estimate = self.p - self.p_estimate

    def experiment(self, true_win_rates, trial_count, time_step=1):
        """
        Runs the experiment using the Epsilon Greedy Algorithm.
    
        Parameters:
        true_win_rates (list): List of true win rates for each arm.
        trial_count (int): The number of trials.
        time_step (int): Initial time step, defaults to 1.
    
        Returns:
        tuple: Average cumulative reward, cumulative reward, cumulative regret, updated bandits,
               chosen bandit at each trial, reward at each trial, count of suboptimal pulls
        """
        
        # Initialize bandits
        bandits = [EpsilonGreedy(rate) for rate in true_win_rates]
        rate_means = np.array(true_win_rates)
        optimal_bandit_index = np.argmax(rate_means)
        suboptimal_pull_count = 0
        epsilon = 1 / time_step
    
        # Arrays to track rewards and bandit choices
        rewards = np.empty(trial_count)
        bandit_choices = np.empty(trial_count)
    
        for i in range(trial_count):
            explore_probability = np.random.random()
            
            if explore_probability < epsilon:
                chosen_bandit_index = np.random.choice(len(bandits))
            else:
                chosen_bandit_index = np.argmax([bandit.p_estimate for bandit in bandits])
    
            reward_obtained = bandits[chosen_bandit_index].pull()
            bandits[chosen_bandit_index].update(reward_obtained)
    
            if chosen_bandit_index != optimal_bandit_index:
                suboptimal_pull_count += 1
            
            rewards[i] = reward_obtained
            bandit_choices[i] = chosen_bandit_index
            
            time_step += 1
            epsilon = 1 / time_step
    
        avg_cumulative_reward = np.cumsum(rewards) / (np.arange(trial_count) + 1)
        total_cumulative_reward = np.cumsum(rewards)
        
        cumulative_regrets = np.empty(trial_count)
        for i in range(trial_count):
            cumulative_regrets[i] = trial_count * max(rate_means) - total_cumulative_reward[i]
    
        return (avg_cumulative_reward, total_cumulative_reward, cumulative_regrets, bandits,
                bandit_choices, rewards, suboptimal_pull_count)



class ThompsonSampling(Bandit):
    """
    ThompsonSampling is a class for implementing the Thompson Sampling algorithm for multi-armed bandit problems.

    Attributes:
    - p (float): The win rate of the bandit arm.
    - lambda_ (float): A parameter for the Bayesian prior.
    - tau (float): A parameter for the Bayesian prior.
    - N (int): The number of times the bandit arm has been pulled.
    - p_estimate (float): The estimated win rate of the bandit arm.

    Methods:
    - pull(): Pull the bandit arm and return the observed reward.
    - sample(): Sample from the posterior distribution of the bandit arm's win rate.
    - update(x): Update the bandit arm's parameters and estimated win rate based on the observed reward.
    - plot(bandits, trial): Plot the probability distribution of the bandit arm's win rate after a given number of trials.
    - experiment(BANDIT_REWARDS, N): Run an experiment to estimate cumulative reward and regret for Thompson Sampling.

    """
    
    def __init__(self, p):
        """
        Initialize a ThompsonSampling bandit arm with the given win rate.

        Parameters:
        p (float): The win rate of the bandit arm.
        """
        super().__init__(p)
        self.lambda_ = 1
        self.tau = 1


    def pull(self):
        """
        Pull the bandit arm and return the observed reward.

        Returns:
        float: The observed reward from the bandit arm.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.p
    
    def sample(self):
        """
        Sample from the posterior distribution of the bandit arm's win rate.

        Returns:
        float: The sampled win rate from the posterior distribution.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate
    
    def update(self, x):
        """
        Update the bandit arm's parameters and estimated win rate based on the observed reward.

        Parameters:
        x (float): The observed reward.
        """
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate
        
    def plot(self, bandits, trial):
        
        """
        Plot the probability distribution of the bandit arm's win rate after a given number of trials.

        Parameters:
        bandits (list): List of ThompsonSampling bandit arms.
        trial (int): The number of trials or rounds.

        Displays a plot of the probability distribution of the bandit arm's win rate.

        """
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.p_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.p:.4f}, num plays: {b.N}")
            plt.title("Bandit distributions after {} trials".format(trial))
        plt.legend()
        plt.show()

    def experiment(self, true_win_rates, total_rounds):
        """
        Run an experiment to estimate cumulative reward and regret for Thompson Sampling.
    
        Parameters:
        true_win_rates (list): List of true win rates for each bandit arm.
        total_rounds (int): The number of rounds in the experiment.
    
        Returns:
        tuple: Cumulative reward statistics, updated bandits, and additional experiment data.
        """
        
        thompson_bandits = [ThompsonSampling(rate) for rate in true_win_rates]
    
        checkpoint_rounds = [5, 20, 50, 100, 200, 500, 1000, 1999, 5000, 10000, 19999]
        rewards = np.empty(total_rounds)
        bandit_selections = np.empty(total_rounds)
        
        for round_index in range(total_rounds):
            selected_bandit = np.argmax([bandit.sample() for bandit in thompson_bandits])
    
            if round_index in checkpoint_rounds:
                self.plot(thompson_bandits, round_index)
    
            reward_received = thompson_bandits[selected_bandit].pull()
            thompson_bandits[selected_bandit].update(reward_received)
    
            rewards[round_index] = reward_received
            bandit_selections[round_index] = selected_bandit
    
        cumulative_reward_average = np.cumsum(rewards) / (np.arange(total_rounds) + 1)
        total_cumulative_rewards = np.cumsum(rewards)
        
        total_regrets = np.empty(total_rounds)
        for index in range(total_rounds):
            total_regrets[index] = total_rounds * max(rate.p for rate in thompson_bandits) - total_cumulative_rewards[index]
    
        return cumulative_reward_average, total_cumulative_rewards, total_regrets, thompson_bandits, bandit_selections, rewards


def comparison(N, cumulative_reward_avg_e, cumulative_reward_avg_t, reward_e, reward_t, regret_e, regret_t, bandits):
    # think of a way to compare the performances of the two algorithms VISUALLY 
    
    """
    Compare performance of Epsilon Greedy and Thompson Sampling algorithms in terms of cumulative average reward.

    Parameters:
    N (int): The number of trials in the experiment.
    results_eg (tuple): A tuple of Epsilon Greedy experiment results.
    results_ts (tuple): A tuple of Thompson Sampling experiment results.
    cumulative_reward_avg_e (np.array): Cumulative average reward for EpsilonGreedy
    cumulative_reward_avg_t (np.array): Cumulative average reward for ThompsonSampling
    reward_e (np.array): Reward for EpsilonGreedy
    reward_t (np.array): Reward for ThompsonSampling
    regret_e (np.array): Regret for EpsilonGreedy
    regret_t (np.array): Regret for ThompsonSampling
    
    """

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cumulative_reward_avg_e, label='Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_avg_t, label='Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence - Linear Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")


    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward_avg_e, label='Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_avg_t, label='Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence - Log Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")
    plt.xscale("log")
    
    
    plt.tight_layout()
    plt.show()
    

    