import numpy as np


class EpsilonGreedy:
    def __init__(self, n, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.arm_values = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms)
        self.n = n
        self.rewards = np.zeros(n)
        self.cumulative_rewards = np.zeros(n)
        self.optimal_action_percentages = np.zeros(n)
        self.losses = np.zeros(n)

    def choose_arms(self):
        if np.random.uniform(0, 1) < self.epsilon:
            arm = np.random.randint(0, self.n_arms - 1)
        else:
            arm = np.argmax(self.arm_values)
        return arm

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        value = self.arm_values[arm]
        self.arm_values[arm] = ((n - 1) / n) * value + (1 / n) * reward

    def calculate_metrics_forEG(self, bandit_probabilities):
        optimal_action = np.argmax(np.mean(bandit_probabilities, axis=1))
        action_counts = np.zeros(self.n_arms)
        for i in range(self.n):
            arm = self.choose_arms()
            self.update(arm, bandit_probabilities[arm][i])
            self.rewards[i] = bandit_probabilities[arm][i]
            self.cumulative_rewards[i] = np.sum(self.rewards)
            self.optimal_action_percentages[i] = 100 * np.sum(action_counts[optimal_action]) / (i + 1)
        optimal_action_percent = 100 * np.sum(action_counts[optimal_action]) / self.n
        self.losses = np.abs(optimal_action_percent - self.rewards[i])
        return np.mean(self.rewards), self.cumulative_rewards[-1], np.sum(self.losses)

class SoftmaxBandit:
    def __init__(self, n, n_arms, temperature=1.0):
        self.n_arms = n_arms
        self.temperature = temperature
        self.Q = np.zeros(n_arms)
        # который содержит текущие оценки
        # реднего значения вознаграждения
        # для каждого действия.

        self.N = np.zeros(n_arms)  # который содержит количество раз,
        # когда каждое действие было выбрано
        # на каждом шаге.
        self.n = n
        self.p = np.ones(n_arms) / n_arms  # который содержит вероятности выбора
        # каждого действия на текущем шаге.

    def select_arm(self):
        if np.random.random() < 0.05:
            return np.random.randint(self.n_arms)
        z = np.exp(self.Q / self.temperature)
        self.p = z / np.sum(z)
        return np.random.choice(np.arange(self.n_arms), p=self.p)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]
        self.p = np.exp(self.Q / self.temperature)
        self.p /= np.sum(self.p)

    def calculate_metrics_forSoftMax(self, bandit_probabilities, temperature):
        self.temperature = temperature
        rewards = np.zeros(self.n)
        optimal_action = np.argmax(np.mean(bandit_probabilities, axis=1))
        action_counts = np.zeros(self.n_arms)
        for i in range(self.n):
            arm = self.select_arm()
            action_counts[arm] += 1
            self.update(arm, bandit_probabilities[arm][i])
            rewards[i] = bandit_probabilities[arm][i]
        cumulative_rewards = np.cumsum(rewards)
        mean_rewards = cumulative_rewards / (np.arange(self.n) + 1)
        optimal_action_percent = 100 * np.sum(action_counts[optimal_action]) / self.n
        loss = np.abs(optimal_action_percent - 100)
        return np.mean(mean_rewards), np.sum(rewards), loss


class UCB:
    def __init__(self, n_arms, n):
        self.n_arms = n_arms
        self.total_rewards = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)
        self.total_pulls = 0
        self.n = n
    def select_arm(self):
        max_upper_bound = 0
        max_arm = 0
        for arm in range(self.n_arms):
            if self.n_pulls[arm] == 0:
                return arm
            else:
                average_reward = self.total_rewards[arm] / self.n_pulls[arm]
                exploration = np.sqrt(2 * np.log(self.total_pulls) / self.n_pulls[arm])
                upper_bound = average_reward + 2 * exploration
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    max_arm = arm
        return max_arm

    def update(self, arm, reward):
        self.total_rewards[arm] += reward
        self.n_pulls[arm] += 1
        self.total_pulls += 1

    def calculate_metrics_forUCB(self, bandit_probabilities):
        rewards = np.zeros(self.n)
        optimal_action = np.argmax(np.mean(bandit_probabilities, axis=1))
        action_counts = np.zeros(self.n_arms)
        for i in range(self.n):
            arm = self.select_arm()
            action_counts[arm] += 1
            self.update(arm, bandit_probabilities[arm][i])
            rewards[i] = bandit_probabilities[arm][i]
        cumulative_rewards = np.cumsum(rewards)
        mean_rewards = cumulative_rewards / (np.arange(self.n) + 1)
        optimal_action_percent = 100 * np.sum(action_counts[optimal_action]) / self.n
        loss = np.abs(optimal_action_percent - 100)
        return np.mean(mean_rewards), np.sum(rewards), loss


class ThompsonSampling:
    def __init__(self, n_arms, n):
        self.n_arms = n_arms
        self.n = n
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.rewards = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)

    def choose_arm(self):
        samples = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if (self.beta[i] > 0) & (self.alpha[i] > 0):
                samples[i] = np.random.beta(self.alpha[i], self.beta[i])
            else:
                samples[i] = 0
        return np.argmax(samples)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    def fit(self, arms_function):
        for i in range(self.n):
            chosen_arm = self.choose_arm()
            reward = arms_function(chosen_arm, i)
            self.update(chosen_arm, reward)

    def calculate_metrics_forThompson(self, bandit_probabilities):
        rewards = np.zeros(self.n)
        optimal_action = np.argmax(np.mean(bandit_probabilities, axis=1))
        action_counts = np.zeros(self.n_arms)
        for i in range(self.n):
            arm = self.choose_arm()
            action_counts[arm] += 1
            self.update(arm, bandit_probabilities[arm][i])
            rewards[i] = bandit_probabilities[arm][i]
        cumulative_rewards = np.cumsum(rewards)
        mean_rewards = cumulative_rewards / (np.arange(self.n) + 1)
        optimal_action_percent = 100 * np.sum(action_counts[optimal_action]) / self.n
        loss = np.abs(optimal_action_percent - 100)
        return np.mean(mean_rewards), np.sum(rewards), loss
