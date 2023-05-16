import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from Nbandit_Classies import EpsilonGreedy, SoftmaxBandit, UCB, ThompsonSampling
import warnings
warnings.simplefilter(action='ignore',  category=FutureWarning)


class BanditExperiment:
    def __init__(self, n_arms, bandit_probabilities, n, method="EpsilonGreedy"):
        self.n_arms = n_arms
        self.bandit_probabilities = bandit_probabilities
        self.n = n
        self.method = method

    def method_fun(self):
        if self.method == 'EpsilonGreedy':
            # эпсилон получает наилучший вариант для методов Эпсилон-Жадного и СофтМакс
            epsilon = self.examine_EG()[1][0]
            self.new_ans_= np.array(self.run_exp_EpsilonGreedy(epsilon))
            return self.new_ans_.T
        elif self.method == 'SoftMax':
            # эпсилон получает наилучший вариант для методов Эпсилон-Жадного и СофтМакс
            epsilon = self.examine_SF()[1][0]
            # _rewards получает данные от данной модели
            self.new_ans_ = np.array(self.run_exp_SoftMax(epsilon))
            return self.new_ans_.T
        elif self.method == 'UCB':
            # _rewards получает данные от данной модели
            self.new_ans_ = np.array(self.run_exp_UCB())
            return self.new_ans_.T
        elif self.method == 'ThompsonSampling':
            # _rewards получает данные от данной модели
            self.new_ans_ = np.array(self.run_exp_ThompsonSampling())
            return self.new_ans_.T

    def metric(self):
        if self.method == 'EpsilonGreedy':
            epsilon = self.examine_EG()[1][0]
            ep = EpsilonGreedy(self.n, self.n_arms, epsilon)
            caleg = ep.calculate_metrics_forEG(self.bandit_probabilities)
            return caleg
        elif self.method == 'SoftMax':
            temperature = self.examine_SF()[1][0]
            sf = SoftmaxBandit(self.n, self.n_arms, temperature)
            calsm = sf.calculate_metrics_forSoftMax(self.bandit_probabilities, temperature)
            return calsm
        elif self.method == 'UCB':
            ucb = UCB(self.n_arms, self.n)
            calucb = ucb.calculate_metrics_forUCB(self.bandit_probabilities)
            return calucb
        elif self.method == 'ThompsonSampling':
            TS = ThompsonSampling(self.n_arms, self.n)
            caltm = TS.calculate_metrics_forThompson(self.bandit_probabilities)
            return caltm

    def examine_EG(self):
        eps = np.arange(0, 1, 0.01)
        ans = []
        for i in eps:
            mean_rewards = np.array(self.run_exp_EpsilonGreedy(i)).T
            ans.append(mean_rewards[len(mean_rewards) - 1][2])
        ans = np.array(ans)
        maxx = np.array([np.argmax(ans), np.max(ans)])

        return ans, maxx

    def examine_SF(self):
        temps = np.arange(0.1, 5, 0.1)
        ans = []
        for i in temps:
            mean_rewards = np.array(self.run_exp_SoftMax(i)).T
            ans.append(mean_rewards[len(mean_rewards) - 1][2])
        ans = np.array(ans)
        maxx = np.array([temps[np.argmax(ans)], np.max(ans)])
        return ans, maxx

    def run_exp_EpsilonGreedy(self, epsilon):
        bandit_probabilities = self.bandit_probabilities
        n_arms = self.n_arms
        n = self.n
        bandit = EpsilonGreedy(n, n_arms, epsilon)
        rewards = np.zeros(n)
        arms = np.zeros(n, dtype=int)  # массив для хранения выбранных ручек

        for i in range(n):
            arm = bandit.choose_arms()
            bandit.update(arm, bandit_probabilities[arm][i])
            rewards[i] = bandit_probabilities[arm][i]
            arms[i] = arm
        cumulative_rewards = np.cumsum(rewards)
        mean_rewards = cumulative_rewards / (np.arange(n) + 1)
        return mean_rewards, arms, cumulative_rewards  # возвращаем также массив выбранных ручек

    def run_exp_SoftMax(self, temperature):
        bandit_probabilities = self.bandit_probabilities
        bandit = SoftmaxBandit(self.n, self.n_arms, temperature)
        rewards = np.zeros(self.n)
        arms = np.zeros(self.n, dtype=int)  # массив для хранения выбранных ручек
        for i in range(self.n):
            arm = bandit.select_arm()
            bandit.update(arm, bandit_probabilities[arm][i])
            rewards[i] = bandit_probabilities[arm][i]
            arms[i] = arm
        cumulative_rewards = np.cumsum(rewards)
        mean_rewards = cumulative_rewards / (np.arange(self.n) + 1)
        return mean_rewards, arms, cumulative_rewards  # возвращаем также массив выбранных ручек

    def run_exp_UCB(self):
        bandit = UCB(self.n_arms, self.n)
        rewards = np.zeros(self.n)
        arms = np.zeros(self.n, dtype=int)
        for i in range(1000):
            arm = bandit.select_arm()
            bandit.update(arm, self.bandit_probabilities[arm][i])
            rewards[i] = self.bandit_probabilities[arm][i]
            arms[i] = arm
        cumulative_rewards = np.cumsum(rewards)
        mean_rewards = cumulative_rewards / (np.arange(self.n) + 1)
        return mean_rewards, arms, cumulative_rewards


    def run_exp_ThompsonSampling(self):
        bandit = ThompsonSampling(self.n_arms, self.n)
        arms_function = lambda arm, i: max(0, min(1, self.bandit_probabilities[arm, i]))
        bandit.fit(arms_function)
        rewards = np.zeros(self.n)
        arms = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            arm = bandit.choose_arm()
            bandit.update(arm, self.bandit_probabilities[arm][i])
            rewards[i] = self.bandit_probabilities[arm][i]
            arms[i] = arm
        cumulative_rewards = np.cumsum(rewards)
        mean_rewards = cumulative_rewards / (np.arange(self.n) + 1)
        return mean_rewards, arms, cumulative_rewards

    def plot(self):
        new_ans_ = self.method_fun()
        bandit_probabilities = self.bandit_probabilities
        n_arms = self.n_arms

        sns.set_style("whitegrid")
        fig, axs = plt.subplots(6, figsize=(20, 20))

        df = pd.DataFrame(new_ans_, columns=['Reward', 'Arm', 'Cumulative_Reward'])
        # print(self.method, 'dfdf', df, '\n', rewards)
        axs[0] = sns.FacetGrid(data=df, height=16, aspect=0.6)
        axs[0].map(sns.lineplot, x=np.arange(df.shape[0]), y=df['Cumulative_Reward'])
        axs[0].set_axis_labels(x_var='Итерация', y_var='Суммируемый выигрыш')

        columns = [f"Arm {i + 1}" for i in range(n_arms)]
        bandit_probabilities = pd.DataFrame(bandit_probabilities.T, columns=columns)

        axs[1] = sns.FacetGrid(data=pd.melt(bandit_probabilities), col='variable', col_wrap=n_arms, height=4, aspect=1)
        axs[1].map(sns.kdeplot, 'value', shade=True)
        axs[1].set_axis_labels(x_var='Значение', y_var='Плотность')

        axs[2] = sns.FacetGrid(data=df, height=16, aspect=0.6)
        axs[2].map(sns.stripplot, x=np.arange(df.shape[0]), y=df['Arm'], jitter=True)
        axs[2].set_axis_labels(x_var='Итерация', y_var='Выбранная рука')

        axs[3] = sns.FacetGrid(data=df, height=16, aspect=0.6)
        axs[3].map(sns.lineplot, x=np.arange(df.shape[0]), y=df['Reward'])
        axs[3].set_axis_labels(x_var='Итерация', y_var='Выигрыш')

        axs[4] = sns.FacetGrid(data=bandit_probabilities, height=16, aspect=0.6)
        axs[4].map(sns.lineplot, data=bandit_probabilities, linewidth=0.9, alpha=0.8)
        axs[4].set_axis_labels(x_var='Итерация', y_var='Вероятности рук')

        df = pd.DataFrame(bandit_probabilities).melt(var_name='actions', value_name='reward')
        sns.boxenplot(x='actions', y='reward', data=df, ax=axs[5], color='#23a98c')
        [axs[5].spines[pos].set_visible(False) for pos in ('right', 'bottom', 'top')]
        [mt.set_color('#0c6575') for mt in axs[5].get_xmajorticklabels()]
        [tl.set_color('none') for tl in axs[5].get_xticklines()]
        axs[5].set(ylim=(np.min(bandit_probabilities), np.max(bandit_probabilities)))
        axs[5].set_title('Boxenplot')

        self.axes = axs
        plt.close()
    def save_plots(self, file):
            with PdfPages(file) as pdf:
                for ax in self.axes:
                    pdf.savefig(ax.figure)

class Tab():
    def __init__(self, bandit_probabilities, methods):
        self.n_arms = bandit_probabilities.shape[0]
        self.n = bandit_probabilities.shape[1]
        self.bandit_probabilities = bandit_probabilities
        self.methods = methods

    def get_metrics_table(self):
        n = self.n
        metric_data = {'Method': [], 'Mean Reward': [], 'Total Reward': [], 'Optimal Action Loss': []}

        for method in self.methods:
            bandit = BanditExperiment(self.n_arms, self.bandit_probabilities, self.n, method)
            mean_reward, total_reward, loss = bandit.metric()
            metric_data['Method'].append(method)
            metric_data['Mean Reward'].append(mean_reward)
            metric_data['Total Reward'].append(total_reward)
            metric_data['Optimal Action Loss'].append(loss)
        return pd.DataFrame(metric_data)
