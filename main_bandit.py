import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
# from Nbandit_Classies import EpsilonGreedy, SoftmaxBandit, UCB, ThompsonSampling
from model import BanditExperiment, Tab
from make_arms import generate_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1000, help="количество итераций ")
parser.add_argument("--n_arms", type=int, default=5, help="Количество рук")
parser.add_argument("--loc", type=float, default=0, help="среднее нормального распределения")
parser.add_argument("--scale", type=float, default=1, help="стандартное отклонение нормального распределения")
args = parser.parse_args()

bandit_probabilities = generate_data(args.n, args.n_arms, args.loc, args.scale)



# Create the my_plots directory if it doesn't exist
if not os.path.exists('my_plots'):
    os.makedirs('my_plots')

# for i in ['EpsilonGreedy', 'SoftMax', 'UCB', 'ThompsonSampling']:
#     ex = BanditExperiment(args.n_arms, bandit_probabilities, args.n, method=i)
#     if i == 'EpsilonGreedy':
#         ex.plot()
#         ex.save_plots(os.path.join('my_plots', 'EpsilonGreedy.pdf'))
#     elif i == 'SoftMax':
#         ex.plot()
#         ex.save_plots(os.path.join('my_plots', 'SoftMax.pdf'))
#     elif i == 'UCB':
#         ex.plot()
#         ex.save_plots(os.path.join('my_plots', 'UCB.pdf'))
#     else:
#         ex.plot()
#         ex.save_plots(os.path.join('my_plots', 'ThompsonSampling.pdf'))
#     print(i)


tab = Tab(bandit_probabilities, ['EpsilonGreedy', 'SoftMax', 'UCB', 'ThompsonSampling'])
ta =tab.get_metrics_table()

print(ta.shape, ta)

# Добавление пустой строки, если количество строк и столбцов не равно
if ta.shape[0] != ta.shape[1]:
    ta.loc[ta.shape[0]] = np.nan

# Создание тепловой карты
with PdfPages(os.path.join('my_plots', 'Table.pdf')) as pdf:
    fig, ax = plt.subplots(figsize=(10, 6))
    ta_num = ta.set_index('Method')   # Convert values to numbers
    heatmap = sns.heatmap(ta_num, fmt=".3f", annot=True, cmap='crest', ax=ax, annot_kws={"size": 12})
    heatmap.set(xlabel="", ylabel="")
    heatmap.xaxis.tick_top()
    ax.set_title('Heatmap')
    pdf.savefig(fig)
