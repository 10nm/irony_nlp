import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def create_bubble_scatter_plot(data):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(data['learning_rate'], data['f1'], s=(((data['epoch']*10)**2)/3), alpha=0.4, edgecolors='w')
    plt.xscale('log')
    plt.xlim(5e-8, 2e-6)
    plt.ylim(0, 0.75)
    print(len(data))
    plt.title(f'学習率とF1スコアの関係 n={len(data)}')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.text(7e-8, 0.1, 'バブルサイズは({エポック数}*10**2)/3\n軸 (5E-8 ~ 2E-6, 0 ~ 0.75)\n', fontsize=12, color='black', weight='bold')
    plt.xlabel('Learning rate')
    plt.ylabel('F1 score')
    plt.grid(True)
    plt.savefig('./plot/bubble_scatter_plot_lim.png')


if __name__ == "__main__":
    data = load_data('./extracted_data.csv')
    create_bubble_scatter_plot(data)
