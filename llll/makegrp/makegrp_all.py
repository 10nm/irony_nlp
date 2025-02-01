import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def create_bubble_scatter_plot(data):
    plt.figure(figsize=(10, 6))

    plt.scatter(data['learning_rate'], data['f1'], s=(((data['epoch']*10)**2)/3), alpha=0.5, edgecolors='w', linewidth=0.7)
    plt.xscale('log')
    plt.ylim(0, 0.75)
    print(len(data))
    plt.title(f'学習率とF1スコアの関係 n={len(data)}')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.text(1E-5, 0.5, f'バブルサイズはエポック数', fontsize=13, color='black', weight='bold')
    plt.xlabel('Learning rate')
    plt.ylabel('F1 score')
    plt.grid(True)
    plt.savefig('./plot/bubble_scatter_plot_all.png')


if __name__ == "__main__":
    data = load_data('./extracted_data.csv')
    create_bubble_scatter_plot(data)
