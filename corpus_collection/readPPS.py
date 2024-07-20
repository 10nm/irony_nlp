import pandas as pd
import matplotlib.pyplot as plt
import datetime

csv_path = 'logs/csv/posts_per_sec.csv'

df = pd.read_csv(csv_path)

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

plt.figure(figsize=(10, 6))  # グラフサイズ
plt.plot(df.index, df['PostsPerSec'], label='Posts per Second')  
plt.title('Posts per Second Over Time')  
plt.xlabel('Time')  
plt.ylabel('Posts per Second')  
plt.legend()  
plt.xticks(rotation=45) 
plt.tight_layout()  
plt.show()

datenow = datetime.datetime.now()  # 現在時刻を取得
formatted_date = datenow.strftime("%Y-%m-%d_%H-%M-%S")  # 現在時刻をフォーマット
output_path = f'logs/pic/output_{formatted_date}.png'
plt.savefig(output_path)  # ファイル名に現在時刻を含める