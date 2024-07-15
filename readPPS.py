import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('csv/posts_per_sec.csv')

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
plt.savefig('output.png') 