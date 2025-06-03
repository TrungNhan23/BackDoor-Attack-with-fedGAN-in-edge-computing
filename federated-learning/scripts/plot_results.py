import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

# Đường dẫn tới thư mục chứa các file CSV
csv_dir = os.path.join(os.path.dirname(__file__), '../../output/csv')
# Lấy danh sách tất cả file CSV trong thư mục
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {csv_dir}")

print(f"Found {len(csv_files)} CSV files:")
for file in csv_files:
    print(f"- {os.path.basename(file)}")

# Đọc từng file CSV và plot
plt.figure(figsize=(10, 6))

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    filename = os.path.basename(csv_file)
    
    # Plot data từ file
    rounds = df['Rounds']
    asr = df['ASR']
    ca = df['CA']
    
    mean_asr = asr[5:].mean()
    mean_ca = ca.mean()
    
    plt.plot(rounds, asr, label=f'{filename} - ASR (mean: {mean_asr:.2f})')
    plt.plot(rounds, ca, label=f'{filename} - CA (mean: {mean_ca:.2f})')

plt.xlabel('Round')
plt.ylabel('Value')
plt.title('ASR and CA over Rounds')
plt.legend()
plt.grid(True)

save_path = os.path.join(os.path.dirname(__file__), '../../output/plots/asr_ca_plot.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()