import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import warnings
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
warnings.filterwarnings('ignore')

# 设置字体
def setup_english_fonts():
    import matplotlib.font_manager as fm
    print("🔧 Setting Times New Roman font...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'x-large'
    print("✅ Font configured.")
    try:
        return fm.FontProperties(family='Times New Roman')
    except:
        return fm.FontProperties(family='serif')

FONT_PROP = setup_english_fonts()

# 选择与频谱图匹配的颜色映射，如 'viridis' ，可根据实际频谱图微调
# 若频谱图是类似从紫到黄渐变，'viridis' 较贴合，也可尝试 'plasma' 等
CUSTOM_CMAP = cm.viridis  

class FeatureArgs:
    def __init__(self):
        self.sample_rate = 16000
        self.desired_length = 8

class ICBHIClassVisualizer:
    def __init__(self, data_folder, sample_rate=16000, desired_length=8, n_mfcc=13, freq_range=None):
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.desired_length = desired_length
        self.n_mfcc = n_mfcc
        self.freq_range = freq_range if freq_range else (0, n_mfcc)
        self.font_prop = FONT_PROP
        self.args = FeatureArgs()
        self.output_base_dir = os.path.join(os.getcwd(), "split_audio_mfcc_spectrograms")
        os.makedirs(self.output_base_dir, exist_ok=True)
        print("🎯 MFCC Visualizer Initialized")
        print(f"   Folder: {data_folder}")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Duration: {desired_length} s")
        print(f"   MFCC Coefficients: {n_mfcc}")
        print(f"   Output Dir: {self.output_base_dir}")

    def load_audio_file(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            audio_tensor = torch.tensor(y, dtype=torch.float32)
            target_samples = self.sample_rate * self.desired_length
            if len(audio_tensor) > target_samples:
                audio_tensor = audio_tensor[:target_samples]
            else:
                pad_len = target_samples - len(audio_tensor)
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, int(pad_len)))
            return audio_tensor
        except Exception as e:
            print(f"❌ Failed to load {audio_path}: {e}")
            return None

    def generate_mfcc(self, audio_tensor):
        audio_np = audio_tensor.cpu().numpy()
        mfcc = librosa.feature.mfcc(y=audio_np, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        return mfcc.T

    def collect_audio_files(self):
        files = []
        if not os.path.exists(self.data_folder):
            print(f"❌ Folder not found: {self.data_folder}")
            return files
        for file in os.listdir(self.data_folder):
            if file.endswith('.wav'):
                files.append((file, os.path.join(self.data_folder, file)))
        files.sort()
        print(f"🔍 Found {len(files)} audio files.")
        return files

    def generate_combined_spectrograms(self):
        print("\n📈 Generating MFCC spectrograms...")
        files = self.collect_audio_files()
        if not files:
            print("❌ No audio files to process.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.flatten()

        # 先收集所有 MFCC 数据，计算整体的 vmin 和 vmax ，让颜色跨度更合理
        all_mfcc_data = []
        for fname, path in files[:4]:
            audio = self.load_audio_file(path)
            if audio is not None:
                mfcc_feat = self.generate_mfcc(audio)
                all_mfcc_data.append(mfcc_feat)

        if all_mfcc_data:
            all_mfcc_np = np.concatenate(all_mfcc_data, axis=0)
            vmin = np.percentile(all_mfcc_np, 10)  # 取 10% 分位数，缩小下限
            vmax = np.percentile(all_mfcc_np, 90)  # 取 90% 分位数，缩小上限
        else:
            vmin = -100
            vmax = 100

        for idx, (fname, path) in enumerate(files[:4]):
            print(f"📁 Processing: {fname}")
            audio = self.load_audio_file(path)
            if audio is None:
                continue

            try:
                ax = axes[idx]
                mfcc_feat = self.generate_mfcc(audio)
                mfcc_disp = mfcc_feat[:, self.freq_range[0]:self.freq_range[1]]

                im = ax.imshow(mfcc_disp.T, aspect='auto', origin='lower',
                               cmap=CUSTOM_CMAP, vmin=vmin, vmax=vmax)

                ax.set_xlabel('Time (s)', fontproperties=self.font_prop)
                ax.set_ylabel('MFCC Index', fontproperties=self.font_prop, labelpad=-7)

                y_range = self.freq_range[1] - self.freq_range[0]
                y_ticks = list(range(0, y_range, 2))
                y_labels = [f'MFCC-{self.freq_range[0] + i + 1}' for i in y_ticks]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels, fontproperties=self.font_prop)

                x_len = mfcc_disp.shape[0]
                total_time = self.desired_length
                x_ticks = [int(t * x_len / total_time) for t in range(0, total_time + 1)]
                x_labels = [str(t) for t in range(0, total_time + 1)]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, fontproperties=self.font_prop)

                ax.set_title(f"MFCC - {os.path.splitext(fname)[0]}", fontsize=24, fontweight='bold', pad=20)

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Coefficient Amplitude (dB)', fontsize=12, fontproperties=self.font_prop, labelpad=5)
                cbar.set_ticks(np.linspace(vmin, vmax, 5))
                cbar.set_ticklabels([f'{x:.1f}' for x in np.linspace(vmin, vmax, 5)])
                for label in cbar.ax.get_yticklabels():
                    label.set_fontproperties(self.font_prop)

                print(f"   ✅ Done: {fname}")
            except Exception as e:
                print(f"   ❌ Error processing {fname}: {e}")
                continue

        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.3, hspace=0.4)
        save_path = os.path.join(self.output_base_dir, "combined_mfcc_spectrograms.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"\n✅ Saved: {save_path}")

    def run_visualization(self):
        print("🚀 Starting MFCC visualization...")
        self.generate_combined_spectrograms()
        print("🎉 All done.")

def main():
    data_folder = r"D:\bishe\MVST-main\basevisualization\basefeature\basefbank\split\split"
    sample_rate = 16000
    desired_length = 8
    n_mfcc = 13
    freq_range = (0, 13)

    visualizer = ICBHIClassVisualizer(
        data_folder=data_folder,
        sample_rate=sample_rate,
        desired_length=desired_length,
        n_mfcc=n_mfcc,
        freq_range=freq_range
    )

    visualizer.run_visualization()

if __name__ == "__main__":
    main()