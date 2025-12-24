# %%
import time
import multiprocessing as mp
import os
import imageio
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

# %%
# 函数定义


def read_file(path: Path):
    df = pd.read_csv(path)
    df = df.iloc[1:]
    df.columns = ['x', 'y', 'v']
    for c in df.columns:
        df[c] = df[c].map(float)
    df['t'] = float(path.stem.split('-')[1])
    return df


def draw_frame(t_val, df_t, i, v_min, v_max, X, Y, ratio, temp_dir):
    width = 5  # inches
    fig, ax = plt.subplots(figsize=(width, width/ratio))

    # 插值
    points = df_t[['x', 'y']].values
    values = df_t['v'].values
    Z = griddata(points, values, (X, Y), method='cubic')

    cmap = 'Reds'

    # 绘制等高线
    contour = ax.contourf(
        X, Y, Z,
        cmap=cmap,
        vmin=v_min,
        vmax=v_max)

    # 绘制数据点
    ax.scatter(
        df_t['x'], df_t['y'],
        c=df_t['v'],
        s=2,
        alpha=0.2,
        linewidth=0.1)

    ax.set_xlabel('y', fontsize=12)
    ax.set_ylabel('z', fontsize=12)
    ax.set_title(f'Contour Plot - t = {t_val}',
                 fontsize=14, fontweight='bold')

    fig.colorbar(contour, label='v')
    fig.tight_layout()

    # 保存
    img_path = os.path.join(temp_dir, f'frame_{i:03d}.png')
    fig.savefig(img_path)
    plt.close(fig)
    return img_path


# %%
if __name__ == '__main__':
    # Constants
    GIF_PATH = Path.cwd().joinpath('generated.gif')

    # 只在主进程中读取一次数据
    print("Reading data in main process...")
    dfs = [read_file(e)
           for e in tqdm(Path('./output').iterdir(), 'Read txt files')]
    df = pd.concat(dfs)
    df = df[df['v'] != 0]
    print(f"Data loaded: {len(df)} rows")

    # 预处理数据
    times = np.array(sorted(df['t'].unique()))
    print(f"Time points: {len(times)}")

    # 将数据按时间分割，这样每个子进程只需处理自己的部分
    df_by_time = {t: df[df['t'] == t]
                  for t in tqdm(times, 'Preparing data by time')}

    x_range = (df['x'].min(), df['x'].max())
    y_range = (df['y'].min(), df['y'].max())
    ratio = (x_range[1] - x_range[0]) / (y_range[1] - y_range[0])

    v_min, v_max = df['v'].min(), df['v'].max()
    v_min, v_max = 0.1, 1.0  # 或者使用实际值

    # 创建插值网格
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    temp_dir = 'img'
    os.makedirs(temp_dir, exist_ok=True)

    # 并行处理
    num_processes = min(mp.cpu_count(), len(times))

    tic = time.time()
    print(
        f"Processing {len(times)} time points using {num_processes} processes...")

    results = []

    with mp.Pool(processes=num_processes) as pool:
        # 准备参数：每个时间点的数据和对应索引
        args = [(t_val, df_by_time[t_val], i, v_min, v_max, X, Y, ratio, temp_dir)
                for i, t_val in enumerate(times)]

        # 使用 imap 或 imap_unordered 保持顺序
        for result in tqdm(pool.starmap(draw_frame, args),
                           total=len(times),
                           desc='Processing frames'):
            results.append(result)

    passed = time.time() - tic
    print(f"Processing complete ({passed:.4f} seconds)!")

    # 创建GIF
    images = []
    for img_path in sorted(results):
        images.append(imageio.v2.imread(img_path))
    imageio.mimsave(GIF_PATH, images, duration=0.5)

    print(f"GIF已创建: {GIF_PATH}")
