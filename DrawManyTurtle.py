import turtle as tl
import ast
import glob
import re
import numpy as np

'''
多路径高精度版:
    1.自动查找当前目录下 datas0.txt, datas1.txt, ... 并依次绘制
    2.每条路径之间自动提笔(正确处理子路径跳点,不再画出多余连接线)
    3.numpy 分块计算全部系数,速度快且不损失画质
    4.不再硬编码 N,任意数量的傅里叶系数自动适配
'''

SAMPLES = [20000, 5000, 5000, 5000]  # 各路径描点数(按文件编号顺序);文件多于列表时,后面的路径用 DEFAULT_SAMPLES
DEFAULT_SAMPLES = 5000

def load_coeffs(path):
    coeffs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                coeffs.append(complex(*ast.literal_eval(line)))

    def freq(i):  # 下标 i -> 频率 m: 0, +1, -1, +2, -2, ...
        return 0 if i == 0 else ((i + 1) // 2) * (1 if i % 2 else -1)

    return [(freq(i), c) for i, c in enumerate(coeffs)]

def compute_points(terms, samples, chunk=1000):
    """分块 numpy 计算:系数再多也不会撑爆内存"""
    m = np.array([t[0] for t in terms], dtype=float)
    c = np.array([t[1] for t in terms], dtype=complex)
    pts = []
    for s in range(0, samples, chunk):
        theta = np.arange(s, min(s + chunk, samples)) / samples * 2 * np.pi
        Z = (c[:, None] * np.exp(1j * m[:, None] * theta[None, :])).sum(axis=0)
        pts.extend(zip(Z.real.tolist(), (-Z.imag).tolist()))
    return pts

def main():
    # 按数字顺序找到 datas0.txt, datas1.txt, ...(不会误匹配单路径版的 datas.txt)
    files = sorted(glob.glob("datas[0-9]*.txt"),
                   key=lambda p: int(re.search(r"\d+", p).group()))
    if not files:
        print("没有找到 datas0.txt, datas1.txt, ...")
        return

    tl.setup(960, 720)
    tl.tracer(0, 0)          # 关闭逐帧动画,统一手动刷新
    tl.hideturtle()
    tl.pensize(1)            # 高分辨率下用细笔

    for num, path in enumerate(files):
        samples = SAMPLES[num] if num < len(SAMPLES) else DEFAULT_SAMPLES
        terms = load_coeffs(path)
        pts = compute_points(terms, samples)

        tl.penup()           # 每条新路径先提笔,跳点处不连线
        tl.goto(pts[0])
        tl.pendown()
        for i, p in enumerate(pts):
            tl.goto(p)
            if i % 200 == 0:
                tl.update()
        print(f"{path} 绘制完成({len(terms)} 个系数, {samples} 个描点)")

    tl.update()
    tl.done()

if __name__ == "__main__":
    main()
