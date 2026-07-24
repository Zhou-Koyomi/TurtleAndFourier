import turtle as tl
import ast
import numpy as np

def load_coeffs(path="datas.txt"):
    coeffs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                coeffs.append(complex(*ast.literal_eval(line)))

    def freq(i):  # 下标 i -> 频率 m: 0, +1, -1, +2, -2, ...
        return 0 if i == 0 else ((i + 1) // 2) * (1 if i % 2 else -1)

    return [(freq(i), c) for i, c in enumerate(coeffs)]

def compute_points(terms, samples=20000, chunk=1000):
    """分块 numpy 计算:控制内存,4001 项 x 20000 点约几秒"""
    m = np.array([t[0] for t in terms], dtype=float)
    c = np.array([t[1] for t in terms], dtype=complex)
    pts = []
    for s in range(0, samples, chunk):
        theta = np.arange(s, min(s + chunk, samples)) / samples * 2 * np.pi
        Z = (c[:, None] * np.exp(1j * m[:, None] * theta[None, :])).sum(axis=0)
        pts.extend(zip(Z.real.tolist(), (-Z.imag).tolist()))
    return pts

def draw(pts):
    tl.setup(960, 720)
    tl.tracer(0, 0)
    tl.hideturtle()
    tl.penup()
    tl.pensize(1.4)          # 可调画笔粗细
    tl.goto(pts[0])
    tl.pendown()
    for i, p in enumerate(pts):
        tl.goto(p)
        if i % 200 == 0:
            tl.update()
    tl.update()
    tl.done()

if __name__ == "__main__":
    terms = load_coeffs("datas.txt")        # 自动读取全部 4001 项,无需改 N
    draw(compute_points(terms, samples=20000))
