# -*- coding: utf-8 -*-

"""
OVITO 导出的薄层轨迹：
在 X-Z 平面上计算平均位移矢量场。

本版本特点：
1. 不拟合轮廓线；
2. 用箭头/小点的空间分布自然表达液滴轮廓；
3. 每个 bin 内使用所有原子的平均位移；
4. 箭头位置默认锚定在终止时刻 t2；
5. 坐标轴字体 Times New Roman，字号 22；
6. 坐标轴刻度朝里；
7. 上轴和右轴不显示刻度线；
8. 坐标轴边框粗细为 1.5；
9. 自动扫描所有时间段，统一坐标范围；
10. 所有图的 X/Z 显示跨度一致；
11. 所有图中基底表面 Z = 0 虚线处于相同高度。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 论文图字体和坐标轴样式
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

AXIS_FONT_SIZE = 22
TICK_FONT_SIZE = 22

AXIS_LINE_WIDTH = 1.5
TICK_WIDTH = 1.5
TICK_LENGTH = 7.0


# ============================================================
# 用户参数区：主要改这里
# ============================================================

DUMP_FILE = r"D:\Desktop\simulation data\displace\\new1825.lammpstrj"

OUTPUT_DIR = r"D:\Desktop\simulation data\displace\\1825slice_vector_field_final"

# 如果导出的薄层里只有液滴原子，设为 None
# 如果还有其他类型原子，例如液滴 type = 2,3，则改成 {2, 3}
LIQUID_TYPES = None
# LIQUID_TYPES = {2, 3}

# timestep -> ps 的换算
# 例如 1 fs 对应 0.001 ps
TIMESTEP_TO_PS = 0.001

# 如果润湿开始不是 timestep = 0，改这里
TIME_ZERO_STEP = 0

# 基底表面高度，用来让图中的 Z = 0 对应基底表面
SURFACE_Z = 23.0


# ============================================================
# 时间区间，单位 ps
# ============================================================

TIME_INTERVALS_PS = [
    (0, 4),
    (5, 8),
    (10, 20),
    # (25, 30),
]


# ============================================================
# 分箱参数
# ============================================================

BIN_X = 2.0
BIN_Z = 2.0

MIN_ATOMS_PER_BIN = 1

VECTOR_STRIDE = 1


# ============================================================
# 箭头分布锚定在哪个时刻
# ============================================================

# "start"  : 箭头分布显示 t1 轮廓
# "end"    : 箭头分布显示 t2 轮廓，推荐
# "middle" : 箭头分布显示中间形状
ANCHOR_FRAME = "end"


# ============================================================
# 位移显示设置
# ============================================================

PLOT_MODE = "displacement"

DISPLACEMENT_MAGNIFICATION = 0.5
VELOCITY_MAGNIFICATION = 1.0

USE_SMALL_VECTOR_AS_DOTS = True

# 0 表示基本所有有效矢量都画成箭头
# 如果想让更多小位移变成点，可以改成 20、30、40
SMALL_VECTOR_PERCENTILE = 0.0

ABSOLUTE_SMALL_THRESHOLD = None


# ============================================================
# 箭头样式
# ============================================================

VECTOR_COLOR = "#4C9BD4"

SMALL_POINT_SIZE = 6
SMALL_POINT_ALPHA = 0.75

ARROW_ALPHA = 0.90

ARROW_WIDTH = 0.005

HEAD_WIDTH = 4.3
HEAD_LENGTH = 5.2
HEAD_AXIS_LENGTH = 4.8

PIVOT = "middle"


# ============================================================
# 图像设置
# ============================================================

FIGSIZE = (5.8, 5.8)
DPI = 300

CENTER_X_BY_DROPLET = True
EQUAL_ASPECT = True

MARGIN_X = 6.0
MARGIN_Z = 6.0

SHOW_SURFACE_LINE = True
SAVE_VECTOR_DATA = True

REMOVE_GLOBAL_TRANSLATION = False


# ============================================================
# 全局坐标范围设置
# ============================================================

# True：所有时间段使用统一坐标范围，适合组图
USE_GLOBAL_AXIS_RANGE = True

# 固定 Z 轴下限，这样 Z = 0 基底线在所有图中位置一致
GLOBAL_Z_MIN = -5.0

# X 轴是否强制以 0 为中心
CENTER_X_AXIS_AT_ZERO = True


# ============================================================
# 读取 dump 文件
# ============================================================

def normalize_column_name(name):
    mapping = {
        "ParticleIdentifier": "id",
        "Particle.Identifier": "id",
        "ParticleType": "type",
        "Particle.Type": "type",
        "Position.X": "x",
        "Position.Y": "y",
        "Position.Z": "z",
        "Velocity.X": "vx",
        "Velocity.Y": "vy",
        "Velocity.Z": "vz",
    }
    return mapping.get(name, name)


def parse_box_bounds(lines):
    vals = []

    for line in lines:
        nums = [float(x) for x in line.split()]
        vals.append(nums)

    xlo_b, xhi_b = vals[0][0], vals[0][1]
    ylo_b, yhi_b = vals[1][0], vals[1][1]
    zlo_b, zhi_b = vals[2][0], vals[2][1]

    xy = vals[0][2] if len(vals[0]) >= 3 else 0.0
    xz = vals[1][2] if len(vals[1]) >= 3 else 0.0
    yz = vals[2][2] if len(vals[2]) >= 3 else 0.0

    xlo = xlo_b - min(0.0, xy, xz, xy + xz)
    xhi = xhi_b - max(0.0, xy, xz, xy + xz)

    ylo = ylo_b - min(0.0, yz)
    yhi = yhi_b - max(0.0, yz)

    return {
        "xlo": xlo,
        "xhi": xhi,
        "ylo": ylo,
        "yhi": yhi,
        "zlo": zlo_b,
        "zhi": zhi_b,
        "xy": xy,
        "xz": xz,
        "yz": yz,
    }


def scaled_to_cartesian(xs, ys, zs, box):
    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    lz = box["zhi"] - box["zlo"]

    x = box["xlo"] + xs * lx + ys * box["xy"] + zs * box["xz"]
    y = box["ylo"] + ys * ly + zs * box["yz"]
    z = box["zlo"] + zs * lz

    return x, y, z


def find_coord_columns(columns):
    candidates = [
        ("unwrapped", ("xu", "yu", "zu")),
        ("cartesian", ("x", "y", "z")),
        ("scaled", ("xs", "ys", "zs")),
    ]

    for style, names in candidates:
        if all(name in columns for name in names):
            coord_idx = [columns.index(name) for name in names]
            return style, coord_idx

    raise ValueError(
        "没有找到坐标列。需要 x y z、xu yu zu、xs ys zs "
        "或 OVITO 的 Position.X Position.Y Position.Z。"
    )


def read_one_frame(f):
    line = f.readline()

    while line and not line.startswith("ITEM: TIMESTEP"):
        line = f.readline()

    if not line:
        return None

    step = int(f.readline().strip())

    line = f.readline()
    if not line.startswith("ITEM: NUMBER OF ATOMS"):
        raise ValueError("格式错误：缺少 ITEM: NUMBER OF ATOMS")

    n_atoms = int(f.readline().strip())

    line = f.readline()
    if not line.startswith("ITEM: BOX BOUNDS"):
        raise ValueError("格式错误：缺少 ITEM: BOX BOUNDS")

    box_lines = [f.readline(), f.readline(), f.readline()]
    box = parse_box_bounds(box_lines)

    line = f.readline()
    if not line.startswith("ITEM: ATOMS"):
        raise ValueError("格式错误：缺少 ITEM: ATOMS")

    raw_columns = line.split()[2:]
    columns = [normalize_column_name(c) for c in raw_columns]

    if "id" not in columns:
        raise ValueError(
            "dump 文件必须包含 id 列。请在 OVITO 导出时保留 Particle Identifier。"
        )

    id_idx = columns.index("id")
    type_idx = columns.index("type") if "type" in columns else None
    coord_style, coord_idx = find_coord_columns(columns)

    ids = np.zeros(n_atoms, dtype=int)
    types = np.ones(n_atoms, dtype=int)
    coords = np.zeros((n_atoms, 3), dtype=float)

    for i in range(n_atoms):
        parts = f.readline().split()

        ids[i] = int(float(parts[id_idx]))

        if type_idx is not None:
            types[i] = int(float(parts[type_idx]))

        a = float(parts[coord_idx[0]])
        b = float(parts[coord_idx[1]])
        c = float(parts[coord_idx[2]])

        if coord_style == "scaled":
            x, y, z = scaled_to_cartesian(a, b, c, box)
        else:
            x, y, z = a, b, c

        coords[i, 0] = x
        coords[i, 1] = y
        coords[i, 2] = z

    return {
        "step": step,
        "time_ps": (step - TIME_ZERO_STEP) * TIMESTEP_TO_PS,
        "box": box,
        "ids": ids,
        "types": types,
        "coords": coords,
    }


def load_all_frames(filename):
    frames = []

    with open(filename, "r") as f:
        while True:
            frame = read_one_frame(f)

            if frame is None:
                break

            frames.append(frame)

    if len(frames) == 0:
        raise RuntimeError("没有从 dump 文件中读取到任何帧。")

    return frames


def find_nearest_frame(frames, target_time_ps):
    times = np.array([frame["time_ps"] for frame in frames])
    idx = np.argmin(np.abs(times - target_time_ps))
    return frames[idx]


# ============================================================
# 工具函数
# ============================================================

def minimum_image(delta, box_length):
    return delta - box_length * np.round(delta / box_length)


def periodic_mean_position(pos, lo, hi):
    length = hi - lo
    angle = 2.0 * np.pi * (pos - lo) / length

    mean_sin = np.mean(np.sin(angle))
    mean_cos = np.mean(np.cos(angle))

    mean_angle = np.arctan2(mean_sin, mean_cos)

    if mean_angle < 0:
        mean_angle += 2.0 * np.pi

    return lo + length * mean_angle / (2.0 * np.pi)


def select_liquid_atoms(frame):
    ids = frame["ids"]
    types = frame["types"]
    coords = frame["coords"]

    if LIQUID_TYPES is None:
        mask = np.ones(len(ids), dtype=bool)
    else:
        mask = np.isin(types, list(LIQUID_TYPES))

    return ids[mask], coords[mask]


def build_id_to_coord(ids, coords):
    return {int(atom_id): coords[i] for i, atom_id in enumerate(ids)}


# ============================================================
# 计算位移矢量场
# ============================================================

def calculate_anchor_based_displacement_field(frame_start, frame_end):
    ids0, coords0 = select_liquid_atoms(frame_start)
    ids1, coords1 = select_liquid_atoms(frame_end)

    id_to_coord0 = build_id_to_coord(ids0, coords0)
    id_to_coord1 = build_id_to_coord(ids1, coords1)

    common_ids = sorted(set(id_to_coord0.keys()) & set(id_to_coord1.keys()))

    if len(common_ids) == 0:
        raise RuntimeError("两个时间帧之间没有共同 atom id，无法追踪位移。")

    box = frame_start["box"]
    lx = box["xhi"] - box["xlo"]

    start_coords = np.array([id_to_coord0[i] for i in common_ids], dtype=float)
    end_coords = np.array([id_to_coord1[i] for i in common_ids], dtype=float)

    dx = end_coords[:, 0] - start_coords[:, 0]
    dz = end_coords[:, 2] - start_coords[:, 2]

    dx = minimum_image(dx, lx)

    if REMOVE_GLOBAL_TRANSLATION:
        dx = dx - np.mean(dx)
        dz = dz - np.mean(dz)

    if ANCHOR_FRAME == "start":
        anchor_coords = start_coords.copy()
    elif ANCHOR_FRAME == "end":
        anchor_coords = end_coords.copy()
    elif ANCHOR_FRAME == "middle":
        anchor_coords = start_coords.copy()
        anchor_coords[:, 0] = start_coords[:, 0] + 0.5 * dx
        anchor_coords[:, 2] = start_coords[:, 2] + 0.5 * dz
    else:
        raise ValueError("ANCHOR_FRAME 必须是 'start'、'end' 或 'middle'。")

    x_anchor = anchor_coords[:, 0]
    z_anchor = anchor_coords[:, 2]

    if CENTER_X_BY_DROPLET:
        x_center = periodic_mean_position(x_anchor, box["xlo"], box["xhi"])
    else:
        x_center = 0.0

    x_plot = minimum_image(x_anchor - x_center, lx)
    z_plot = z_anchor - SURFACE_Z

    x_min = np.floor(np.min(x_plot) / BIN_X) * BIN_X
    x_max = np.ceil(np.max(x_plot) / BIN_X) * BIN_X
    z_min = np.floor(np.min(z_plot) / BIN_Z) * BIN_Z
    z_max = np.ceil(np.max(z_plot) / BIN_Z) * BIN_Z

    x_edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    z_edges = np.arange(z_min, z_max + BIN_Z, BIN_Z)

    bin_x = np.digitize(x_plot, x_edges) - 1
    bin_z = np.digitize(z_plot, z_edges) - 1

    n_x = len(x_edges) - 1
    n_z = len(z_edges) - 1

    dt_ps = frame_end["time_ps"] - frame_start["time_ps"]

    if dt_ps <= 0:
        raise ValueError("结束帧时间必须大于起始帧时间。")

    Xc_list, Zc_list = [], []
    Dx_list, Dz_list = [], []
    Vx_list, Vz_list = [], []
    Count_list = []
    StdDx_list, StdDz_list = [], []

    for ix in range(n_x):
        for iz in range(n_z):
            atom_mask = (bin_x == ix) & (bin_z == iz)

            if np.count_nonzero(atom_mask) < MIN_ATOMS_PER_BIN:
                continue

            dx_bin = dx[atom_mask]
            dz_bin = dz[atom_mask]
            x_bin = x_plot[atom_mask]
            z_bin = z_plot[atom_mask]

            dx_mean = np.mean(dx_bin)
            dz_mean = np.mean(dz_bin)

            Xc_list.append(np.mean(x_bin))
            Zc_list.append(np.mean(z_bin))
            Dx_list.append(dx_mean)
            Dz_list.append(dz_mean)
            Vx_list.append(dx_mean / dt_ps)
            Vz_list.append(dz_mean / dt_ps)
            Count_list.append(len(dx_bin))
            StdDx_list.append(np.std(dx_bin))
            StdDz_list.append(np.std(dz_bin))

    Xc = np.asarray(Xc_list)
    Zc = np.asarray(Zc_list)
    Dx = np.asarray(Dx_list)
    Dz = np.asarray(Dz_list)
    Vx = np.asarray(Vx_list)
    Vz = np.asarray(Vz_list)
    Count = np.asarray(Count_list)
    StdDx = np.asarray(StdDx_list)
    StdDz = np.asarray(StdDz_list)

    return Xc, Zc, Dx, Dz, Vx, Vz, Count, StdDx, StdDz


# ============================================================
# 保存数据
# ============================================================

def save_vector_data(output_path, Xc, Zc, Dx, Dz, Vx, Vz, Count, StdDx, StdDz):
    data = np.column_stack([Xc, Zc, Dx, Dz, Vx, Vz, Count, StdDx, StdDz])

    header = (
        "X_center Z_center "
        "mean_displacement_x mean_displacement_z "
        "mean_velocity_x mean_velocity_z "
        "atom_count std_displacement_x std_displacement_z"
    )

    np.savetxt(
        output_path,
        data,
        fmt="%.8e %.8e %.8e %.8e %.8e %.8e %.0f %.8e %.8e",
        header=header,
        comments=""
    )


# ============================================================
# 全局坐标范围
# ============================================================

def compute_global_axis_limits(all_results):
    """
    根据所有时间段的矢量点，自动计算统一坐标范围。

    目标：
    1. 所有图使用同一个 xlim 和 zlim；
    2. X/Z 坐标跨度相同；
    3. Z = 0 基底虚线在所有图中处于相同高度；
    4. 不需要手动为每张图设置范围。
    """

    all_x = []
    all_z = []

    for result in all_results:
        Xc = result["Xc"]
        Zc = result["Zc"]

        if len(Xc) == 0:
            continue

        all_x.append(Xc)
        all_z.append(Zc)

    if len(all_x) == 0:
        raise RuntimeError("没有有效矢量点，无法计算全局坐标范围。")

    all_x = np.concatenate(all_x)
    all_z = np.concatenate(all_z)

    x_data_min = np.min(all_x) - MARGIN_X
    x_data_max = np.max(all_x) + MARGIN_X

    z_data_min = np.min(all_z) - MARGIN_Z
    z_data_max = np.max(all_z) + MARGIN_Z

    # 保证基底表面线 Z = 0 在范围内
    z_data_min = min(z_data_min, GLOBAL_Z_MIN)
    z_data_max = max(z_data_max, 0.0)

    # 固定 Z 轴下限，使 Z=0 在所有图中的高度一致
    z_low = GLOBAL_Z_MIN

    z_span_needed = z_data_max - z_low

    if CENTER_X_AXIS_AT_ZERO:
        x_abs = max(abs(x_data_min), abs(x_data_max))
        x_span_needed = 2.0 * x_abs
    else:
        x_span_needed = x_data_max - x_data_min

    final_span = max(x_span_needed, z_span_needed)

    if CENTER_X_AXIS_AT_ZERO:
        x_low = -0.5 * final_span
        x_high = 0.5 * final_span
    else:
        x_center = 0.5 * (x_data_min + x_data_max)
        x_low = x_center - 0.5 * final_span
        x_high = x_center + 0.5 * final_span

    z_high = z_low + final_span

    return {
        "xlim": (x_low, x_high),
        "zlim": (z_low, z_high),
        "span": final_span,
    }


def set_axis_ranges(ax, Xc, Zc, global_limits=None):
    """
    设置坐标范围。

    如果 global_limits 不为 None：
        所有图使用同一个 xlim/zlim；
        基底线 Z=0 在所有图中的位置一致。

    如果 global_limits 为 None：
        每张图自动范围，但 X/Z 跨度相同。
    """

    if global_limits is not None:
        ax.set_xlim(global_limits["xlim"])
        ax.set_ylim(global_limits["zlim"])
        return

    x_low = np.min(Xc) - MARGIN_X
    x_high = np.max(Xc) + MARGIN_X

    z_low = np.min(Zc) - MARGIN_Z
    z_high = np.max(Zc) + MARGIN_Z

    if SHOW_SURFACE_LINE:
        z_low = min(z_low, 0.0)
        z_high = max(z_high, 0.0)

    z_low = min(z_low, GLOBAL_Z_MIN)

    x_span = x_high - x_low
    z_span = z_high - z_low

    final_span = max(x_span, z_span)

    if CENTER_X_AXIS_AT_ZERO:
        x_center = 0.0
    else:
        x_center = 0.5 * (x_low + x_high)

    ax.set_xlim(
        x_center - 0.5 * final_span,
        x_center + 0.5 * final_span
    )

    ax.set_ylim(
        z_low,
        z_low + final_span
    )


# ============================================================
# 绘图
# ============================================================

def plot_vector_field(
    frame_start,
    frame_end,
    Xc,
    Zc,
    Dx,
    Dz,
    Vx,
    Vz,
    output_path,
    global_limits=None,
):
    if len(Xc) == 0:
        print("没有有效矢量可以绘制。")
        return

    if VECTOR_STRIDE > 1:
        idx = np.arange(0, len(Xc), VECTOR_STRIDE)
        Xc = Xc[idx]
        Zc = Zc[idx]
        Dx = Dx[idx]
        Dz = Dz[idx]
        Vx = Vx[idx]
        Vz = Vz[idx]

    if PLOT_MODE == "displacement":
        U = Dx * DISPLACEMENT_MAGNIFICATION
        W = Dz * DISPLACEMENT_MAGNIFICATION
        magnitude = np.sqrt(Dx ** 2 + Dz ** 2)
    elif PLOT_MODE == "velocity":
        U = Vx * VELOCITY_MAGNIFICATION
        W = Vz * VELOCITY_MAGNIFICATION
        magnitude = np.sqrt(Vx ** 2 + Vz ** 2)
    else:
        raise ValueError("PLOT_MODE 必须是 'displacement' 或 'velocity'。")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if USE_SMALL_VECTOR_AS_DOTS:
        if ABSOLUTE_SMALL_THRESHOLD is not None:
            small_threshold = ABSOLUTE_SMALL_THRESHOLD
        else:
            small_threshold = np.percentile(magnitude, SMALL_VECTOR_PERCENTILE)

        small_mask = magnitude < small_threshold
        large_mask = ~small_mask

        ax.scatter(
            Xc[small_mask],
            Zc[small_mask],
            s=SMALL_POINT_SIZE,
            color=VECTOR_COLOR,
            alpha=SMALL_POINT_ALPHA,
            linewidths=0,
            zorder=3,
        )

        if np.any(large_mask):
            ax.quiver(
                Xc[large_mask],
                Zc[large_mask],
                U[large_mask],
                W[large_mask],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=ARROW_WIDTH,
                headwidth=HEAD_WIDTH,
                headlength=HEAD_LENGTH,
                headaxislength=HEAD_AXIS_LENGTH,
                pivot=PIVOT,
                alpha=ARROW_ALPHA,
                color=VECTOR_COLOR,
                zorder=4,
            )
    else:
        ax.quiver(
            Xc,
            Zc,
            U,
            W,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=ARROW_WIDTH,
            headwidth=HEAD_WIDTH,
            headlength=HEAD_LENGTH,
            headaxislength=HEAD_AXIS_LENGTH,
            pivot=PIVOT,
            alpha=ARROW_ALPHA,
            color=VECTOR_COLOR,
            zorder=4,
        )

    if SHOW_SURFACE_LINE:
        ax.axhline(
            0.0,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            zorder=1,
        )

    ax.set_xlabel(
        "X (Å)",
        fontname="Times New Roman",
        fontsize=AXIS_FONT_SIZE,
    )

    ax.set_ylabel(
        "Z (Å)",
        fontname="Times New Roman",
        fontsize=AXIS_FONT_SIZE,
    )

    if EQUAL_ASPECT:
        ax.set_aspect("equal", adjustable="box")

    set_axis_ranges(ax, Xc, Zc, global_limits=global_limits)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        labelsize=TICK_FONT_SIZE,
    )

    for tick_label in ax.get_xticklabels():
        tick_label.set_fontname("Times New Roman")
        tick_label.set_fontsize(TICK_FONT_SIZE)

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontname("Times New Roman")
        tick_label.set_fontsize(TICK_FONT_SIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)

    # 固定输出图片尺寸，不使用 bbox_inches="tight"
    fig.subplots_adjust(
        left=0.20,
        right=0.96,
        bottom=0.18,
        top=0.96,
    )

    plt.savefig(output_path, dpi=DPI)
    plt.close()

    print(f"图片已保存: {output_path}")


# ============================================================
# 主程序
# ============================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始读取 OVITO 导出的薄层 dump 文件...")
    print(f"输入文件: {DUMP_FILE}")
    print(f"ANCHOR_FRAME = {ANCHOR_FRAME}")

    frames = load_all_frames(DUMP_FILE)

    print(f"总帧数: {len(frames)}")
    print(f"第一帧: step = {frames[0]['step']}, time = {frames[0]['time_ps']:.3f} ps")
    print(f"最后帧: step = {frames[-1]['step']}, time = {frames[-1]['time_ps']:.3f} ps")
    print()

    # ========================================================
    # 第一次循环：先计算所有时间段的数据
    # ========================================================

    all_results = []

    for index, (t_start, t_end) in enumerate(TIME_INTERVALS_PS, start=1):
        frame_start = find_nearest_frame(frames, t_start)
        frame_end = find_nearest_frame(frames, t_end)

        print(f"预计算时间段 {index}: 目标 {t_start}-{t_end} ps")
        print(
            f"实际使用帧: "
            f"{frame_start['time_ps']:.3f} ps -> {frame_end['time_ps']:.3f} ps"
        )

        Xc, Zc, Dx, Dz, Vx, Vz, Count, StdDx, StdDz = (
            calculate_anchor_based_displacement_field(frame_start, frame_end)
        )

        print(f"有效矢量数: {len(Xc)}")

        all_results.append(
            {
                "index": index,
                "t_start": t_start,
                "t_end": t_end,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "Xc": Xc,
                "Zc": Zc,
                "Dx": Dx,
                "Dz": Dz,
                "Vx": Vx,
                "Vz": Vz,
                "Count": Count,
                "StdDx": StdDx,
                "StdDz": StdDz,
            }
        )

        print()

    # ========================================================
    # 自动计算所有图共用坐标范围
    # ========================================================

    if USE_GLOBAL_AXIS_RANGE:
        global_limits = compute_global_axis_limits(all_results)

        print("全局坐标范围:")
        print(f"  xlim = {global_limits['xlim']}")
        print(f"  zlim = {global_limits['zlim']}")
        print(f"  span = {global_limits['span']:.3f}")
        print()
    else:
        global_limits = None

    # ========================================================
    # 第二次循环：统一坐标范围后出图
    # ========================================================

    for result in all_results:
        index = result["index"]
        t_start = result["t_start"]
        t_end = result["t_end"]

        Xc = result["Xc"]
        Zc = result["Zc"]
        Dx = result["Dx"]
        Dz = result["Dz"]
        Vx = result["Vx"]
        Vz = result["Vz"]
        Count = result["Count"]
        StdDx = result["StdDx"]
        StdDz = result["StdDz"]
        frame_start = result["frame_start"]
        frame_end = result["frame_end"]

        if len(Xc) == 0:
            print(f"时间段 {index} 没有有效矢量，跳过。")
            print()
            continue

        base_name = f"slice_vector_field_{t_start:g}_{t_end:g}ps_anchor_{ANCHOR_FRAME}"

        fig_path = output_dir / f"{base_name}.png"
        data_path = output_dir / f"{base_name}.txt"

        plot_vector_field(
            frame_start=frame_start,
            frame_end=frame_end,
            Xc=Xc,
            Zc=Zc,
            Dx=Dx,
            Dz=Dz,
            Vx=Vx,
            Vz=Vz,
            output_path=fig_path,
            global_limits=global_limits,
        )

        if SAVE_VECTOR_DATA:
            save_vector_data(
                output_path=data_path,
                Xc=Xc,
                Zc=Zc,
                Dx=Dx,
                Dz=Dz,
                Vx=Vx,
                Vz=Vz,
                Count=Count,
                StdDx=StdDx,
                StdDz=StdDz,
            )
            print(f"矢量数据已保存: {data_path}")

        print()

    print("全部处理完成。")


if __name__ == "__main__":
    main()