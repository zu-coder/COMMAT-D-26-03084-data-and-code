# -*- coding: utf-8 -*-
"""
计算润湿过程中液滴铺展半径随时间变化：优化版

适用场景：
1. LAMMPS dump / lammpstrj 轨迹文件；
2. 可处理“只保留液滴原子”的 dump，也可处理“含基底+液滴”的 dump；
3. z 方向为基底法向，x-y 平面为润湿铺展平面；
4. 液滴铺展形貌近似圆形。

核心计算方法：
1. 根据基底上表面 z 坐标选取液滴接触层原子；
2. 对接触层原子在 x-y 平面进行周期性边界修正；
3. 使用稳健中心和径向分位数计算铺展半径；
4. 同时输出最大半径、面积等效半径、外缘圆拟合半径作为对照。

推荐论文主结果：
- 使用 R_quantile 作为铺展半径；
- 推荐 RADIUS_QUANTILE = 0.95 或 0.98；
- R_max 只作为参考，因为它容易受单个边缘原子影响；
- R_circle 只作为外缘形貌接近圆形时的辅助验证。

输出：
1. CSV 数据文件；
2. PNG 曲线图；
3. 可选：接触层原子数量、中心坐标、多个半径定义。
"""

import os
import csv
import gzip
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 一、用户主要修改这里
# ============================================================

# ------------------------------------------------------------
# 1. 输入输出文件
# ------------------------------------------------------------

# dump 文件路径
# Windows 路径务必使用 r"..." 原始字符串，避免反斜杠转义问题。
DUMP_FILE = r"D:\Desktop\simulation data\\R\\ywet300_1825.lammpstrj"

# 输出文件
OUTPUT_CSV = r"D:\Desktop\simulation data\\R\spreading_radius_optimized1825.csv"
OUTPUT_PNG = r"D:\Desktop\simulation data\\R\spreading_radius_optimized1825.png"


# ------------------------------------------------------------
# 2. dump 中是否只包含液滴原子
# ------------------------------------------------------------

# 如果你已经在 OVITO 中删除了基底原子，只保留液滴原子，设为 True。
# 如果 dump 中仍然包含基底和液滴，设为 False，并正确设置 LIQUID_TYPES。
DUMP_CONTAINS_ONLY_LIQUID = True

# 当 DUMP_CONTAINS_ONLY_LIQUID = False 时使用。
# 例如液滴 Fe 和 Si 分别为 type 2、3，则写 {2, 3}。
LIQUID_TYPES = {2, 3}


# ------------------------------------------------------------
# 3. 时间参数
# ------------------------------------------------------------

# LAMMPS timestep 对应真实时间。
# units metal 中，如果 timestep 0.001，则 1 step = 0.001 ps。
DT = 0.001
TIME_UNIT = "ps"

# 时间是否从第一帧归零。
# True: time = (timestep - first_timestep) * DT
# False: time = timestep * DT
ZERO_TIME_AT_FIRST_FRAME = True


# ------------------------------------------------------------
# 4. 基底表面和接触层定义
# ------------------------------------------------------------

# 推荐使用基底上表面的真实 z 坐标。
USE_SUBSTRATE_SURFACE_Z = True

# 基底上表面 z 坐标，单位与 dump 坐标一致，metal 单位下通常为 Å。
# 必须改成你模型中最上层牛顿层基底表面的 z 坐标。
SUBSTRATE_Z_TOP = 22.0

# 允许液滴原子略低于基底表面的容差。
# 推荐 0.0 ~ 2.0 Å，不建议太大。
Z_TOLERANCE_BELOW_SURFACE = 1.0

# 接触层厚度：z <= SUBSTRATE_Z_TOP + LAYER_THICKNESS。
# Fe / Fe-Si 体系常可先用 4.0 ~ 6.0 Å。
LAYER_THICKNESS = 5.0

# 若 USE_SUBSTRATE_SURFACE_Z = False，则用液滴底部低分位数估计底部高度。
BOTTOM_PERCENTILE = 1.0


# ------------------------------------------------------------
# 5. 半径计算参数
# ------------------------------------------------------------

# 主结果：径向距离分位数半径。
# 推荐 0.95 或 0.98。0.95 更稳健，0.98 更接近外边界。
RADIUS_QUANTILE = 0.95

# 用于稳健计算中心和剔除极端远离点。
# 0.995 表示只去掉最外侧 0.5% 的极端点。
# 若希望几乎不剔除，可设为 1.0。
OUTER_OUTLIER_QUANTILE = 0.995

# 外缘圆拟合只使用较外侧的点，而不是整个圆盘接触层。
# 例如 0.80 表示只使用 r >= 80% 分位数的外缘点做圆拟合。
BOUNDARY_FIT_INNER_QUANTILE = 0.90
BOUNDARY_FIT_OUTER_QUANTILE = 0.995

# 每帧接触层最少原子数。
# 太小则结果不可靠。实际体系建议至少 20；若初始帧接触很少，可临时设小。
MIN_CONTACT_ATOMS = 10


# ------------------------------------------------------------
# 6. 周期性边界和平滑
# ------------------------------------------------------------

# 如果液滴可能接近或跨越 x-y 周期边界，设为 True。
# 如果液滴一直在盒子中央，True 也通常没问题。
UNWRAP_XY = True

# 对输出曲线做简单滑动平均。
# 1 表示不平滑；5 表示 5 帧平滑。
# 建议论文拟合使用原始数据，平滑曲线只用于展示。
SMOOTH_WINDOW = 1


# ============================================================
# 二、文件读取部分
# ============================================================

def open_text_file(filename):
    """打开普通文本文件或 .gz 压缩文本文件。"""
    filename = str(filename)
    if filename.endswith(".gz"):
        return gzip.open(filename, "rt")
    return open(filename, "r")


def parse_box_bounds(lines):
    """
    读取 BOX BOUNDS。

    支持正交盒子：
        xlo xhi
        ylo yhi
        zlo zhi

    对 triclinic 盒子只读取每行前两个数字。
    若你的盒子有明显倾斜，建议先在 dump 中输出转换后的 xu/yu/zu 或单独处理 tilt factors。
    """
    lo = []
    hi = []

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"BOX BOUNDS 行格式错误: {line}")
        lo.append(float(parts[0]))
        hi.append(float(parts[1]))

    return np.array(lo, dtype=float), np.array(hi, dtype=float)


def get_coordinate_columns(columns):
    """
    自动识别 dump 中的坐标列。

    优先级：
    1. xu yu zu：展开坐标；
    2. x y z：普通坐标；
    3. xs ys zs：归一化坐标；
    4. xsu ysu zsu：归一化展开坐标。
    """
    col_index = {name: i for i, name in enumerate(columns)}

    if all(c in col_index for c in ["xu", "yu", "zu"]):
        return col_index["xu"], col_index["yu"], col_index["zu"], "unwrapped"

    if all(c in col_index for c in ["x", "y", "z"]):
        return col_index["x"], col_index["y"], col_index["z"], "wrapped"

    if all(c in col_index for c in ["xs", "ys", "zs"]):
        return col_index["xs"], col_index["ys"], col_index["zs"], "scaled"

    if all(c in col_index for c in ["xsu", "ysu", "zsu"]):
        return col_index["xsu"], col_index["ysu"], col_index["zsu"], "scaled"

    raise ValueError(
        "无法识别坐标列。dump 文件中需要包含以下任意一种坐标：\n"
        "1. xu yu zu\n"
        "2. x y z\n"
        "3. xs ys zs\n"
        "4. xsu ysu zsu\n"
        f"当前检测到的列为: {columns}"
    )


def read_lammps_dump(filename):
    """
    逐帧读取 LAMMPS dump 文件。

    返回字典：
        timestep
        coords: N x 3 坐标
        atom_types: N 个原子类型；如果 dump 没有 type 列，则为 None
        box_lo, box_hi
        columns
    """
    with open_text_file(filename) as f:
        while True:
            line = f.readline()

            if not line:
                break

            line = line.strip()

            if not line:
                continue

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(f.readline().strip())

            number_header = f.readline().strip()
            if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("dump 格式错误：缺少 ITEM: NUMBER OF ATOMS")

            n_atoms = int(f.readline().strip())

            box_header = f.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("dump 格式错误：缺少 ITEM: BOX BOUNDS")

            box_lines = [f.readline().strip() for _ in range(3)]
            box_lo, box_hi = parse_box_bounds(box_lines)

            atoms_header = f.readline().strip()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError("dump 格式错误：缺少 ITEM: ATOMS")

            columns = atoms_header.split()[2:]
            ix, iy, iz, coord_mode = get_coordinate_columns(columns)

            type_col = None
            if "type" in columns:
                type_col = columns.index("type")

            coords = np.empty((n_atoms, 3), dtype=float)
            atom_types = None if type_col is None else np.empty(n_atoms, dtype=int)

            for i in range(n_atoms):
                parts = f.readline().split()

                if len(parts) < len(columns):
                    raise ValueError(
                        f"第 {timestep} 步中有原子行列数不足：{parts}"
                    )

                coords[i, 0] = float(parts[ix])
                coords[i, 1] = float(parts[iy])
                coords[i, 2] = float(parts[iz])

                if type_col is not None:
                    atom_types[i] = int(float(parts[type_col]))

            if coord_mode == "scaled":
                box_len = box_hi - box_lo
                coords = box_lo + coords * box_len

            yield {
                "timestep": timestep,
                "coords": coords,
                "atom_types": atom_types,
                "box_lo": box_lo,
                "box_hi": box_hi,
                "columns": columns,
            }


# ============================================================
# 三、几何计算部分
# ============================================================

def filter_liquid_atoms(coords, atom_types):
    """根据设置筛选液滴原子坐标。"""
    if DUMP_CONTAINS_ONLY_LIQUID:
        return coords

    if atom_types is None:
        raise ValueError(
            "DUMP_CONTAINS_ONLY_LIQUID = False，但 dump 中没有 type 列，"
            "无法根据 LIQUID_TYPES 筛选液滴原子。"
        )

    mask = np.isin(atom_types, list(LIQUID_TYPES))
    return coords[mask]


def unwrap_xy_relative_to_median(coords, box_lo, box_hi):
    """
    用 minimum-image 思路把液滴在 x-y 平面拼回一起。

    注意：该方法要求液滴横向尺寸明显小于盒子尺寸。
    如果液滴铺展后接近盒子边界，请增大盒子或使用 xu yu zu 坐标。
    """
    if coords.shape[0] == 0:
        return coords

    out = coords.copy()
    box_len = box_hi - box_lo

    for dim in (0, 1):
        L = box_len[dim]
        if L <= 0:
            continue

        ref = np.median(out[:, dim])
        delta = out[:, dim] - ref
        delta -= np.round(delta / L) * L
        out[:, dim] = ref + delta

    return out


def select_contact_layer(coords):
    """根据 z 坐标选择接触层液滴原子。"""
    z = coords[:, 2]

    if USE_SUBSTRATE_SURFACE_Z:
        z_ref = SUBSTRATE_Z_TOP
        z_min_contact = SUBSTRATE_Z_TOP - Z_TOLERANCE_BELOW_SURFACE
        z_max_contact = SUBSTRATE_Z_TOP + LAYER_THICKNESS

        mask = (z >= z_min_contact) & (z <= z_max_contact)
        method = "substrate_surface_z"

    else:
        if BOTTOM_PERCENTILE <= 0.0:
            z_ref = np.min(z)
        else:
            z_ref = np.percentile(z, BOTTOM_PERCENTILE)

        z_min_contact = -np.inf
        z_max_contact = z_ref + LAYER_THICKNESS
        mask = z <= z_max_contact
        method = "droplet_bottom_percentile"

    return coords[mask], float(z_ref), float(z_min_contact), float(z_max_contact), method


def robust_contact_center_and_radii(xy):
    """
    稳健计算接触层中心和径向距离。

    步骤：
    1. 用中位数估计初始中心；
    2. 根据 OUTER_OUTLIER_QUANTILE 去掉极端远点；
    3. 用剩余点重新计算中心；
    4. 计算最终径向距离。
    """
    if xy.shape[0] == 0:
        return np.nan, np.nan, np.array([]), xy

    cx0 = np.median(xy[:, 0])
    cy0 = np.median(xy[:, 1])

    r0 = np.sqrt((xy[:, 0] - cx0) ** 2 + (xy[:, 1] - cy0) ** 2)

    if OUTER_OUTLIER_QUANTILE < 1.0 and len(r0) >= MIN_CONTACT_ATOMS:
        cutoff = np.quantile(r0, OUTER_OUTLIER_QUANTILE)
        keep = r0 <= cutoff
        xy_clean = xy[keep]
    else:
        xy_clean = xy.copy()

    if xy_clean.shape[0] == 0:
        return np.nan, np.nan, np.array([]), xy_clean

    cx = np.median(xy_clean[:, 0])
    cy = np.median(xy_clean[:, 1])

    r = np.sqrt((xy_clean[:, 0] - cx) ** 2 + (xy_clean[:, 1] - cy) ** 2)

    return float(cx), float(cy), r, xy_clean


def convex_hull(points):
    """二维点集凸包，Andrew monotonic chain 算法。"""
    if len(points) <= 1:
        return points

    pts = sorted(set(map(tuple, points)))

    if len(pts) <= 1:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


def polygon_area(points):
    """计算二维多边形面积。"""
    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def fit_circle_least_squares(xy):
    """
    最小二乘圆拟合。

    输入应尽量是外缘点，而不是整个圆盘内部点。
    返回：xc, yc, radius, rms_error
    """
    if xy.shape[0] < 3:
        return np.nan, np.nan, np.nan, np.nan

    x = xy[:, 0]
    y = xy[:, 1]

    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x ** 2 + y ** 2)

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan

    Acoef, Bcoef, Ccoef = coeffs
    xc = -Acoef / 2.0
    yc = -Bcoef / 2.0
    radius_sq = xc ** 2 + yc ** 2 - Ccoef

    if radius_sq <= 0:
        return np.nan, np.nan, np.nan, np.nan

    radius = math.sqrt(radius_sq)
    rr = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    rms = math.sqrt(np.mean((rr - radius) ** 2))

    return float(xc), float(yc), float(radius), float(rms)


def boundary_circle_radius(xy_clean, cx, cy):
    """
    只使用接触区外缘点做圆拟合。

    这样避免“用整个圆盘点拟合圆”导致的半径不合理问题。
    """
    if xy_clean.shape[0] < MIN_CONTACT_ATOMS:
        return np.nan, np.nan, np.nan, np.nan, 0

    r = np.sqrt((xy_clean[:, 0] - cx) ** 2 + (xy_clean[:, 1] - cy) ** 2)

    r_low = np.quantile(r, BOUNDARY_FIT_INNER_QUANTILE)
    r_high = np.quantile(r, BOUNDARY_FIT_OUTER_QUANTILE)

    shell_mask = (r >= r_low) & (r <= r_high)
    shell = xy_clean[shell_mask]

    if shell.shape[0] < max(3, MIN_CONTACT_ATOMS // 2):
        return np.nan, np.nan, np.nan, np.nan, int(shell.shape[0])

    xcf, ycf, rf, rms = fit_circle_least_squares(shell)
    return xcf, ycf, rf, rms, int(shell.shape[0])


# ============================================================
# 四、单帧计算
# ============================================================

def failed_result(status, n_total=0):
    """生成失败帧结果。"""
    return {
        "z_ref": np.nan,
        "z_min_contact": np.nan,
        "z_max_contact": np.nan,
        "n_total": int(n_total),
        "n_contact_raw": 0,
        "n_contact_used": 0,
        "center_x": np.nan,
        "center_y": np.nan,
        "R_quantile": np.nan,
        "R_max": np.nan,
        "R_mean": np.nan,
        "R_std": np.nan,
        "R_area": np.nan,
        "R_circle": np.nan,
        "circle_center_x": np.nan,
        "circle_center_y": np.nan,
        "circle_rms_error": np.nan,
        "n_boundary_fit": 0,
        "contact_method": "none",
        "status": status,
    }


def compute_radius_for_frame(frame):
    """计算单帧铺展半径。"""
    coords_all = frame["coords"]
    atom_types = frame["atom_types"]
    box_lo = frame["box_lo"]
    box_hi = frame["box_hi"]

    liquid_coords = filter_liquid_atoms(coords_all, atom_types)

    if liquid_coords.shape[0] < MIN_CONTACT_ATOMS:
        return failed_result("too_few_liquid_atoms", liquid_coords.shape[0])

    if UNWRAP_XY:
        liquid_coords = unwrap_xy_relative_to_median(liquid_coords, box_lo, box_hi)

    contact, z_ref, z_min_contact, z_max_contact, method = select_contact_layer(liquid_coords)

    if contact.shape[0] < MIN_CONTACT_ATOMS:
        res = failed_result("too_few_contact_atoms", liquid_coords.shape[0])
        res.update({
            "z_ref": z_ref,
            "z_min_contact": z_min_contact,
            "z_max_contact": z_max_contact,
            "n_contact_raw": int(contact.shape[0]),
            "contact_method": method,
        })
        return res

    xy_raw = contact[:, :2]
    cx, cy, r, xy_clean = robust_contact_center_and_radii(xy_raw)

    if xy_clean.shape[0] < MIN_CONTACT_ATOMS or len(r) < MIN_CONTACT_ATOMS:
        res = failed_result("too_few_after_outlier_filter", liquid_coords.shape[0])
        res.update({
            "z_ref": z_ref,
            "z_min_contact": z_min_contact,
            "z_max_contact": z_max_contact,
            "n_contact_raw": int(contact.shape[0]),
            "n_contact_used": int(xy_clean.shape[0]),
            "contact_method": method,
        })
        return res

    R_quantile = float(np.quantile(r, RADIUS_QUANTILE))
    R_max = float(np.max(r))
    R_mean = float(np.mean(r))
    R_std = float(np.std(r))

    hull = convex_hull(xy_clean)
    area = polygon_area(hull)
    R_area = float(math.sqrt(area / math.pi)) if area > 0 else np.nan

    circle_x, circle_y, R_circle, circle_rms, n_boundary = boundary_circle_radius(
        xy_clean, cx, cy
    )

    return {
        "z_ref": float(z_ref),
        "z_min_contact": float(z_min_contact),
        "z_max_contact": float(z_max_contact),
        "n_total": int(liquid_coords.shape[0]),
        "n_contact_raw": int(contact.shape[0]),
        "n_contact_used": int(xy_clean.shape[0]),
        "center_x": float(cx),
        "center_y": float(cy),
        "R_quantile": R_quantile,
        "R_max": R_max,
        "R_mean": R_mean,
        "R_std": R_std,
        "R_area": R_area,
        "R_circle": R_circle,
        "circle_center_x": circle_x,
        "circle_center_y": circle_y,
        "circle_rms_error": circle_rms,
        "n_boundary_fit": n_boundary,
        "contact_method": method,
        "status": "ok",
    }


# ============================================================
# 五、输出与绘图
# ============================================================

def rolling_mean(values, window):
    """中心滑动平均，自动忽略 nan。"""
    values = np.asarray(values, dtype=float)

    if window <= 1:
        return values.copy()

    out = np.full_like(values, np.nan, dtype=float)
    half = window // 2

    for i in range(len(values)):
        left = max(0, i - half)
        right = min(len(values), i + half + 1)
        segment = values[left:right]

        if not np.all(np.isnan(segment)):
            out[i] = np.nanmean(segment)

    return out


def format_value(v):
    """CSV 输出格式。"""
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.10g}"
    return str(v)


def save_csv(rows, output_csv):
    """保存 CSV。"""
    headers = [
        "frame",
        "timestep",
        "time",
        "z_ref",
        "z_min_contact",
        "z_max_contact",
        "n_total",
        "n_contact_raw",
        "n_contact_used",
        "center_x",
        "center_y",
        "R_quantile",
        "R_max",
        "R_mean",
        "R_std",
        "R_area",
        "R_circle",
        "circle_center_x",
        "circle_center_y",
        "circle_rms_error",
        "n_boundary_fit",
        "contact_method",
        "status",
    ]

    if SMOOTH_WINDOW > 1:
        headers.append("R_quantile_smooth")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in rows:
            writer.writerow([format_value(row.get(h, "")) for h in headers])


def plot_radius(rows, output_png):
    """绘制 R-t 曲线。"""
    time = np.array([row["time"] for row in rows], dtype=float)
    R = np.array([row["R_quantile"] for row in rows], dtype=float)
    R_max = np.array([row["R_max"] for row in rows], dtype=float)
    R_area = np.array([row["R_area"] for row in rows], dtype=float)
    R_circle = np.array([row["R_circle"] for row in rows], dtype=float)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5.5))

    plt.plot(time, R, linewidth=1.8, label=f"R_quantile q={RADIUS_QUANTILE}")

    if SMOOTH_WINDOW > 1:
        R_smooth = np.array([row["R_quantile_smooth"] for row in rows], dtype=float)
        plt.plot(time, R_smooth, linewidth=2.2, label=f"R_quantile smooth, window={SMOOTH_WINDOW}")

    # 辅助曲线：线宽更细，便于比较但不作为主结果。
    plt.plot(time, R_max, linewidth=0.8, alpha=0.45, label="R_max reference")
    plt.plot(time, R_area, linewidth=0.8, alpha=0.45, label="R_area reference")
    plt.plot(time, R_circle, linewidth=0.8, alpha=0.45, label="R_circle reference")

    plt.xlabel(f"Time / {TIME_UNIT}")
    plt.ylabel("Spreading radius / Angstrom")
    plt.title("Spreading radius vs time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def print_parameter_summary():
    """打印参数，方便检查。"""
    print("========== 当前计算参数 ==========")
    print(f"DUMP_FILE = {DUMP_FILE}")
    print(f"OUTPUT_CSV = {OUTPUT_CSV}")
    print(f"OUTPUT_PNG = {OUTPUT_PNG}")
    print(f"DUMP_CONTAINS_ONLY_LIQUID = {DUMP_CONTAINS_ONLY_LIQUID}")
    print(f"LIQUID_TYPES = {LIQUID_TYPES}")
    print(f"DT = {DT}")
    print(f"TIME_UNIT = {TIME_UNIT}")
    print(f"ZERO_TIME_AT_FIRST_FRAME = {ZERO_TIME_AT_FIRST_FRAME}")
    print(f"USE_SUBSTRATE_SURFACE_Z = {USE_SUBSTRATE_SURFACE_Z}")
    print(f"SUBSTRATE_Z_TOP = {SUBSTRATE_Z_TOP}")
    print(f"Z_TOLERANCE_BELOW_SURFACE = {Z_TOLERANCE_BELOW_SURFACE}")
    print(f"LAYER_THICKNESS = {LAYER_THICKNESS}")
    print(f"BOTTOM_PERCENTILE = {BOTTOM_PERCENTILE}")
    print(f"RADIUS_QUANTILE = {RADIUS_QUANTILE}")
    print(f"OUTER_OUTLIER_QUANTILE = {OUTER_OUTLIER_QUANTILE}")
    print(f"BOUNDARY_FIT_INNER_QUANTILE = {BOUNDARY_FIT_INNER_QUANTILE}")
    print(f"BOUNDARY_FIT_OUTER_QUANTILE = {BOUNDARY_FIT_OUTER_QUANTILE}")
    print(f"MIN_CONTACT_ATOMS = {MIN_CONTACT_ATOMS}")
    print(f"UNWRAP_XY = {UNWRAP_XY}")
    print(f"SMOOTH_WINDOW = {SMOOTH_WINDOW}")
    print("==================================\n")


# ============================================================
# 六、主程序
# ============================================================

def main():
    print_parameter_summary()

    dump_path = Path(DUMP_FILE)

    if not dump_path.exists():
        raise FileNotFoundError(
            f"找不到 dump 文件: {dump_path}\n"
            "请确认 DUMP_FILE 路径是否正确。"
        )

    if not (0.0 < RADIUS_QUANTILE <= 1.0):
        raise ValueError("RADIUS_QUANTILE 必须在 0 到 1 之间。")

    if not (0.0 < OUTER_OUTLIER_QUANTILE <= 1.0):
        raise ValueError("OUTER_OUTLIER_QUANTILE 必须在 0 到 1 之间。")

    if not (0.0 < BOUNDARY_FIT_INNER_QUANTILE < 1.0):
        raise ValueError("BOUNDARY_FIT_INNER_QUANTILE 必须在 0 到 1 之间。")

    if not (0.0 < BOUNDARY_FIT_OUTER_QUANTILE <= 1.0):
        raise ValueError("BOUNDARY_FIT_OUTER_QUANTILE 必须在 0 到 1 之间。")

    if BOUNDARY_FIT_INNER_QUANTILE >= BOUNDARY_FIT_OUTER_QUANTILE:
        raise ValueError("BOUNDARY_FIT_INNER_QUANTILE 必须小于 BOUNDARY_FIT_OUTER_QUANTILE。")

    rows = []
    first_timestep = None

    print("开始计算润湿铺展半径...\n")

    for frame_id, frame in enumerate(read_lammps_dump(dump_path)):
        timestep = frame["timestep"]

        if first_timestep is None:
            first_timestep = timestep

        if ZERO_TIME_AT_FIRST_FRAME:
            physical_time = (timestep - first_timestep) * DT
        else:
            physical_time = timestep * DT

        result = compute_radius_for_frame(frame)

        row = {
            "frame": frame_id,
            "timestep": timestep,
            "time": float(physical_time),
            **result,
        }

        rows.append(row)

        if frame_id % 50 == 0:
            print(
                f"frame={frame_id:6d}, "
                f"step={timestep:10d}, "
                f"time={physical_time:12.6g} {TIME_UNIT}, "
                f"R_q={result['R_quantile']:12.6g}, "
                f"R_max={result['R_max']:12.6g}, "
                f"n_contact={result['n_contact_used']:6d}, "
                f"status={result['status']}"
            )

    if len(rows) == 0:
        raise RuntimeError("没有读取到任何帧，请检查 dump 文件格式。")

    if SMOOTH_WINDOW > 1:
        R_values = np.array([row["R_quantile"] for row in rows], dtype=float)
        R_smooth = rolling_mean(R_values, SMOOTH_WINDOW)
        for row, value in zip(rows, R_smooth):
            row["R_quantile_smooth"] = float(value)

    save_csv(rows, OUTPUT_CSV)
    plot_radius(rows, OUTPUT_PNG)

    ok_frames = sum(1 for row in rows if row["status"] == "ok")
    total_frames = len(rows)

    print("\n计算完成！")
    print(f"结果数据已保存: {OUTPUT_CSV}")
    print(f"结果图片已保存: {OUTPUT_PNG}")
    print(f"成功计算帧数: {ok_frames} / {total_frames}")

    if ok_frames < total_frames:
        print("\n注意：部分帧没有成功计算，请检查 CSV 文件中的 status 列。")

    print("\nCSV 中建议重点查看：")
    print("time, R_quantile, R_max, R_area, R_circle, n_contact_used, status")
    print("\n论文主结果建议使用 R_quantile。")
    print("R_max、R_area、R_circle 主要作为辅助验证。")


if __name__ == "__main__":
    main()
