import os
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import torch
from tqdm import tqdm
import soundfile
from loguru import logger
import pickle

def calculate_pressure_gpu(frequency,
                         room_dims,
                         source_pos,
                         observation_pos,
                         device,
                         sound_speed=343*torch.sqrt(torch.tensor(1+0.01j))):
    Lx, Ly, Lz = room_dims
    x0, y0, z0 = source_pos
    x, y, z = observation_pos
    V = Lx * Ly * Lz
    Power = 0.0055
    density = 1.2

    # 将所有标量转换为GPU tensor
    omega = 2 * np.pi * frequency
    omega = torch.tensor(omega, device=device, dtype=torch.complex64)
    k = omega / sound_speed
    k_square = k ** 2
    Q = torch.tensor(0.0001, device=device, dtype=torch.complex64)

    # 生成网格数据
    mnl_array = torch.arange(50, device=device)
    m_grid, n_grid, l_grid = torch.meshgrid(mnl_array, mnl_array, mnl_array, indexing='ij')
    # logger.info(f'm_grid: {m_grid}, n_grid: {n_grid}, l_grid: {l_grid}')
    # # 条件计算
    con_m = torch.where(m_grid == 0, torch.tensor(1., device=device), torch.tensor(2., device=device))
    con_n = torch.where(n_grid == 0, torch.tensor(1., device=device), torch.tensor(2., device=device))
    con_l = torch.where(l_grid == 0, torch.tensor(1., device=device), torch.tensor(2., device=device))
    # logger.debug(f'con_m: {con_m}, con_n: {con_n}, con_l: {con_l}')
    Con_ = con_m * con_n * con_l / V

    # 计算k_mnl_square
    k_mnl_square = ((m_grid * np.pi / Lx) ** 2 +
                    (n_grid * np.pi / Ly) ** 2 +
                    (l_grid * np.pi / Lz) ** 2)

    # 计算Psi函数
    Psi_mnl_source = (torch.cos(m_grid * np.pi * x0 / Lx) *
                      torch.cos(n_grid * np.pi * y0 / Ly) *
                      torch.cos(l_grid * np.pi * z0 / Lz))

    Psi_mnl_obs = (torch.cos(m_grid * np.pi * x / Lx) *
                   torch.cos(n_grid * np.pi * y / Ly) *
                   torch.cos(l_grid * np.pi * z / Lz))

    # 计算最终结果
    k_error = k_square - k_mnl_square
    A_mnl = Con_ * Psi_mnl_obs * Psi_mnl_source / k_error
    G = torch.sum(A_mnl)

    P = -1j * Q * omega * density * G
    return P.cpu().numpy()

def calculate_pressure(frequency,
                       room_dims,
                       source_pos,
                       observation_pos,
                       sound_speed=343*np.sqrt(1+0.01*1j)):
    Lx, Ly, Lz = room_dims
    x0, y0, z0 = source_pos
    x, y, z = observation_pos
    V = Lx * Ly * Lz
    Power = 0.0055  # 声源功率
    density = 1.2  # 空气密度 (kg/m^3)

    omega = 2 * np.pi * frequency
    k = omega / sound_speed
    k_square = k ** 2
    # Q = np.sqrt((8 * np.pi * Power) / (density * sound_speed * k_square))
    Q = 0.0001

    # logger.debug(f'Frequency: {frequency}')

    G = 0
    mnl_array = np.arange(50)
    m_grid, n_grid, l_grid = np.meshgrid(mnl_array, mnl_array, mnl_array, indexing='ij')

    con_m = np.where(m_grid == 0, 1, 2)
    con_n = np.where(n_grid == 0, 1, 2)
    con_l = np.where(l_grid == 0, 1, 2)
    Con_ = con_m * con_n * con_l / V

    k_mnl_square = ((m_grid * np.pi / Lx) ** 2 +
                    (n_grid * np.pi / Ly) ** 2 +
                    (l_grid * np.pi / Lz) ** 2)

    Psi_mnl_source = (np.cos(m_grid * np.pi * x0 / Lx) *
                      np.cos(n_grid * np.pi * y0 / Ly) *
                      np.cos(l_grid * np.pi * z0 / Lz))

    Psi_mnl_obs = (np.cos(m_grid * np.pi * x / Lx) *
                   np.cos(n_grid * np.pi * y / Ly) *
                   np.cos(l_grid * np.pi * z / Lz))

    k_error = k_square - k_mnl_square
    A_mnl = Con_ * Psi_mnl_obs * Psi_mnl_source / k_error
    G = np.sum(A_mnl)

    P = -1j * Q * omega * density * G
    return P
    # return np.abs(P)


def pressure_to_spl(pressure, reference_pressure=20e-6):
    return 20 * np.log10(pressure / reference_pressure)


def impulse_response(pressure, device):
    pressure_ = torch.tensor(pressure, device=device)
    p_ir = torch.fft.ifft(pressure_)
    return p_ir


def save_complex_to_txt(complex,save_path):
    p_real = np.real(complex)
    p_imag = np.imag(complex)
    p_txt = np.stack((p_real, p_imag), axis=1)
    basename = os.path.basename(save_path)
    save_path_ = os.path.join(os.path.dirname(save_path), f'{basename}.txt')
    # logger.debug(f'Saving to {save_path_}')
    np.savetxt(save_path_, p_txt)


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def cart2sph(cart):
    xy2 = cart[0]**2 + cart[1]**2
    sph = np.zeros_like(cart)

    sph[0] = np.sqrt(xy2 + cart[2]**2) # distance
    sph[1] = np.arctan2(cart[2], np.sqrt(xy2)) # Elevation angle defined from Z-axis down
    sph[2] = np.arctan2(cart[0], cart[1]) # azimuth angle

    return sph


    # sph_degrees[0] = sph[0] # distance
    # sph_degrees[1] = np.degrees(sph[1]) # Elevation angle defined from Z-axis down
    # sph_degrees[2] = np.degrees(sph[2]) # azimuth angle

    # return sph, sph_degrees # [r, phi, theta]

# generate random room dimensions
def generate_random_room_params(room_dims_range):
    room_dims = []
    room_dims_min = room_dims_range[0]
    room_dims_max = room_dims_range[1]
    # logger.debug(f'len(room_dims_range): {len(room_dims_range[0])}')
    for i in range(len(room_dims_range[0])):
        dim_min = room_dims_min[i]
        dim_max = room_dims_max[i]
        room_dims.append(dim_min + np.random.random() * (dim_max - dim_min))
    return room_dims




def generate_random_source_pos(room_dims):
    """
    根据房间维度随机生成声源位置
    Args:
        room_dims: [length, width, height] 房间尺寸
    Returns:
        source_pos: [x, y, z] 声源位置
    """
    # NOTE: The minimum distance from the wall
    min_distance_from_wall = 5e-3

    source_pos = []
    source_pos_min = np.array([0.0,0.0,0.0]) + min_distance_from_wall
    source_pos_max = room_dims

    source_pos = source_pos_min + np.random.random(3) * (source_pos_max - source_pos_min)
    # NOTE: Keep the source at the center of the room
    # source_pos[2] = 0.5 * room_dims[2]
    return source_pos

def generate_random_microphone_pos(room_dims, source_pos):
    """
    生成随机的观测点位置
    Args:
        room_dims: [length, width, height] 房间尺寸
        source_pos: [x, y, z] 声源位置坐标
    Returns:
        (x, y, z): 观测点位置坐标
    """
    # NOTE: The minimum distance from the wall
    min_distance = 5e-3 # 与声源的最小距离（米）
    max_attempts = 100  # 最大尝试次数
    min_x_offset = 0.05

    source_x, source_y, source_z = source_pos

    for _ in range(max_attempts):
        # NOTE: Generate random observation position
        observation_pos_min = source_pos + np.array([min_x_offset, 0, 0])
        observation_pos_max = room_dims
        observation_pos = observation_pos_min + np.random.random(3) * (observation_pos_max - observation_pos_min)
        observation_pos[2] = source_z

        # 计算与声源的距离
        distance = np.sqrt(((observation_pos- source_pos)**2).sum())
        # 检查是否满足距离条件
        if distance >= min_distance:
            # logger.debug(f"Generated observation position: ({x:.2f}, {y:.2f}, {z:.2f})")
            # logger.debug(f"Distance to source: {distance:.2f}m")
            return observation_pos

    # 如果多次尝试都失败，则使用简单的后备方案
    logger.warning("Failed to find suitable observation position, using fallback")

    # 后备方案：在声源前方min_x_offset处放置观测点
    x = min(source_x + min_x_offset, room_dims[0] - 0.1)  # 留出一点边距
    y = min(source_y, room_dims[1] - 0.1)
    z = min(source_z, room_dims[2] - 0.1)
    observation_pos = np.array([x, y, z])
    return observation_pos


def process_single_position(
    observation_pos,
    freqs,
    room_dims,
    source_pos,
    device,
    ):
    # x0, y0, z0 = source_pos
    # if x < x0:
    #     pbar.update(1)
    #     return
    x, y, z = observation_pos
    observation_pos_L = (x, y, z)
    observation_pos_R = (x, y+0.08, z)

    pressures_L = []
    pressures_R = []

    # the pbars for the frequencies
    freq_pbar = tqdm(freqs, leave=False, desc=f'Processing frequencies for position ({x:.2f}, {y:.2f}, {z:.2f})')

    for freq in freq_pbar:
        pressure_L = calculate_pressure_gpu(freq, room_dims, source_pos,
                                             observation_pos_L, device)
        pressure_R = calculate_pressure_gpu(freq, room_dims, source_pos,
                                             observation_pos_R, device)
        pressures_L.append(pressure_L)
        pressures_R.append(pressure_R)

    ap = np.vstack(pressures_L, pressures_R)

    return ap


def cart2sph_(cart):
    xy2 = cart[:,0]**2 + cart[:,1]**2
    sph = np.zeros_like(cart)
    # NOTE: the acrctan2 is difference from with mine
    sph[:, 0] = np.sqrt(xy2 + cart[:, 2]**2) # distance
    sph[:, 1] = np.arctan2(np.sqrt(xy2), cart[:, 2]) # Elevation angle defined from Z-axis down
    sph[:, 2] = np.arctan2(cart[:, 1], cart[:, 0]) # azimuth angle

    return sph

def gMD(min_distance_range):
    """
    Generate the Minimum Distance from the Wall
    Args:
        min_distance_range: [min, max] The range of the minimum distance from the wall
    Returns:
        min_distance: The minimum distance from the wall
    """
    min_distance = np.random.rand() * (min_distance_range[1] - min_distance_range[0]) + min_distance_range[0]
    return min_distance


def gRMAP(microphone_array_range, room_dims):

    range_min = microphone_array_range[0]
    range_max = microphone_array_range[1]

    min_ = np.array(range_min) * room_dims
    max_ = np.array(range_max) * room_dims

    array_pos = np.random.rand(3) * (max_ - min_) + min_

    return array_pos



def gNS(num_source_range, mode):
    """
    Generate the Number of Sources
    Args:
        num_source_range: [min, max] The range of the number of sources
        mode: 'fixed' or 'random'
    Returns:
        num_source: The number of sources
    """
    # NOTE: The minimum number of sources is 1

    if mode == 'fixed':
        num_source = num_source_range
    elif mode == 'random':
        num_source = np.random.randint(num_source_range[0], num_source_range[1] + 1)
    return num_source


# NOTE: To read & load the Simulation Dataset
def save_file(mic_signal, acoustic_scene, sig_path, acous_path):

    if sig_path is not None:
        soundfile.write(sig_path, mic_signal, acoustic_scene.fs)

    if acous_path is not None:
        file = open(acous_path,'wb')
        file.write(pickle.dumps(acoustic_scene.__dict__))
        file.close()

def load_file(acoustic_scene, sig_path, acous_path):

    if sig_path is not None:
        mic_signal, fs = soundfile.read(sig_path)

    if acous_path is not None:
        file = open(acous_path,'rb')
        dataPickle = file.read()
        file.close()
        acoustic_scene.__dict__ = pickle.loads(dataPickle)

    if (sig_path is not None) & (acous_path is not None):
        return mic_signal, acoustic_scene
    elif (sig_path is not None) & (acous_path is None):
        return mic_signal
    elif (sig_path is None) & (acous_path is not None):
        return acoustic_scene