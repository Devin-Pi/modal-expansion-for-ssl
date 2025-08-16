import torch
import numpy as np
from utils.data_generation_tools import impulse_response
from utils.figrue import plot2D, plot2D_semilogx, plot2D_two, pressure_to_spl
from loguru import logger
import matplotlib.pyplot as plt
from tqdm import tqdm


def RIR_single_generation(freqs,
                    room_dims,
                    source_pos,
                    observation_pos,
                    device,
                    sound_speed=343*torch.sqrt(torch.tensor(1+0.01j))):
    Lx, Ly, Lz = room_dims
    x0, y0, z0 = source_pos
    # transform all scalars to GPU tensor
    Lx = torch.tensor(Lx, device=device, dtype=torch.float32)
    Ly = torch.tensor(Ly, device=device, dtype=torch.float32)
    Lz = torch.tensor(Lz, device=device, dtype=torch.float32)
    x0 = torch.tensor(x0, device=device, dtype=torch.float32)
    y0 = torch.tensor(y0, device=device, dtype=torch.float32)
    z0 = torch.tensor(z0, device=device, dtype=torch.float32)
    V = Lx * Ly * Lz
    Power = 0.0055
    density = 1.2

    MIC_RIRs = []
    # 将所有标量转换为GPU tensor
    for mic in observation_pos:
        x, y, z = mic
        pressure = []
        for frequency in freqs:
            # logger.debug(f'Frequency: {frequency}')
            omega = 2 * np.pi * frequency
            omega = torch.tensor(omega, device=device, dtype=torch.complex64)
            k = omega / sound_speed
            k_square = k ** 2
            Q = torch.tensor(0.0001, device=device)

            # 生成网格数据
            mnl_array = torch.arange(50, device=device)
            m_grid, n_grid, l_grid = torch.meshgrid(mnl_array, mnl_array, mnl_array, indexing='ij')
            # logger.info(f'm_grid: {m_grid}, n_grid: {n_grid}, l_grid: {l_grid}')
            #  CONSTANTS Calculation
            con_m = torch.where(m_grid == 0, torch.tensor(1., device=device), torch.tensor(2., device=device))
            con_n = torch.where(n_grid == 0, torch.tensor(1., device=device), torch.tensor(2., device=device))
            con_l = torch.where(l_grid == 0, torch.tensor(1., device=device), torch.tensor(2., device=device))
            # logger.debug(f'con_m: {con_m}, con_n: {con_n}, con_l: {con_l}')
            Con_ = con_m * con_n * con_l / V

            # 计算k_mnl_square
            k_mnl_square = ((m_grid * torch.pi / Lx) ** 2 +
                    (n_grid * torch.pi / Ly) ** 2 +
                    (l_grid * torch.pi / Lz) ** 2)

            # 计算Psi函数
            # logger.debug(f'm_grid device: {m_grid.device}')

            Psi_mnl_source = (torch.cos(m_grid * torch.pi * x0 / Lx) *
                      torch.cos(n_grid * torch.pi * y0 / Ly) *
                      torch.cos(l_grid * torch.pi * z0 / Lz))

            Psi_mnl_obs = (torch.cos(m_grid * torch.pi * x / Lx) *
                   torch.cos(n_grid * torch.pi * y / Ly) *
                   torch.cos(l_grid * torch.pi * z / Lz))

            # 计算最终结果
            k_error = k_square - k_mnl_square
            A_mnl = Con_ * Psi_mnl_obs * Psi_mnl_source / k_error
            G = torch.sum(A_mnl)

            P = -1j * Q * omega * density * G

            pressure.append(P)

        rir = impulse_response(pressure, device=device)
        rir = rir.cpu().numpy()
        MIC_RIRs.append(rir)
        # NOTE: debug by plotting
        # plot2D(np.arange(0, len(rir)), rir, 'IR from Python', 'Time', 'Amplitude')
        # plt.savefig('/workspaces/_debug/' + f'IR_L_{x}_{y}_{z}.png')
        # spl calculation and plot

        # pre_ = [i.cpu().numpy() for i in pressure]

        # mag = np.abs(pre_)
        # logger.debug(f"mag: {mag.shape}")
        # spl = [pressure_to_spl(p) for p in mag]
        # logger.debug(f"SPL: {spl[0:10]}")
        # plot2D(np.linspace(10, 1000, 496), spl, 'SPL vs Frequency', 'Frequency (Hz)', 'SPL (dB)')
        # plt.savefig('/workspaces/_debug/' + f'RTF_{x}_{y}_{z}.png')
    # NOTE: debug by plotting
    # plot2D_two(np.arange(0, len(rir)), MIC_RIRs[0], np.arange(0, len(rir)), MIC_RIRs[1], 'IR from Python', 'Time', 'Amplitude')
    # plt.savefig('/workspaces/_debug/' + f'IR_{x}_{y}_{z}.png')
    return MIC_RIRs


def RIRs_generation(
    freqs,
    room_dims,
    source_state,
    microphone_pos,
    trajectory_points,
    device,
    sound_speed=343*torch.sqrt(torch.tensor(1+0.01j))
    ):
    # TODO: Implement this function for generating RIRs based on the Trajectory Points.
    RIRs = []
    i = 0
    if source_state == 'static':
        source_pos = trajectory_points[0]
        RIR = RIR_single_generation(
            freqs = freqs,
            room_dims=room_dims,
            source_pos=source_pos,
            observation_pos=microphone_pos,
            device=device,
            sound_speed = sound_speed
            )
        RIR  = np.array(RIR)
        RIRs = np.tile(RIR, (trajectory_points.shape[0], 1, 1))
    else:
        for source_pos in trajectory_points:
            RIR = RIR_single_generation(
            freqs = freqs,
            room_dims=room_dims,
            source_pos=source_pos,
            observation_pos=microphone_pos,
            device=device,
            sound_speed = sound_speed
            )
            # logger.debug(f'idx: {i}')
            RIRs.append(RIR)
            i+=1
        RIRs = np.array(RIRs) # [50, 2, 496] -> [num_points, num_mics, len_RIR]

    return RIRs


def MicrophoneSignalGeneration(
    source_signal,
    RIRs,
    device,
    time_steps=None,
    fs=None,

    ):
    # source_signal = source_signal.squeeze(1)
    num_samples = len(source_signal)
    num_points = RIRs.shape[0]
    num_mics = RIRs.shape[1]
    len_RIR = RIRs.shape[2]

    assert time_steps is None or fs is not None, "fs must be indicated for custom time_steps"
    assert time_steps is None or time_steps[0] == 0, "The first timestamp must be 0"
    if time_steps is None:
        fs = num_samples / num_points
        time_steps = np.arange(num_points)
    # win_ini: shape 51 [0,1600, ... ,80000]
    win_ini = np.append((time_steps*fs).astype(int), num_samples)
    win_len = np.diff(win_ini) # win_len: 1600
    segments = np.zeros((num_points, win_len.max())) # shape [50, 1600]
    # source_signal: [80000, ] -> segments [50, 1600]
    for n in range(num_points):
        segments[n, 0:win_len[n]] = source_signal[win_ini[n]:win_ini[n+1]]
    # segments = segments.astype('float32', order='C', copy=False)
    # convolution = cuda_convolutions_time_domain(segments, RIRs, device=device)
    convolution = cuda_convolutions_freq_domain(segments, RIRs, device=device)

    filtered_signal = np.zeros((num_samples+len_RIR-1, num_mics)) # [80495, 2]
    for m in range(num_mics):
        for n in range(num_points):
            filtered_signal[win_ini[n]:win_ini[n+1]+len_RIR-1, m] += convolution[n, m, 0:win_len[n]+len_RIR-1]
    return filtered_signal

def vad_generator(vad, ir):

    left_channel, right_channel = ir
    vad = vad.squeeze(1)
    vad_sources_L = np.convolve(vad, left_channel.real, mode='full')
    vad_sources_R = np.convolve(vad, right_channel.real, mode='full')
    vad_sources = np.stack((vad_sources_L, vad_sources_R), axis=-1)
    vad_sources = vad_sources.mean(axis=1) > vad_sources.max() * 1e-3
    # plot vad_sources
    # plt.figure(12,12)
    # plt.plot(vad_sources_L)
    # plt.plot(vad_sources_R)
    # plt.plot(vad_sources)
    # plt.show()
    # vad_sources_ = np.sum(vad_sources)>0.5
    return vad_sources

def cuda_convolutions_time_domain(
    source_segments,  # shape: [Num_source_points, segment_len]
    RIR,             # shape: [Num_source_points, M_rcv, RIR_len],
    device,
    ):             # return: [M_src, M_rcv, conv_len]

    M_src, segment_len = source_segments.shape
    M_src, M_rcv, RIR_len = RIR.shape
    conv_len = segment_len + RIR_len - 1

    # 初始化输出数组
    result = np.zeros((M_src, M_rcv, conv_len)) # [50, 2, 2095]
    # print("Max imaginary part in source:", np.abs(source_segments.imag).max())
    # print("Max imaginary part in RIR:", np.abs(RIR.imag).max())
    # 对每个源和接收器进行卷积
    for rcv in range(M_rcv):
        for src in range(M_src):
            # 直接使用numpy的卷积函数
            result[src, rcv] = np.convolve(
                source_segments[src],
                RIR[src, rcv],
                mode='full'
            )

    return result


def pow2roundup(x):
    return 1 if x == 0 else 2**np.ceil(np.log2(x)).astype(int)

def cuda_convolutions_freq_domain(source_segments, RIR, device=None):
    """
    Parameters:
        source_segments: numpy array of shape (M_src, segment_len)
        RIR: numpy array of shape (M_src, M_rcv, RIR_len), can be complex
    Returns:
        convolved_segments: numpy array of shape (M_src, M_rcv, conv_len)
    """
    # 确保RIR使用复数类型
    RIR = RIR.astype(np.complex64)
    source_segments = source_segments.astype(np.complex64)

    # Get dimensions
    M_src, segment_len = source_segments.shape
    M_src, M_rcv, RIR_len = RIR.shape

    # Calculate FFT length
    N_fft = pow2roundup(segment_len + RIR_len - 1)

    # Zero padding for source segments
    padded_signal = np.zeros((M_src, N_fft), dtype=np.complex64)
    padded_signal[:, :segment_len] = source_segments

    # Zero padding for RIR (使用复数类型)
    padded_RIR = np.zeros((M_src, M_rcv, N_fft), dtype=np.complex64)
    padded_RIR[:, :, :RIR_len] = RIR

    # Perform FFT
    signal_freq = np.fft.fft(padded_signal, axis=1)  # [M_src, N_fft//2 + 1]

    # 对于复数RIR，使用fft而不是rfft
    RIR_freq = np.fft.fft(padded_RIR, axis=2)       # [M_src, M_rcv, N_fft]
    # 只保留正频率部分以匹配signal_freq的维度
    # RIR_freq = RIR_freq[:, :, :N_fft//2 + 1]

    # Expand dimensions for broadcasting
    signal_freq = signal_freq[:, np.newaxis, :]      # [M_src, 1, N_fft//2 + 1]

    # Multiply in frequency domain
    result_freq = signal_freq * RIR_freq

    # Inverse FFT
    result = np.fft.irfft(result_freq, n=N_fft, axis=2)

    # Extract valid convolution length
    conv_len = segment_len + RIR_len - 1
    convolved_segments = result[:, :, :conv_len]

    return convolved_segments



class GPURIRConvolution:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def _pow2roundup(self, x: int) -> int:
        """计算大于x的最小2的幂"""
        return 1 << (x - 1).bit_length()

    def cuda_convolutions(self,
                         source_segments: np.ndarray,  # shape: [M_src, segment_len]
                         RIR: np.ndarray,             # shape: [M_src, M_rcv, RIR_len]
                         ) -> np.ndarray:             # return: [M_src, M_rcv, conv_len]
        """
        使用GPU加速的卷积运算

        参数:
            source_segments: 源信号段, shape [M_src, segment_len]
            RIR: 房间冲激响应, shape [M_src, M_rcv, RIR_len]

        返回:
            卷积结果, shape [M_src, M_rcv, conv_len]
        """
        # 获取维度信息
        M_src, segment_len = source_segments.shape
        M_src, M_rcv, RIR_len = RIR.shape

        # 计算FFT长度
        N_fft = self._pow2roundup(segment_len + RIR_len - 1)

        # 将数据转移到GPU并进行零填充
        d_signal = torch.zeros((M_src, N_fft), dtype=torch.float32, device=self.device)
        d_signal[:, :segment_len] = torch.from_numpy(source_segments)

        d_RIR = torch.zeros((M_src, M_rcv, N_fft), dtype=torch.float32, device=self.device)
        d_RIR[:, :, :RIR_len] = torch.from_numpy(RIR)

        # 执行FFT
        d_signal_freq = torch.fft.rfft(d_signal, dim=1)  # [M_src, N_fft//2 + 1]
        d_RIR_freq = torch.fft.rfft(d_RIR, dim=2)       # [M_src, M_rcv, N_fft//2 + 1]

        # 扩展维度以便广播
        d_signal_freq = d_signal_freq.unsqueeze(1)  # [M_src, 1, N_fft//2 + 1]

        # 频域乘法
        d_result_freq = d_signal_freq * d_RIR_freq

        # 执行IFFT并归一化
        d_result = torch.fft.irfft(d_result_freq, n=N_fft, dim=2)

        # 提取有效长度
        conv_len = segment_len + RIR_len - 1
        result = d_result[:, :, :conv_len].cpu().numpy()

        return result

    def process_batch(self,
                     source_segments: np.ndarray,
                     RIR: np.ndarray,
                     batch_size: int = 32
                     ) -> np.ndarray:
        """
        批处理版本，用于处理大规模数据

        参数:
            source_segments: 源信号段
            RIR: 房间冲激响应
            batch_size: 批大小

        返回:
            卷积结果
        """
        M_src = source_segments.shape[0]
        results = []

        for i in range(0, M_src, batch_size):
            batch_end = min(i + batch_size, M_src)
            batch_result = self.cuda_convolutions(
                source_segments[i:batch_end],
                RIR[i:batch_end]
            )
            results.append(batch_result)

        return np.concatenate(results, axis=0)

def _conv(segments, RIRs):
    # check the shape of the segments
    assert segments.ndim == 2, "segments must be 2D"
    assert RIRs.ndim == 3, "RIRs must be 3D"
    assert segments.shape[0] == RIRs.shape[0], "The number of segments must be equal to the number of RIRs"

    num_source_points = segments.shape[0]
    len_segment = segments.shape[1]
    num_mics = RIRs.shape[1]
    len_RIR = RIRs.shape[2]



    left_ir, right_ir = RIRs
    audio = audio.squeeze(1)


    convolved_left = np.convolve(audio, left_ir.real, mode='full')
    convolved_right = np.convolve(audio, right_ir.real, mode='full')
    return np.stack((convolved_left, convolved_right), axis=-1)