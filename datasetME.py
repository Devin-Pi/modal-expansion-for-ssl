import torch
from  torch.utils.data import Dataset

import numpy as np

from utils.data_generation_tools import generate_random_source_pos, generate_random_room_params, gMD, gNS, gRMAP, cart2sph_
from utils.AcousticEnvironment import AcousticEnvironment
from loguru import logger

class ModelExpansionDataset(Dataset):
    """Modal Expansion Dataset

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        num_data,
        num_points,
        min_dist_range,
        source_data,
        num_source_range,
        num_source_state,
        source_state,
        room_dims_range,
        array_setup,
        array_pos_range,
        noise_data,
        SNR,
        transform,
        fmin,
        fmax,
        device,
        sound_speed=343.0,
    ):
        self.device = device
        self.num_data = num_data
        self.num_points = num_points
        self.source_data = source_data
        self.num_source_state = num_source_state
        self.num_source_range = num_source_range
        self.source_state = source_state
        self.room_dims_range = room_dims_range
        self.array_setup = array_setup
        self.array_pos_range = array_pos_range
        self.noise_data = noise_data
        self.SNR = SNR
        self.min_dist_range = min_dist_range
        self.sound_speed = sound_speed
        self.transform = transform
        self.freqs = torch.arange(fmin, fmax+1, 2)
        self.fs = source_data.fs


    def __len__(self):
        return len(self.num_data)

    def __getitem__(self, idx):
        if idx < 0: idx = len(self) + idx

        acoustic_scene = self.RandomAcousticEnvironment(idx)
        audio_signals = acoustic_scene.data_generation()

        if self.transform is not None:
            for trans in self.transform:
                audio_signals, acoustic_scene = trans(audio_signals, acoustic_scene)

        return audio_signals, acoustic_scene

    def RandomAcousticEnvironment(self, idx):
        source_data, vad_sources, source_fs = self.source_data[idx]

        num_source = gNS(
            num_source_range=self.num_source_range,
            mode=self.num_source_state
            )

        room_dims = generate_random_room_params(
            room_dims_range=self.room_dims_range
            )

        source_pos = generate_random_source_pos(
            room_dims=room_dims
            )

        mic_array_pos = gRMAP(
            microphone_array_range=self.array_pos_range,
            room_dims=room_dims,
        )

        min_dist =gMD(self.min_dist_range)

        microphone_pos = mic_array_pos + self.array_setup.mic_pos
        # logger.debug(f"Microphone positions: {microphone_pos.shape}")
        # logger.debug(f"Microphone positions: {microphone_pos}")
        # microphone_pos = np.array(((0.23575455, 0.0712623, 0.18715587),
        #                           (0.31575455, 0.0712623, 0.18715587)))
        # source_pos = np.array((0.36107247, 0.22794664, 0.18715587))
        # TODO: NOISE SIGNALS SHOULD BE CONSIDERED

        # NOTE: Trajectory of the source

        source_pos_min = np.array([0.0, 0.0, 0.0])
        source_pos_max = np.array(room_dims)

        if self.array_setup.array_type == "planar":
            if np.sum(self.array_setup.orVec) > 0:
                source_pos_min[np.nonzero(self.array_setup.orVec)] = mic_array_pos[np.nonzero(self.array_setup.orVec)]
            else:
                source_pos_max[np.nonzero(self.array_setup.orVec)] = mic_array_pos[np.nonzero(self.array_setup.orVec)]

        source_pos_min[np.nonzero(self.array_setup.orVec)] += min_dist

        time_steps = np.arange(0, self.num_points) * len(source_data) / self.fs / self.num_points

        t = np.arange(0, len(source_data)) / self.fs
        # NOTE: Initialize the trajectory points & DOA
        trajectory_points = np.zeros((self.num_points, 3, num_source))
        trajectory = np.zeros((len(t), 3, num_source))
        trajectory_relative = np.zeros((len(t), 3, num_source))
        DOA = np.zeros((len(t), 2, num_source))

        for source_i in range(num_source):
            if self.source_state == "static":
                # source_pos = np.array((0.36107247, 0.22794664, 0.18715587))
                source_pos = source_pos_min + np.random.random(3) * (source_pos_max - source_pos_min)
                trajectory_points[:, :, source_i] = np.tile(source_pos, (self.num_points, 1))

            elif self.source_state == "moving":
                source_pos_start = source_pos_min + np.random.random(3) * (source_pos_max - source_pos_min)
                source_pos_end = source_pos_min + np.random.random(3) * (source_pos_max - source_pos_min)

                A_max = np.min(np.stack((source_pos_start - source_pos_min,
                                         source_pos_max - source_pos_start,
                                         source_pos_end - source_pos_min,
                                            source_pos_max - source_pos_end)), axis=0)
                A = np.random.random(3) * np.minimum(A_max, 1)

                w = 2*np.pi / self.num_points * np.random.random(3) *2
                trajectory_points[:, :, source_i] = np.array([np.linspace(i, j, self.num_points) for i, j in zip(source_pos_start, source_pos_end)]).transpose()

                trajectory_points[:, :, source_i] += A * np.sin(w * np.arange(self.num_points)[:,np.newaxis])

            else:
                raise ValueError("Invalid source state")

            # set the ele = pi/2
            trajectory_points[:,2,:] = microphone_pos[0,2]

            # Interpolate the trajectory points -> [80000, 3, 1]
            trajectory[:, :, source_i] = np.array([np.interp(t, time_steps, trajectory_points[:, i, source_i]) for i in range(3)]).transpose()

            # Calculate the DOA
            # [r, phi, theta][:,1:3] - > [phi, theta]
            DOA[:, :, source_i] = cart2sph_(trajectory[:,:,source_i] - mic_array_pos)[:, 1:3]
            # NOTE: Calculate the relative trajectory
            trajectory_relative[:,:,source_i] = trajectory[:,:,source_i] - mic_array_pos

        acoustic_environment = AcousticEnvironment(
            room_dims=room_dims,
            source_signal=source_data,
            microphone_pos=microphone_pos,
            trajectory_points=trajectory_points,
            trajectory=trajectory,
            trajectory_relative=trajectory_relative,
            DOA = DOA,
            freqs=self.freqs,
            time_steps=time_steps,
            device = self.device,
            fs = self.fs,
            source_state = self.source_state,
            t = t
        )

        acoustic_environment.source_vad = vad_sources[:, 0:num_source]


        return acoustic_environment