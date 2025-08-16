import numpy as np
from utils import gpuME
from loguru import logger
from matplotlib import pyplot as plt
from loguru import logger
from utils.figrue import plot2D, plot2D_two, pressure_to_spl

class AcousticEnvironment(object):
    """
    AcousticEnvironment class.
    """

    def __init__(self,
                 room_dims,
                 source_signal,
                 microphone_pos,
                 trajectory_points,
                 trajectory,
                 trajectory_relative,
                 DOA,
                 device,
                 freqs,
                 time_steps,
                 fs,
                 source_state,
                 t
                 ):
        """
        Constructor.
        """
        self.room_dims = room_dims
        self.source_signal = source_signal

        self.microphone_pos = microphone_pos
        self.trajectory_points = trajectory_points
        self.trajectory = trajectory
        self.trajectory_relative = trajectory_relative
        self.DOA = DOA
        self.freqs = freqs
        self.device = device
        self.time_steps = time_steps
        self.fs = fs
        self.source_state = source_state
        self.t = t

    def data_generation(self):
        """
        Data generation.
        """
        num_sources = self.trajectory.shape[-1]
        RIRs_sources = []
        mic_signals_sources = []

        for source_i in range(num_sources):
            # logger.debug(f'microphone_pos: {self.microphone_pos}')
            RIRs = gpuME.RIRs_generation(
                freqs=self.freqs,
                room_dims=self.room_dims,
                microphone_pos=self.microphone_pos,
                trajectory_points=self.trajectory_points,
                device=self.device,
                source_state=self.source_state,
                # sound_speed=self.sound_speed
            )

            mic_signals = gpuME.MicrophoneSignalGeneration(
                source_signal=self.source_signal[:, 0],
                RIRs=RIRs,
                device=self.device,
                time_steps=self.time_steps,
                fs=self.fs
            )
            mic_signals = mic_signals[0:len(self.t), :]

            RIRs_sources += [RIRs]
            mic_signals_sources += [mic_signals]

        RIRs_sources = np.array(RIRs_sources) # [1, 50, 2, 496] [num_sources, num_points, num_mics, num_samples]
        mic_signals_sources = np.array(mic_signals_sources) # [1, 80000, 2]
        mic_signals = mic_signals_sources.sum(axis=0) # [80000, 2] num_samples, num_mics
        # draw the RIR and SPL
        # logger.debug(f'mic pos: {self.microphone_pos}')
        # logger.debug(f'source pos: {self.trajectory_points}')
        # for source_i in range(num_sources):
        #     for mic_i in range(2):
        #         plt.figure()
        #         plt.plot(RIRs_sources[source_i, 0, mic_i, :].flatten())
        #         plt.title('RIRs_sources')
        #         plt.show()
        #         # rtf calculation and plot
        #         RIR = RIRs_sources[source_i, 0, mic_i, :]
        #         logger.debug(f"RIR: {RIR.shape}")
        #         rtf = np.fft.fft(RIR)
        #         rtf = np.abs(rtf)
        #         spl = [pressure_to_spl(p) for p in rtf]
        #         plt.title('spl!')
        #         plt.plot(spl)
        #         plt.show()
        #         #spl calculation and plot
        #         RIR = RIRs_sources[source_i, 0, mic_i, :]
        #         logger.debug(f"RIR: {RIR.shape}")
        #         rir_fft = np.fft.fft(RIR)
        #         mag = np.abs(rir_fft)
        #         logger.debug(f"mag: {mag.shape}")
        #         spl = [pressure_to_spl(p) for p in mag]
        #         logger.debug(f"SPL: {spl[0:10]}")
        #         plot2D(np.linspace(10, 1000, 496), spl, 'SPL vs Frequency', 'Frequency (Hz)', 'SPL (dB)')
        #         plt.figure()
        #         plt.xlim(0, 1000)
        #         plt.plot(spl)
        #         plt.title('SPL')
        #         plt.show()


        if hasattr(self, 'source_vad'):
            # NOTE: VAD of separate sensor signals of source
            self.mic_vad_sources = []
            for source_i in range(num_sources):
                vad_sources = gpuME.MicrophoneSignalGeneration(
                    source_signal=self.source_vad[:, source_i],
                    RIRs=RIRs_sources[source_i,...],
                    device=self.device,
                    time_steps= self.time_steps,
                    fs=self.fs
                )
                vad_sources = vad_sources[0:len(self.t), :].mean(axis=1) > vad_sources[0:len(self.t), :].max() * 1e-3 # threshold for voice activity is 0.001  [80000, ]
                # logger.debug(vad_sources.sum())
                self.mic_vad_sources += [vad_sources]
            self.mic_vad_sources = np.array(self.mic_vad_sources) # [1, 80000]
            self.mic_vad = np.sum(self.mic_vad_sources, axis=0) > 0.5
            assert self.mic_vad.all() == self.mic_vad_sources.all()

        return mic_signals