from datasetME import ModelExpansionDataset
from utils.dataset import LibriSpeechDataset
import lightning as l
from omegaconf import DictConfig
import hydra
from utils.data_generation_tools import ensure_dir, save_file, pressure_to_spl
from tqdm import tqdm
from collections import namedtuple
import numpy as np
@hydra.main(config_path="config/", config_name="dataSIMU")
def main(cfg: DictConfig) -> None:

    # Initialize the dataset
    if cfg.DATA_SIMU.TRAIN:
        data_num = cfg.DATA_SIMU.TRAIN_NUM
        l.seed_everything(1848)
        source_path = cfg.DATA_SIMU.SOU_PATH.TRAIN
        save_path = cfg.DATA_SIMU.SAVE_PATH.TRAIN
    elif cfg.DATA_SIMU.TEST:
        data_num = cfg.DATA_SIMU.TEST_NUM
        l.seed_everything(1858)
        source_path = cfg.DATA_SIMU.SOU_PATH.TEST
        save_path = cfg.DATA_SIMU.SAVE_PATH.TEST
    else:
        data_num = cfg.DATA_SIMU.DEV_NUM
        l.seed_everything(1868)
        source_path = cfg.DATA_SIMU.SOU_PATH.DEV
        save_path = cfg.DATA_SIMU.SAVE_PATH.DEV

    sourceDataset = LibriSpeechDataset(
        path= source_path,
        T=5,
        fs=16000,
        num_source=cfg.NS,
        return_vad=True,
        clean_silence=True
    )
    ArraySetup = namedtuple('ArraySetup', 'array_type, orVec, mic_scale, mic_pos, mic_orV, mic_pattern')

    dualch_array_setup = ArraySetup(
    array_type='planar',
    orVec = np.array([0.0, 1.0, 0.0]),
    mic_scale = 1,
    mic_pos = np.array(((
        (-0.04, 0.0, 0.0),
        (0.04, 0.0, 0.0),
        # (0.0, 0.0, 0.04),
        # (0.0, 0.0, -0.04),
    ))),  # 2 microphones with a distance of 0.08m are randomly placed in the room
    mic_orV = None,
    mic_pattern = 'omni'
    )

    dataset = ModelExpansionDataset(
        num_data=data_num,
        num_points=cfg.TRAJ_POINTS,
        min_dist_range=cfg.MIN_DIST_RANGE,
        source_data=sourceDataset,
        num_source_range=cfg.NS,
        num_source_state=cfg.NS_STATE,
        source_state=cfg.S_STATE,
        room_dims_range=cfg.ROOM_DIMS_RANGE,
        # array_setup=cfg.ARRAY_SETUP,
        array_setup=dualch_array_setup,
        array_pos_range=cfg.ARRAY_POS_RANGE,
        noise_data=None,
        SNR=cfg.SNR,
        transform=None,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        device=cfg.DEVICE,
        sound_speed=cfg.SPEED
    )

    save_path = save_path
    ensure_dir(save_path)
    for i in tqdm(range(data_num), desc='Generating Simulation Data'):
        audio_signals, acoustic_environment = dataset[i]
        sig_path = save_path + '/' + str(i) + '.wav'
        acous_path = save_path + '/' + str(i) + '.npz'
        save_file(audio_signals, acoustic_environment, sig_path, acous_path)



if __name__ == "__main__":
    main()