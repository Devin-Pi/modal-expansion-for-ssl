import os
import webrtcvad
import soundfile
import numpy as np
from torch.utils.data import Dataset
from loguru import logger
import librosa

import matplotlib.pyplot as plt

class Parameter:
    """ Random parammeter class.
        Random PARAMETER
    """
    def __init__(self, *args, discrete=False):
        self.discrete = discrete
        if discrete == False:
            if len(args) == 1:
                self.random = False
                self.value = np.array(args[0])
                self.min_value = None
                self.max_value = None
            elif len(args) == 2:
                self.random = True
                self.min_value = np.array(args[0])
                self.max_value = np.array(args[1])
                self.value = None
            else:
                raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
        else:
      		# IF discrete == TRUE diffuse
            self.value_range = args[0]

    def getValue(self):
        if self.discrete == False:
            if self.random:
                return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
            else:
                return self.value
        else:
            idx = np.random.randint(0, len(self.value_range))
            return self.value_range[idx]




class LibriSpeechDataset(Dataset):
	""" Dataset with random LibriSpeech utterances.
	You need to indicate the path to the root of the LibriSpeech dataset in your file system
	and the length of the utterances in seconds.
	The dataset length is equal to the number of chapters in LibriSpeech (585 for train-clean-100 subset)
	but each time you ask for dataset[idx] you get a random segment from that chapter.
	It uses webrtcvad to clean the silences from the LibriSpeech utterances.
	"""

	def _exploreCorpus(self, path, file_extension):
    # return a list including abosolute path for the libspeech file (xxx/xxx/xxx/xxx.flac). The format of the direcory_tress is like {reader:{chapter:xxx/flac},{chapter:xxx/flac}}
		directory_tree = {}
		for item in os.listdir(path):
			if os.path.isdir( os.path.join(path, item) ):
				directory_tree[item] = self._exploreCorpus( os.path.join(path, item), file_extension )
			elif item.split(".")[-1] == file_extension:
				directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
		return directory_tree

	def _cleanSilences(self, s, aggressiveness, return_vad=False):
		self.vad.set_mode(aggressiveness)

		vad_out = np.zeros_like(s) # (76640,)
		vad_frame_len = int(10e-3 * self.fs)  # 0.001s,16samples gives one same vad results -> 10ms, 16khz
		n_vad_frames = len(s) // vad_frame_len # 1000/s,1/0.001s
		for frame_idx in range(n_vad_frames):
			frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
			frame_bytes = (frame * 32767).astype('int16').tobytes()
			vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, self.fs)
		s_clean = s * vad_out

		return (s_clean, vad_out) if return_vad else s_clean

	def __init__(self, path, T, fs, num_source, size=None, return_vad=False, readers_range=None, clean_silence=True):
		self.corpus = self._exploreCorpus(path, 'flac')
		if readers_range is not None:
			for key in list(map(int, self.nChapters.keys())):
				if int(key) < readers_range[0] or int(key) > readers_range[1]:
					del self.corpus[key]

		self.nReaders = len(self.corpus) # the number of the whole list -> 40
		self.nChapters = {reader: len(self.corpus[reader]) for reader in self.corpus.keys()} # the number of the chapters read by each reader, as for the list, its format is like "key:value"
		self.nUtterances = {reader: {
				chapter: len(self.corpus[reader][chapter]) for chapter in self.corpus[reader].keys()
			} for reader in self.corpus.keys()}
		# the number of segment of each chapter of each reader
		self.chapterList = []
		for chapters in list(self.corpus.values()):
			self.chapterList += list(chapters.values())
		# get the all chapter, the len is 97
		# self.readerList = []
		# for reader in self.corpus.keys():
		# 	self.readerList += list(reader)

		self.fs = fs # sampling rate 16khz
		self.T = T # the length of trajectory
		self.num_source = num_source

		self.clean_silence = clean_silence
		self.return_vad = return_vad
		self.vad = webrtcvad.Vad() # init for VAD

		self.sz = len(self.chapterList) if size is None else size

	def __len__(self):
		return self.sz

	def __getitem__(self, idx):
		if idx < 0: idx = len(self) + idx
		while idx >= len(self.chapterList): idx -= len(self.chapterList)

		s_sources = []
		s_clean_sources = []
		vad_out_sources = []
		speakerID_list = []

		for source_idx in range(self.num_source):
			if source_idx==0:
				chapter = self.chapterList[idx]
				utts = list(chapter.keys())
				# print(utts)
				spakerID = utts[0].split('-')[0]

			else:
				idx_othersources = np.random.randint(0, len(self.chapterList))
				chapter = self.chapterList[idx_othersources]
				utts = list(chapter.keys())
				spakerID = utts[0].split('-')[0]
				while spakerID in speakerID_list:
        		# avoid the same reader for different microphones
					idx_othersources = np.random.randint(0, len(self.chapterList))
					chapter = self.chapterList[idx_othersources]
					utts = list(chapter.keys())
					spakerID = utts[0].split('-')[0]

			speakerID_list += [spakerID]

			# Get a random speech segment from the selected chapter
			s = np.array([])
			utt_paths = list(chapter.values())
			n = np.random.randint(0, len(chapter))

			while s.shape[0] < self.T * self.fs:
				utterance, fs = soundfile.read(utt_paths[n])
				# logger.debug(f'fs: {fs}, self.fs: {self.fs}')
				if fs != self.fs:
					# Resample to self.fs
					utterance = librosa.resample(utterance, orig_sr=fs, target_sr=self.fs)
					fs = self.fs
					s = np.concatenate([s, utterance])
					n += 1
					if n >= len(chapter): n=0

				else:
					assert fs == self.fs
					s = np.concatenate([s, utterance])
					n += 1
					if n >= len(chapter): n=0 # assure that the samples are among the chapters
			s = s[0: int(self.T * fs)]
			s -= s.mean() # ? why decrease the mean value

			# Clean silences, it starts with the highest aggressiveness of webrtcvad,
			# but it reduces it if it removes more than the 66% of the samples
			s_clean, vad_out = self._cleanSilences(s, 3, return_vad=True)
			if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
				s_clean, vad_out = self._cleanSilences(s, 2, return_vad=True)
			if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
				s_clean, vad_out = self._cleanSilences(s, 1, return_vad=True)

			s_sources += [s]
			s_clean_sources += [s_clean]
			vad_out_sources += [vad_out]

		s_sources = np.array(s_sources).transpose(1,0) # [76640, 1]
		s_clean_sources = np.array(s_clean_sources).transpose(1,0) # [76640, 1]
		vad_out_sources = np.array(vad_out_sources).transpose(1,0) # [76640]


		# NOTE: plot the waveform & VAD
		# soundfile.write('test.wav', s_sources, self.fs)
		# soundfile.write('test_clean.wav', s_clean_sources, self.fs)
		# plt.plot(s_sources)
		# plt.title('original')
		# plt.savefig('original.png')
		# plt.figure()
		# plt.plot(s_clean_sources)
		# plt.title('clean')
		# plt.savefig('clean.png')
		# plt.figure()
		# plt.plot(vad_out_sources)
		# plt.title('vad')
		# plt.savefig('vad.png')

		if self.clean_silence:
			return (s_clean_sources, vad_out_sources, self.fs) if self.return_vad else s_clean_sources
		else:
			return (s_sources, vad_out_sources, self.fs) if self.return_vad else s_sources


# %% Transform classes
class Segmenting_SRPDNN(object):
	""" Segmenting transform.
	"""
	def __init__(self, K, step, window=None):
		self.K = K
		self.step = step
		if window is None:
			self.w = np.ones(K)
		elif callable(window):
			try: self.w = window(K)
			except: raise Exception('window must be a NumPy window function or a Numpy vector with length K')
		elif len(window) == K:
			self.w = window
		else:
			raise Exception('window must be a NumPy window function or a Numpy vector with length K')

	def __call__(self, x, acoustic_scene):
		# N_mics = x.shape[1]
		N_dims = acoustic_scene.DOA.shape[1]
		num_source = acoustic_scene.DOA.shape[2]
		L = x.shape[0]
		N_w = np.floor(L/self.step - self.K/self.step + 1).astype(int)

		if self.K > L:
			raise Exception('The window size can not be larger than the signal length ({})'.format(L))
		elif self.step > L:
			raise Exception('The window step can not be larger than the signal length ({})'.format(L))

		DOA = []
		for source_idx in range(num_source):
			DOA += [np.append(acoustic_scene.DOA[:,:,source_idx], np.tile(acoustic_scene.DOA[-1,:,source_idx].reshape((1,2)),
				[N_w*self.step+self.K-L, 1]), axis=0)] # Replicate the last known DOA
		DOA = np.array(DOA).transpose(1,2,0)

		shape_DOAw = (N_w, self.K, N_dims) # (nwindow, win_len, naziele)
		strides_DOAw = [self.step*N_dims, N_dims, 1]
		strides_DOAw = [strides_DOAw[i] * DOA.itemsize for i in range(3)]
		acoustic_scene.DOAw = []
		for source_idx in range(num_source):
			DOAw = np.lib.stride_tricks.as_strided(DOA[:,:,source_idx], shape=shape_DOAw, strides=strides_DOAw)
			DOAw = np.ascontiguousarray(DOAw)
			for i in np.flatnonzero(np.abs(np.diff(DOAw[..., 1], axis=1)).max(axis=1) > np.pi):
				DOAw[i, DOAw[i,:,1]<0, 1] += 2*np.pi # Avoid jumping from -pi to pi in a window
			DOAw = np.mean(DOAw, axis=1)
			DOAw[DOAw[:,1]>np.pi, 1] -= 2*np.pi
			acoustic_scene.DOAw += [DOAw]
		acoustic_scene.DOAw = np.array(acoustic_scene.DOAw).transpose(1, 2, 0) # (nsegment,naziele,nsource)

		# Pad and window the VAD if it exists
		if hasattr(acoustic_scene, 'mic_vad'): # (nsample,1)
			vad = acoustic_scene.mic_vad[:, np.newaxis]
			vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

			shape_vadw = (N_w, self.K, 1)
			strides_vadw = [self.step * 1, 1, 1]
			strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]

			acoustic_scene.mic_vad = np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0]

		# Pad and window the VAD if it exists
		if hasattr(acoustic_scene, 'mic_vad_sources'): # (nsample,nsource)
			shape_vadw = (N_w, self.K, 1)

			num_sources = acoustic_scene.mic_vad_sources.shape[1]
			vad_sources = []
			for source_idx in range(num_sources):
				vad = acoustic_scene.mic_vad_sources[:, source_idx:source_idx+1]
				vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

				strides_vadw = [self.step * 1, 1, 1]
				strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]
				vad_sources += [np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0]]

			acoustic_scene.mic_vad_sources = np.array(vad_sources).transpose(1,2,0) # (nsegment, nsample, nsource)

		# Timestamp for each window
		acoustic_scene.tw = np.arange(0, (L-self.K), self.step) / acoustic_scene.fs

		return x, acoustic_scene
