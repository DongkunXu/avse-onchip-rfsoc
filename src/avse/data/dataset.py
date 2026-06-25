import os
import wave
import torch
import torchaudio
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm


class AVSEDataset(Dataset):
    """
    流式窗口数据集 - 固定窗口大小，无需填充
    每个样本都是固定长度的窗口，模拟流式处理
    """

    def __init__(self, root_dir: str, split: str, config):
        self.root_dir = root_dir
        self.split = split
        self.config = config

        # 动态窗口参数（从配置文件读取）
        self.window_duration = getattr(config.data, 'window_duration', 0.64)  # 默认640ms
        self.overlap_ratio = getattr(config.data, 'overlap_ratio', 0.20)      # 默认20%重叠
        self.hop_duration = self.window_duration * (1 - self.overlap_ratio)   # 动态计算hop

        self.window_samples = int(self.window_duration * config.audio.sample_rate)
        self.hop_samples = int(self.hop_duration * config.audio.sample_rate)

        self.window_frames = int(self.window_duration * config.video.fps)
        self.hop_frames = int(self.hop_duration * config.video.fps)

        # 音频路径
        self.scenes_dir = os.path.join(root_dir, split, 'scenes')
        # 预处理过的 96x96 灰度 uint8 视频 (由 LRS3/preprocess_video_96.py 生成)
        # 形如 <split>/96/<scene_id>.npy, shape [T, 96, 96], dtype uint8
        # 必填：缺失则抛错而非隐式回退到 mp4 解码（避免悄悄走慢速路径）
        self.video_npy_dir = os.path.join(root_dir, split, '96')
        if not os.path.isdir(self.video_npy_dir):
            raise FileNotFoundError(
                f"Preprocessed video dir not found: {self.video_npy_dir}\n"
                f"  Run: python LRS3/preprocess_video_96.py --root <data_root> --splits {split}"
            )
        
        # JSON metadata paths follow the selected dataset root.
        self.metadata_paths = {
            'train': os.path.join(root_dir, 'scenes.train.json'),
            'dev': os.path.join(root_dir, 'scenes.dev.json')
        }
        
        # Data filtering mode
        self.data_mode = getattr(config.data, 'data_mode', 'full')
        
        # Load JSON metadata if available
        self.scene_metadata = self._load_json_metadata()

        print(f"\n🚀 Initializing {split} dataset...")
        print(f"   Mode: {self.data_mode}")
        print(f"   Workers: {self.config.data.init_workers}")
        print(f"   Window: {self.window_duration*1000:.0f}ms ({self.window_samples} samples, {self.window_frames} frames)")
        print(f"   Hop: {self.hop_duration*1000:.0f}ms ({self.hop_samples} samples, {self.hop_frames} frames)")
        print(f"   Overlap: {self.overlap_ratio*100:.0f}%")
        print(f"   Video source: preprocessed .npy ({self.video_npy_dir})")
        
        start_time = time.time()
        self.windows = self._create_windows_parallel()
        end_time = time.time()
        
        print(f"\n✅ Dataset ready: {len(self.windows)} windows ({end_time - start_time:.2f}s)")
        print("-" * 50)
    
    def _load_json_metadata(self) -> Dict[str, Dict]:
        """Load and parse JSON metadata for scene filtering"""
        if self.split not in self.metadata_paths:
            print(f"   ⚠️  No metadata path configured for split '{self.split}', using full dataset")
            return {}
        
        json_path = self.metadata_paths[self.split]
        if not os.path.exists(json_path):
            print(f"   ⚠️  JSON metadata file not found: {json_path}, using full dataset")
            return {}
        
        try:
            print(f"   📖 Loading metadata from: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            # Convert list to dict keyed by scene_id
            metadata_dict = {}
            for item in metadata_list:
                scene_id = item.get('scene', '')
                metadata_dict[scene_id] = item
            
            print(f"   ✅ Loaded metadata for {len(metadata_dict)} scenes")
            return metadata_dict
            
        except Exception as e:
            print(f"   ⚠️  Error loading JSON metadata: {e}, using full dataset")
            return {}
    
    def _filter_scenes_by_interference_type(self, scene_ids: List[str]) -> List[str]:
        """Filter scenes based on interference type for noise_only mode"""
        if not self.scene_metadata:
            return scene_ids
        
        original_count = len(scene_ids)
        filtered_scenes = []
        
        for scene_id in scene_ids:
            if scene_id in self.scene_metadata:
                interferer_info = self.scene_metadata[scene_id].get('interferer', {})
                interference_type = interferer_info.get('type', '')
                
                if self.data_mode == 'noise_only':
                    if interference_type == 'noise':
                        filtered_scenes.append(scene_id)
                else:
                    filtered_scenes.append(scene_id)
            else:
                if self.data_mode == 'full':
                    filtered_scenes.append(scene_id)
        
        filtered_count = len(filtered_scenes)
        
        if self.data_mode == 'noise_only' and self.scene_metadata:
            noise_count = sum(1 for sid in scene_ids if sid in self.scene_metadata and 
                            self.scene_metadata[sid].get('interferer', {}).get('type') == 'noise')
            speech_count = sum(1 for sid in scene_ids if sid in self.scene_metadata and 
                             self.scene_metadata[sid].get('interferer', {}).get('type') == 'speech')
            print(f"\n🔍 Filtering scenes by interference type:")
            print(f"   Available: {noise_count} noise + {speech_count} speech")
            print(f"   Filtered: {filtered_count}/{original_count} scenes ({filtered_count/original_count*100:.1f}%)")
            print(f"   Using: {filtered_count} noise-only scenes")
        
        return filtered_scenes

    def _validate_scene_files(self, scene_id: str) -> bool:
        """并行验证单个场景的文件完整性"""
        required_audio = [
            f"{scene_id}_mixed.wav",
            f"{scene_id}_target.wav",
            f"{scene_id}_interferer.wav",
        ]

        for filename in required_audio:
            if not os.path.exists(os.path.join(self.scenes_dir, filename)):
                return False
        # 视频统一从 <split>/96/<scene_id>.npy 读取，不再要求原始 silent.mp4
        if not os.path.exists(os.path.join(self.video_npy_dir, f"{scene_id}.npy")):
            return False
        return True
    
    def _get_scene_ids_parallel(self) -> List[str]:
        """并行获取有效场景ID"""
        # 快速获取所有target文件
        target_files = glob.glob(os.path.join(self.scenes_dir, "*_target.wav"))
        scene_ids_candidate = [
            os.path.basename(f).replace("_target.wav", "") for f in target_files
        ]
        
        print(f"\n📁 Found {len(scene_ids_candidate)} potential scenes, validating files...")
        
        # Validate file completeness in parallel
        valid_scene_ids = []
        max_workers = min(self.config.data.init_workers, len(scene_ids_candidate))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_scene = {
                executor.submit(self._validate_scene_files, scene_id): scene_id 
                for scene_id in scene_ids_candidate
            }
            
            for future in as_completed(future_to_scene):
                scene_id = future_to_scene[future]
                try:
                    if future.result():
                        valid_scene_ids.append(scene_id)
                except Exception:
                    pass  # Skip invalid scenes silently
        
        print(f"   ✅ Validated {len(valid_scene_ids)} complete scenes")
        
        # Apply data filtering based on metadata
        if self.data_mode == 'noise_only' and self.scene_metadata:
            filtered_scene_ids = self._filter_scenes_by_interference_type(valid_scene_ids)
            return sorted(filtered_scene_ids)
        else:
            return sorted(valid_scene_ids)

    def _get_audio_length(self, audio_path: str) -> int:
        """获取音频长度（样本数）"""
        try:
            samples, sample_rate = self._get_audio_metadata(audio_path)
            if sample_rate != self.config.audio.sample_rate:
                samples = int(samples * self.config.audio.sample_rate / sample_rate)
            return samples
        except Exception as e:
            print("ERROR AUDIO:", audio_path, e)
            return 0

    def _get_audio_metadata(self, audio_path: str) -> Tuple[int, int]:
        """获取音频元信息，优先使用 soundfile (Windows 友好)"""
        # 优先使用 soundfile (稳定，跨平台兼容)
        try:
            info = sf.info(audio_path)
            return info.frames, info.samplerate
        except Exception:
            pass

        # 备用：torchaudio (可能触发 torchcodec 问题)
        if hasattr(torchaudio, 'info'):
            try:
                info = torchaudio.info(audio_path)
                return info.num_frames, info.sample_rate
            except Exception:
                pass

        # 最后备用：Python 标准库
        with wave.open(audio_path, 'rb') as wav_file:
            return wav_file.getnframes(), wav_file.getframerate()
    
    def _get_scene_audio_lengths(self, scene_id: str) -> Tuple[str, int, int]:
        """并行获取单个场景的音频长度"""
        target_path = os.path.join(self.scenes_dir, f"{scene_id}_target.wav")
        interferer_path = os.path.join(self.scenes_dir, f"{scene_id}_interferer.wav")
        
        target_length = self._get_audio_length(target_path)
        interferer_length = self._get_audio_length(interferer_path)
        
        return scene_id, target_length, interferer_length
    
    def _get_all_audio_lengths_parallel(self, scene_ids: List[str]) -> Dict[str, Tuple[int, int]]:
        """并行获取所有场景的音频长度"""
        print(f"\n🎵 Getting audio lengths for {len(scene_ids)} scenes...")
        
        audio_lengths = {}
        max_workers = min(self.config.data.init_workers, len(scene_ids))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_scene = {
                executor.submit(self._get_scene_audio_lengths, scene_id): scene_id 
                for scene_id in scene_ids
            }
            
            pbar = tqdm(total=len(scene_ids), desc="Getting audio lengths", 
                       unit="scene", leave=True, ncols=80)
            
            for future in as_completed(future_to_scene):
                try:
                    scene_id, target_length, interferer_length = future.result()
                    audio_lengths[scene_id] = (target_length, interferer_length)
                except Exception:
                    pass  # Skip failed scenes silently
                pbar.update(1)
            
            pbar.close()
        
        print(f"   ✅ Got audio lengths for {len(audio_lengths)} scenes")
        return audio_lengths

    def _create_scene_windows(self, scene_data: Tuple[str, int, int]) -> List[Dict]:
        """为单个场景创建窗口"""
        scene_id, target_length, interferer_length = scene_data
        min_length = min(target_length, interferer_length)
        
        if min_length < self.window_samples:
            return []  # 跳过太短的样本
        
        windows = []
        start_sample = 0
        while start_sample + self.window_samples <= min_length:
            windows.append({
                'scene_id': scene_id,
                'start_sample': start_sample,
                'end_sample': start_sample + self.window_samples,
                'start_frame': int(start_sample * self.config.video.fps / self.config.audio.sample_rate),
                'end_frame': int(
                    (start_sample + self.window_samples) * self.config.video.fps / self.config.audio.sample_rate)
            })
            start_sample += self.hop_samples
        
        return windows
    
    def _create_windows_parallel(self) -> List[Dict]:
        """并行创建固定大小的窗口列表"""
        # 1. 并行获取有效场景ID
        scene_ids = self._get_scene_ids_parallel()
        
        if not scene_ids:
            print("⚠️  No valid scenes found!")
            return []
        
        # 2. 并行获取所有音频长度
        audio_lengths = self._get_all_audio_lengths_parallel(scene_ids)
        
        # 3. 准备场景数据
        scene_data_list = []
        for scene_id in scene_ids:
            if scene_id in audio_lengths:
                target_length, interferer_length = audio_lengths[scene_id]
                scene_data_list.append((scene_id, target_length, interferer_length))
        
        print(f"\n🪟 Creating windows for {len(scene_data_list)} scenes...")
        
        # Create windows in parallel
        all_windows = []
        max_workers = min(self.config.data.init_workers, len(scene_data_list))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_scene = {
                executor.submit(self._create_scene_windows, scene_data): scene_data[0]
                for scene_data in scene_data_list
            }
            
            pbar = tqdm(total=len(scene_data_list), desc="Creating windows", 
                       unit="scene", leave=True, ncols=80)
            
            for future in as_completed(future_to_scene):
                try:
                    windows = future.result()
                    all_windows.extend(windows)
                except Exception:
                    pass  # Skip failed scenes silently
                pbar.update(1)
            
            pbar.close()
        
        print(f"   ✅ Created {len(all_windows)} total windows")
        return all_windows

    def _load_audio_window(self, audio_path: str, start_sample: int, num_samples: int) -> torch.Tensor:
        """加载固定长度的音频窗口 (使用 soundfile，Windows 友好)"""
        try:
            # 获取原始采样率
            _, original_sr = self._get_audio_metadata(audio_path)

            if original_sr != self.config.audio.sample_rate:
                # 需要重采样：计算原始采样率下的起始位置和长度
                original_start = int(start_sample * original_sr / self.config.audio.sample_rate)
                original_length = int(num_samples * original_sr / self.config.audio.sample_rate)

                # 使用 soundfile 分段加载
                waveform_np, sr = sf.read(
                    audio_path,
                    start=original_start,
                    stop=original_start + original_length,
                    dtype='float32'
                )

                # 转换为 Tensor: [T] or [T, C] -> [C, T]
                waveform = torch.from_numpy(waveform_np).float()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
                else:
                    waveform = waveform.T  # [T, C] -> [C, T]

                # 重采样 (使用 torchaudio.transforms，不触发 torchcodec)
                resampler = torchaudio.transforms.Resample(sr, self.config.audio.sample_rate)
                waveform = resampler(waveform)
                del resampler
            else:
                # 无需重采样：直接加载目标采样率的数据
                waveform_np, sr = sf.read(
                    audio_path,
                    start=start_sample,
                    stop=start_sample + num_samples,
                    dtype='float32'
                )

                # 转换为 Tensor: [T] or [T, C] -> [C, T]
                waveform = torch.from_numpy(waveform_np).float()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
                else:
                    waveform = waveform.T  # [T, C] -> [C, T]

            # 转单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 确保长度正确（可能因为重采样有轻微差异）
            if waveform.shape[1] != num_samples:
                if waveform.shape[1] < num_samples:
                    padding = num_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                else:
                    waveform = waveform[:, :num_samples]

            return waveform

        except Exception as e:
            print(f"  ⚠️  Audio load failed: {audio_path} | {e}")
            return torch.zeros(1, num_samples)

    def _load_video_window(self, scene_id: str, start_frame: int, num_frames: int) -> torch.Tensor:
        """从预处理 .npy mmap 切出固定长度窗口。

        数据由 LRS3/preprocess_video_96.py 离线生成: <split>/96/<scene_id>.npy
        shape [T, H, W] uint8，对齐离线 BGR->GRAY + INTER_AREA resize 流水线。

        缺失文件时直接抛 FileNotFoundError；不足帧数时重复最后一帧 padding。
        """
        npy_path = os.path.join(self.video_npy_dir, f"{scene_id}.npy")
        arr = np.load(npy_path, mmap_mode='r')  # [T, H, W] uint8

        total = arr.shape[0]
        end = start_frame + num_frames

        if end <= total:
            window = np.array(arr[start_frame:end])  # copy out of mmap
        else:
            avail = max(0, total - start_frame)
            pad_len = num_frames - avail
            if avail > 0:
                head = np.array(arr[start_frame:total])
            else:
                head = np.empty((0, *arr.shape[1:]), dtype=arr.dtype)
            if total > 0:
                last = np.array(arr[total - 1:total])  # [1, H, W]
                pad = np.repeat(last, pad_len, axis=0)
            else:
                pad = np.zeros((pad_len, *tuple(self.config.video.input_size)), dtype=np.uint8)
            window = np.concatenate([head, pad], axis=0) if avail > 0 else pad

        return torch.from_numpy(window).float() / 255.0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        scene_id = window['scene_id']

        try:
            # 音频路径 (视频路径在 _load_video_window 内部根据 .npy / mp4 自动选择)
            mixed_path = os.path.join(self.scenes_dir, f"{scene_id}_mixed.wav")
            target_path = os.path.join(self.scenes_dir, f"{scene_id}_target.wav")

            # 加载固定大小的窗口
            mixed_audio = self._load_audio_window(
                mixed_path, window['start_sample'], self.window_samples
            )
            target_audio = self._load_audio_window(
                target_path, window['start_sample'], self.window_samples
            )
            video_frames = self._load_video_window(
                scene_id, window['start_frame'], self.window_frames
            )

            # 联合归一化: 用 mixed 的 max 作为公共 scale，保持 mixed 与 target 的幅度关系
            max_val = mixed_audio.abs().max()
            if max_val > 0:
                scale = 0.8 / max_val
                mixed_audio = mixed_audio * scale
                target_audio = target_audio * scale

            return {
                'video_frames': video_frames,  # [T, H, W] 固定大小，由 config.video.input_size 决定
                'mixed_audio': mixed_audio,  # [1, T_audio] 固定大小
                'target_audio': target_audio,  # [1, T_audio] 固定大小
                'scene_id': scene_id,
                'window_start': window['start_sample']
            }

        except Exception as e:
            print(f"  ⚠️  Sample load failed: scene_id={scene_id} | {e}")
            return {
                'video_frames': torch.zeros(self.window_frames, *self.config.video.input_size),
                'mixed_audio': torch.zeros(1, self.window_samples),
                'target_audio': torch.zeros(1, self.window_samples),
                'scene_id': scene_id,
                'window_start': window['start_sample']
            }
