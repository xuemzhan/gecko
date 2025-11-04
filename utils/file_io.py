# agno/utils/file_io.py

"""
文件输入/输出工具模块

该模块提供了一系列用于处理文件读写、序列化和反序列化的辅助函数。
支持的格式和操作包括：
- YAML 文件的读写。
- Python 对象的 pickle 序列化与反序列化。
- 音频文件的写入，支持 base64 编码的音频和原始 PCM 数据。
"""

import base64
import os
import pickle
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Union

import logging
logger = logging.getLogger(__name__)


# --- YAML 文件操作 ---

def read_yaml_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    安全地读取一个 YAML 文件并将其内容解析为 Python 字典。

    Args:
        file_path: YAML 文件路径（字符串或 Path 对象）。

    Returns:
        解析后的字典，若失败则返回 None。
    """
    path = Path(file_path)
    if not (path.exists() and path.is_file()):
        logger.info(f"YAML 文件未找到: {path}")
        return None

    try:
        import yaml
        logger.info(f"正在读取 YAML 文件: {path}")
        content = path.read_text(encoding='utf-8')
        data = yaml.load(content, Loader=yaml.SafeLoader)  # 显式使用 SafeLoader
        if isinstance(data, dict):
            return data
        else:
            logger.error(f"YAML 文件内容不是字典: {path}")
            return None
    except ImportError:
        logger.error("PyYAML 未安装。请运行: pip install pyyaml")
        return None
    except Exception as e:
        logger.error(f"读取或解析 YAML 文件失败 {path}: {e}")
        return None


def write_yaml_file(
    file_path: Union[str, Path],
    data: Dict[str, Any],
    **kwargs: Any
) -> None:
    """
    将字典安全写入 YAML 文件。

    Args:
        file_path: 目标文件路径。
        data: 要写入的数据。
        **kwargs: 传递给 yaml.safe_dump 的参数（如 sort_keys=False）。
    """
    path = Path(file_path)
    try:
        import yaml
        logger.info(f"正在写入 YAML 文件: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        yaml_str = yaml.safe_dump(data, **kwargs)
        path.write_text(yaml_str, encoding='utf-8')
    except ImportError:
        logger.error("PyYAML 未安装。请运行: pip install pyyaml")
    except Exception as e:
        logger.error(f"写入 YAML 文件失败 {path}: {e}")


# --- Pickle 对象序列化 ---

def pickle_object_to_file(obj: Any, file_path: Union[str, Path]) -> None:
    """
    将 Python 对象序列化到文件（使用 pickle）。

    ⚠️ 警告: Pickle 不安全！仅用于可信数据源。
    """
    path = Path(file_path)
    try:
        logger.info(f"正在 pickle 对象到文件: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f)  # nosec B301
    except Exception as e:
        logger.error(f"Pickle 写入失败 {path}: {e}")


def unpickle_object_from_file(file_path: Union[str, Path]) -> Optional[Any]:
    """
    从文件反序列化 Python 对象（使用 pickle）。

    ⚠️ 警告: Pickle 不安全！仅用于可信数据源。
    """
    path = Path(file_path)
    if not (path.exists() and path.is_file()):
        logger.info(f"Pickle 文件未找到: {path}")
        return None

    try:
        logger.info(f"正在从文件 unpickle 对象: {path}")
        with path.open("rb") as f:
            return pickle.load(f)  # nosec B301
    except Exception as e:
        logger.error(f"Unpickle 失败 {path}: {e}")
        return None


# --- 音频文件写入 ---

def write_base64_audio_to_file(
    encoded_audio: str,
    file_path: Union[str, Path]
) -> None:
    """
    将 base64 编码的音频数据解码并写入文件。
    """
    path = Path(file_path)
    try:
        logger.info(f"正在将 Base64 音频写入文件: {path}")
        audio_bytes = base64.b64decode(encoded_audio)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio_bytes)
        logger.info(f"音频文件已保存至 {path}")
    except Exception as e:
        logger.error(f"写入 Base64 音频失败 {path}: {e}")


def write_wav_audio_to_file(
    file_path: Union[str, Path],
    pcm_data: bytes,
    channels: int = 1,
    sample_rate: int = 24000,
    sample_width: int = 2
) -> None:
    """
    将原始 PCM 数据写入 WAV 文件。

    Args:
        file_path: 输出路径。
        pcm_data: 原始音频字节数据。
        channels: 通道数（1=mono, 2=stereo）。
        sample_rate: 采样率（Hz）。
        sample_width: 采样宽度（字节），通常为 1、2 或 4。
    """
    if not pcm_data:
        logger.warning("PCM 数据为空，跳过写入。")
        return

    path = Path(file_path)
    try:
        logger.info(f"正在将 PCM 数据写入 WAV 文件: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        logger.info(f"WAV 文件已保存至 {path}")
    except Exception as e:
        logger.error(f"写入 WAV 文件失败 {path}: {e}")


# --- 测试代码 ---
if __name__ == "__main__":
    import logging
    from dataclasses import dataclass 
    import shutil
    import math
    import struct

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("--- 正在运行 agno/utils/file_io.py 的测试代码 ---")

    temp_dir = Path("./temp_test_files")
    temp_dir.mkdir(exist_ok=True)

    # 1. YAML 测试
    print("\n[1] 测试 YAML 文件操作:")
    yaml_path = temp_dir / "test.yaml"
    yaml_data = {"user": "test", "permissions": ["read", "write"], "config": {"retries": 3}}
    write_yaml_file(yaml_path, yaml_data, sort_keys=False)
    read_data = read_yaml_file(yaml_path)
    print(f"  写入/读取一致? {yaml_data == read_data}")

    # 2. Pickle 测试
    print("\n[2] 测试 Pickle 对象序列化:")

    @dataclass
    class MyObject:
        name: str
        value: int

    obj = MyObject("pickle_test", 123)
    pickle_path = temp_dir / "test.pkl"
    pickle_object_to_file(obj, pickle_path)
    restored = unpickle_object_from_file(pickle_path)
    print(f"  对象一致? {obj == restored}")

    # 3. 音频测试
    print("\n[3] 测试音频文件写入:")

    # Base64 音频（静音 WAV 片段）
    dummy_wav_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABgAAABkYXRhAAAAAA=="
    b64_path = temp_dir / "test_base64.wav"
    write_base64_audio_to_file(dummy_wav_b64, b64_path)
    print(f"  Base64 音频大小: {b64_path.stat().st_size} 字节")

    # PCM 正弦波
    sample_rate = 8000
    duration = 1
    freq = 440
    pcm = b"".join(
        struct.pack("<h", int(32767 * math.sin(2 * math.pi * freq * i / sample_rate)))
        for i in range(duration * sample_rate)
    )
    pcm_path = temp_dir / "test_pcm.wav"
    write_wav_audio_to_file(pcm_path, pcm, sample_rate=sample_rate)
    print(f"  PCM WAV 大小: {pcm_path.stat().st_size} 字节")

    # 清理
    shutil.rmtree(temp_dir)
    print(f"\n临时目录 '{temp_dir}' 已清理。")
    print("\n--- 测试结束 ---")