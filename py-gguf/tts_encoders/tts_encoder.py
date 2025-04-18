import gguf
from pathlib import Path
import torch
import torch.nn as nn
import logging
import abc
import os
from os.path import dirname
import numpy as np

logging.basicConfig(level=logging.INFO)


class TTSEncoder(abc.ABC):
    """
    Abstract class for converting TTS models to the GGUF file format.
    The purpose of this class is to standardize the tensor encoding and model configuration pattern for preparing
    a GGUF file from a pytorch TTS model.
    """
    def __init__(self, model_path: Path | str, architecture: str):
        """
        :param Path or str model_path: the path to save the gguf tensors and configuration.
        :param str architecture: the model archeticture to assign to the GGUF file.
        """
        self.path = model_path if isinstance(model_path, Path) else Path(model_path)
        # Configure GGUF Writer (path is specified at write time in accordance with existing patterns in llama.cpp)
        self.gguf_writer = gguf.GGUFWriter(path=None, arch=architecture)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.architecture = architecture

    def prepare_tensors(self):
        """
        Implementations of TTSEncoder are expected to define this function. This function should be responsible
        for assign all relevant model tensors to the GGUF file.
        """
        pass

    def prepare_metadata(self):
        """
        Implementations of TTSEncoder are expected to define this function. This function should be responsible
        for determining and assigning general model configuration to the GGUF file.
        """
        pass

    def set_tensor(self, name: str, tensor: torch.Tensor | nn.Parameter | np.ndarray,
                    dtype: any = np.float32, gguf_dtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.F32):
        """
        Adds a torch.Tensor, torch.nn.Parameter or a NumPy array to the GGUF writer with the assigned
        named and GGUF quantization type.
        """
        data = tensor

        # if the tensor is already a NumPy array then we can just add it to the GGUF file writer.
        if not isinstance(tensor, np.ndarray):
            # if this is a parameter then we need to get the data field.
            if isinstance(tensor, nn.Parameter):
                data = tensor.data

            # detach the tensor from the gradients, assign to the cpu, and convert to numpy.
            data = data.detach().cpu().contiguous().numpy()

        data = data.astype(dtype)
        self.gguf_writer.add_tensor(name, data, raw_dtype=gguf_dtype)
        self.logger.debug(f"Added tensor '{name}' with shape {data.shape} and type {gguf_dtype}.")

    def write(self):
        """
        This function encapsulates the core behavior for preparing and writing a GGUF file.
        """
        try:
            self.logger.info(f"Preparing tensors for GGUF file, {self.path}.")
            self.prepare_tensors()
            self.logger.info(f"Preparing configuration for GGUF file, {self.path}.")
            self.prepare_metadata()
            self.logger.info(f"Beginning to write to GGUF file, {self.path}.")
            os.makedirs(dirname(self.path), exist_ok=True)
            self.gguf_writer.write_header_to_file(path=self.path)
            self.gguf_writer.write_kv_data_to_file()
            self.gguf_writer.write_tensors_to_file(progress=True)
        except Exception as e:
            self.logger.exception(
                f"Failed with exception, {e}, Did not complete GGUF conversion attempting to obtain obtain tokenizer at path or repo: '{self.repo_id}'"
            )
            raise e
        finally:
            self.gguf_writer.close()
            self.logger.info(f"Finishing GGUF converstion process for file, {self.path}.")
