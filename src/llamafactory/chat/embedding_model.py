# Copyright 2024 THUDM and the LlamaFactory team.
#
# This code is inspired by the THUDM's ChatGLM implementation.
# https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence, Union
from numpy import ndarray

from ..extras.misc import torch_gc
from ..hparams import get_infer_args
from sentence_transformers import SentenceTransformer
from torch import Tensor


if TYPE_CHECKING:
    from numpy.typing import NDArray


class EmbeddingModel:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        self.engine: Optional[SentenceTransformer] = None
        
        if not model_args.embedding_model_name_or_path or len(model_args.embedding_model_name_or_path) == 0:
            return
        
        embedding_path = model_args.embedding_model_name_or_path

        self.load_model(embedding_path)
        self.executor = ThreadPoolExecutor(max_workers=1)

    
    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def load_model(self, model_path) -> None:
        if self.loaded:
            raise Exception("Model already installed!")
        try:
            self.engine = SentenceTransformer(model_path, trust_remote_code=True)
        except Exception as e:
            print(e)
            raise e
        print("Embedding Model loaded")

    async def achat(
        self,
        messages: Union[str, List[str]]
    ) -> List[List[float]]:

        result = await asyncio.get_event_loop().run_in_executor(self.executor, self.engine.encode, messages)
        
        if isinstance(result, Tensor):
            result = [result.tolist()]
        elif isinstance(result, list):
            result = [res.tolist() for res in result]
        elif isinstance(result, ndarray):
            result = [res.flatten().tolist() for res in result]
        else:
            raise NotImplementedError("Not implemented")
        
        return result
