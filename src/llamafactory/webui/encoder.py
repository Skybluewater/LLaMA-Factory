# Copyright 2024 the LlamaFactory team.
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

import json
import os
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Sequence, Tuple

from numpy.typing import NDArray

from ..data import Role
from ..extras.constants import PEFT_METHODS
from ..extras.misc import torch_gc
from ..extras.packages import is_gradio_available
from .common import QUANTIZATION_BITS, get_save_dir
from .locales import ALERTS

from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

if TYPE_CHECKING:
    from ..chat import BaseEngine
    from .manager import Manager

if is_gradio_available():
    import gradio as gr


class EncoderModel(ABC):
    SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer. Limit your answer in 500 words"""

    def __init__(self, manager: "Manager", lazy_init: bool = True) -> None:
        self.manager = manager
        self.engine: Optional[SentenceTransformer] = None
        self.data = None
        if not lazy_init:  # read arguments from command line
            super().__init__()
    
    @property
    def loaded(self) -> bool:
        return self.engine is not None and self.data is not None

    def load_model(self, data) -> Generator[str, None, None]:
        if self.engine is not None:
            error = ALERTS["err_exists"]["en"]
        try:
            get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
            lang, embedding_path = get("top.lang"), get("rag.embedding_path")
            self.engine = SentenceTransformer(embedding_path, trust_remote_code=True)
        except Exception as e:
            print(e)
            yield str(e)
            return
        print("Embedding Model loaded")
        yield ALERTS["info_loaded"][lang]

    def load_data(self, data) -> Generator[str, None, None]:
        if self.engine is None:
            yield ALERTS["err_no_model"]["en"]
            return
        try:
            get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
            lang, rag_path = get("top.lang"), get("rag.rag_path")
            dataset = load_dataset("csv", data_files=rag_path, trust_remote_code=True)
        except Exception as e:
            print(e)
            yield str(e)
            return

        data = dataset["train"]

        # Compute embeddings for each entry in the dataset
        def compute_embeddings(batch):
            batch["embeddings"] = self.engine.encode(batch["text"], convert_to_tensor=True).tolist()
            return batch

        # Apply the function to compute embeddings
        data = data.map(compute_embeddings, batched=True, batch_size=32)

        self.data = data.add_faiss_index("embeddings") # column name that has the embeddings of the dataset

        print("Dataset loaded and indexed")
        print("Loading status: " + str(self.loaded))
        yield ALERTS["info_loaded"][lang]
    
    def search(self, query: str, k: int = 3):
        """a function that embeds a new query and returns the most probable results"""
        embedded_query = self.engine.encode(query) # embed new query
        scores, retrieved_examples = self.data.get_nearest_examples( # retrieve results
            "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
            k=k # get only top k results
        )
        
        filtered = [[score, example] for score, example in zip(scores, retrieved_examples['text']) if score <= 300]
        scores = [score for score, _ in filtered]
        retrieved_examples = [example for _, example in filtered]
        return scores, retrieved_examples
    
    @staticmethod
    def format_prompt(prompt, retrieved_documents, k):
        """using the retrieved documents we will prompt the model to generate our responses"""
        k = min(k, len(retrieved_documents))
        if k == 0:
            return ""
        
        PROMPT = f"Command:{EncoderModel.SYS_PROMPT}\nQuestion:{prompt}\nContext:"
        for idx in range(k):
            PROMPT+= f"{retrieved_documents[idx]}\n"
        return PROMPT

    def unload_model(self, data) -> Generator[str, None, None]:
        lang = data[self.manager.get_elem_by_id("top.lang")]

        # if self.demo_mode:
        #     gr.Warning(ALERTS["err_demo"][lang])
        #     yield ALERTS["err_demo"][lang]
        #     return

        yield ALERTS["info_unloading"][lang]
        self.engine = None
        yield ALERTS["info_unloaded"][lang]

    def unload_data(self, data) -> Generator[str, None, None]:
        lang = data[self.manager.get_elem_by_id("top.lang")]

        # if self.demo_mode:
        #     gr.Warning(ALERTS["err_demo"][lang])
        #     yield ALERTS["err_demo"][lang]
        #     return

        yield ALERTS["info_unloading"][lang]
        self.data = None
        yield ALERTS["info_unloaded"][lang]


    def stream(
        self,
        chatbot: List[List[Optional[str]]],
        messages: Sequence[Dict[str, str]],
        system: str,
        tools: str,
        image: Optional[NDArray],
        max_new_tokens: int,
        top_p: float,
        temperature: float,
    ) -> Generator[Tuple[List[List[Optional[str]]], List[Dict[str, str]]], None, None]:
        chatbot[-1][1] = ""
        response = ""
        for new_text in self.stream_chat(
            messages, system, tools, image, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature
        ):
            response += new_text
            if tools:
                result = self.engine.template.extract_tool(response)
            else:
                result = response

            if isinstance(result, list):
                tool_calls = [{"name": tool[0], "arguments": json.loads(tool[1])} for tool in result]
                tool_calls = json.dumps(tool_calls, indent=4, ensure_ascii=False)
                output_messages = messages + [{"role": Role.FUNCTION.value, "content": tool_calls}]
                bot_text = "```json\n" + tool_calls + "\n```"
            else:
                output_messages = messages + [{"role": Role.ASSISTANT.value, "content": result}]
                bot_text = result

            chatbot[-1][1] = bot_text
            yield chatbot, output_messages


if __name__ == "__main__":
    encoder = EncoderModel(None, lazy_init=True)
    encoder.load_model(None)
    encoder.load_data(None)
    scores, retrieved_documents = encoder.search("数字表演是什么")
    print(encoder.format_prompt("数字表演是什么？", retrieved_documents, 3))
    print(scores)
    print(retrieved_documents)