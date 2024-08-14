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

from typing import TYPE_CHECKING, Dict

from ...extras.packages import is_gradio_available
from ..common import save_rag_config, save_embedding_config


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


from ..variables import embedding_model


def create_rag_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        embedding_path = gr.Textbox(scale=3)
        load_embedding_btn = gr.Button()
        unload_embedding_btn = gr.Button()

    with gr.Row():
        rag_path = gr.Textbox(scale=3)
        load_rag_btn = gr.Button()
        unload_rag_btn = gr.Button()

    info_box = gr.Textbox(show_label=False, interactive=False)

    elem_dict.update(
        dict(
            embedding_path=embedding_path,
            load_embedding_btn=load_embedding_btn,
            unload_embedding_btn=unload_embedding_btn,
            rag_path=rag_path,
            load_rag_btn=load_rag_btn,
            unload_rag_btn=unload_rag_btn,
            info_box=info_box,
        )
    )
    
    input_elems.update({embedding_path, rag_path})

    # embedding_path.input(save_embedding_config, inputs=input_elems, queue=False)
    # rag_path.input(save_rag_config, inputs=input_elems, queue=False)

    load_embedding_btn.click(engine.embedding_model.load_model, input_elems, [info_box])
    # .then(
    #     engine.embedding_model.load_data, input_elems, None
    # )

    load_rag_btn.click(engine.embedding_model.load_data, input_elems, [info_box])

    unload_embedding_btn.click(engine.embedding_model.unload_model, input_elems, [info_box])
    unload_rag_btn.click(engine.embedding_model.unload_data, input_elems, [info_box])

    # load_btn.click(engine.chatter.load_model, input_elems, [info_box]).then(
    #     lambda: gr.Column(visible=engine.chatter.loaded), outputs=[chat_elems["chat_box"]]
    # )

    # unload_btn.click(engine.chatter.unload_model, input_elems, [info_box]).then(
    #     lambda: ([], []), outputs=[chatbot, messages]
    # ).then(lambda: gr.Column(visible=engine.chatter.loaded), outputs=[chat_elems["chat_box"]])

    # engine.manager.get_elem_by_id("top.visual_inputs").change(
    #     lambda enabled: gr.Column(visible=enabled),
    #     [engine.manager.get_elem_by_id("top.visual_inputs")],
    #     [chat_elems["image_box"]],
    # )

    return elem_dict
