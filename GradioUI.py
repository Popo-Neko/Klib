import io
import os
import chardet

import gradio as gr
import pandas as pd
from jinja2 import Template

from scripts.inference4type import classification4type


def dashboard(file):
    if type(file) == bytes:
        result = chardet.detect(file)
        file = io.BytesIO(file)
    elif type(file) == str:
        with open(file, mode="rb") as f:
            content = f.read()
            result = chardet.detect(content)
    else:
        raise TypeError("Input to create a dashborad html must be csv file path or csv file in bytes type")
    df = pd.read_csv(file, encoding="ansi" if result['encoding'][:2] == "GB" else result['encoding'])
    df_html = df.to_html(index=False, max_rows=10, max_cols=5, col_space=100, table_id="data-preview")
    with open("./temp_files/DashBoardPreview.html", 'r') as template:
        template_content = template.read()
    template = Template(template_content)
    html_output = template.render(table=df_html)
    return html_output


def get_dropdown_list(path):
    dropdown_list = [filename for _, _, filename in os.walk(path)][0]
    return dropdown_list


def get_html_table(data_path, index=False, max_rows=5, max_cols=5, col_space=100, table_id="data-preview",
                   template_path="./temp_files/DashBoardPreview.html"):
    assert data_path[-3:] == "csv", "data for table must be a csv file"
    df = pd.read_csv("./temp_files/example_table.csv")
    df_html = df.to_html(index=index, max_rows=max_rows, max_cols=max_cols,
                         col_space=col_space, table_id=table_id)
    with open(template_path, 'r') as template:
        template_content = template.read()
    template = Template(template_content)
    example_html_table = template.render(table=df_html)
    return example_html_table


def create_file_input(file_component, dropdown_list):
    file_input_component = [file_component,
                            gr.Dropdown(choices=dropdown_list, value=r"model_checkpoint_utf8_30.pth", label="checkpoint",
                                        interactive=True),
                            gr.Textbox(value='combinedText', label="column name", interactive=True),
                            gr.Slider(0, 16, 16, step=1, label='BatchSize', interactive=True)
                            ]
    return file_input_component


dropdown_list = get_dropdown_list(r"./models/checkpoints")
example_html_table = get_html_table(r"./temp_files/example_table.csv")

with gr.Blocks(css=r'./temp_files/table.css') as demo:
    with gr.Tab(label="classification4type"):
        with gr.Row():
            with gr.Box():
                with gr.Column():
                    file = gr.File(file_types=[".csv"], type="binary")
                    file_input_component = create_file_input(file, dropdown_list)
                    btn = gr.Button("Submit")
                    file_output_component = gr.File(label="File Path")
                    btn.click(fn=classification4type, inputs=file_input_component, outputs=file_output_component)
            with gr.Box():
                with gr.Column():
                    gr.Markdown("Input Preview")
                    file_preview_component = gr.HTML(value=example_html_table, label="Data Preview",
                                                     elem_id="data-preview", show_label=True)
                    file.change(dashboard, file, file_preview_component)
                    gr.Markdown("Output Preview")
                    output_preview_component = gr.HTML(value=example_html_table, label="Data Preview",
                                                       elem_id="data-preview", show_label=True)
                    file_output_component.change(dashboard, file_output_component, output_preview_component)
    with gr.Tab(label="classification4subtitle"):
        with gr.Row():
            with gr.Box():
                with gr.Column():
                    file = gr.File(file_types=[".csv"], type="binary")
                    input_para = [
                        file,
                        gr.Dropdown(choices=dropdown_list, value="model_checkpoint_utf8_30.pth",
                                    label="checkpoint", interactive=True),
                        gr.Textbox(value='combinedText', label="column name", interactive=True),
                        gr.Slider(0, 16, 16, step=1, label='BatchSize', interactive=True)
                    ]
                    btn = gr.Button("Submit")
                    output_file = gr.File(label="File Path")
                    btn.click(fn=classification4type, inputs=input_para, outputs=output_file)
            with gr.Box():
                data_preview = gr.HTML(value=example_html_table, label="Data Preview", elem_id="data-preview")
                file.change(dashboard, file, data_preview)

if __name__ == '__main__':
    demo.launch()
