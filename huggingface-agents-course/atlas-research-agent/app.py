import json
import os
from pathlib import Path
import gradio as gr
from agent import AtlasAgent, is_local_endpoint

LOCAL_MODE=is_local_endpoint(os.getenv("ATLAS_ENDPOINT","https://router.huggingface.co/v1/chat/completions"))

def answer(question, token):
    if not LOCAL_MODE and not (token or '').strip():raise gr.Error('Enter your Hugging Face inference token. This Space does not use an owner-funded token.')
    if not question.strip():raise gr.Error('Enter a question.')
    result=AtlasAgent(token=(token or "").strip()).solve(question)
    return result['answer'],result['events'],result['usage']

with gr.Blocks(title='Atlas Research Agent') as demo:
    gr.Markdown('# Atlas Research Agent\nResearch questions with web search, source reading, spreadsheets and bounded calculations.')
    gr.Markdown('The agent selects tools, reads their results and produces a concise answer. It has no built-in benchmark answers. Audio and video analysis are not supported by this text backend.')
    gr.Markdown('Local model mode — no Hugging Face token required.' if LOCAL_MODE else 'Cloud model mode — use your Hugging Face inference token.')
    with gr.Tab('Ask Atlas'):
        token=gr.Textbox(label='Your Hugging Face inference token',type='password',visible=not LOCAL_MODE,info='Used only for your requests to Hugging Face. Provider usage can consume your account credits. Do not enter a token unless you agree to that use.')
        question=gr.Textbox(label='Research question',lines=4)
        go=gr.Button('Research',variant='primary')
        output=gr.Textbox(label='Answer')
        trace=gr.JSON(label='Sources and tool observations')
        usage=gr.JSON(label='Token usage')
        go.click(answer,[question,token],[output,trace,usage],concurrency_limit=1)
    with gr.Tab('Evaluation'):
        report=Path('evaluation-summary.json')
        gr.JSON(value=json.loads(report.read_text()) if report.exists() else {'status':'Evaluation pending'},label='Recorded evaluation')
        gr.Markdown('Scores are obtained from the course API. Unsupported or failed questions are submitted as UNKNOWN; all 20 questions remain in the submission.')

if __name__=='__main__':demo.launch()
