"""Optional loopback-only server for an existing Qwen3.5 text snapshot."""
import argparse
import json
import re
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

# 4B model, 32 layers x 4 KV heads x 256 head dim -> about 128 KB of KV cache per token.
CONTEXT_TRIM = int(os.getenv('ATLAS_LOCAL_CONTEXT_TRIM', '13000'))
CONTEXT_MAX = int(os.getenv('ATLAS_LOCAL_CONTEXT_MAX', '16000'))
MAX_NEW = int(os.getenv('ATLAS_LOCAL_MAX_NEW', '900'))

def parse_tools(text):
    calls=[]
    for block in re.findall(r'<tool_call>(.*?)</tool_call>',text,re.S):
        match=re.search(r'<function=([^>]+)>(.*?)</function>',block,re.S)
        if match:
            name,body=match.groups()
            args={k:v.strip() for k,v in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>',body,re.S)}
        else:
            parsed=json.loads(block);name=parsed['name'];args=parsed['arguments']
        calls.append({'id':f'call_{len(calls)}','type':'function','function':{'name':name,'arguments':json.dumps(args)}})
    return calls

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(args.snapshot,dtype=torch.bfloat16,device_map={'':'cuda:0'},local_files_only=True,use_safetensors=True,trust_remote_code=False,attn_implementation='sdpa')
    model.eval()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length',0))
                if length>500000:raise ValueError('Request too large')
                body=json.loads(self.rfile.read(length));messages=body['messages']
                # Keep full question and recent complete tool exchanges; bound older tool observations.
                for m in messages:
                    if m['role']=='tool':m['content']=m['content'][:int(os.getenv('ATLAS_LOCAL_TOOL_CHARS','7000'))]
                while True:
                    tokens=tokenizer.apply_chat_template(messages,tools=body.get('tools'),tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=False)
                    if tokens.shape[-1]<=CONTEXT_TRIM or len(messages)<=3:break
                    # Remove the oldest assistant/tool exchange, preserving system and question.
                    end=3
                    while end<len(messages) and messages[end]['role']=='tool':end+=1
                    del messages[2:end]
                if tokens.shape[-1]>CONTEXT_MAX:raise ValueError('Context budget exceeded')
                tokens=tokens.to('cuda')
                with torch.inference_mode():
                    output=model.generate(tokens,attention_mask=torch.ones_like(tokens),max_new_tokens=min(body.get('max_tokens',MAX_NEW),MAX_NEW),do_sample=False,pad_token_id=tokenizer.eos_token_id,use_cache=True)
                text=tokenizer.decode(output[0,tokens.shape[-1]:],skip_special_tokens=True)
                text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
                calls=parse_tools(text)
                message={'role':'assistant','content':text if not calls else None}
                if calls:message['tool_calls']=calls
                result={'choices':[{'message':message}],'usage':{'prompt_tokens':tokens.shape[-1],'completion_tokens':output.shape[-1]-tokens.shape[-1]}}
                data=json.dumps(result).encode();self.send_response(200)
            except Exception as error:
                data=json.dumps({'error':str(error)[:300]}).encode();self.send_response(500)
            self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
    print('Atlas local model ready on 127.0.0.1:'+str(args.port),flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
