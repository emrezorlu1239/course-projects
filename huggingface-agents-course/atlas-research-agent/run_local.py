"""Use the settings recorded for the successful local evaluation."""
import os
import argparse

DEFAULTS={
    'ATLAS_ENDPOINT':'http://127.0.0.1:1234/v1/chat/completions',
    'ATLAS_MODEL':'qwen3.5-9b',
    'ATLAS_DOC_CHARS':'7000', 'ATLAS_OBS_CHARS':'7000',
    'ATLAS_SNIPPET_BEFORE':'500','ATLAS_SNIPPET_AFTER':'1300',
    'ATLAS_MAX_SNIPPETS':'5','ATLAS_MAX_LINKS':'25','ATLAS_CONTEXT_CHARS':'34000',
}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ui',action='store_true')
    parser.add_argument('--output',default='work/evaluation-local-new.json')
    args=parser.parse_args()
    for key,value in DEFAULTS.items():os.environ.setdefault(key,value)
    if args.ui:
        from app import demo
        demo.launch()
    else:
        from agent import AtlasAgent
        from evaluate import run_evaluation
        run_evaluation(args.output,agent=AtlasAgent(max_steps=8,calls_per_step=3,observation_chars=7000))

if __name__=='__main__':main()
