"""Run actual agent calls and submit a saved run separately."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from agent import AtlasAgent

API='https://agents-course-unit4-scoring.hf.space'

def log(*parts):
    """Print progress without crashing on consoles that use a legacy code page."""
    line=' '.join(str(part) for part in parts)
    stream=sys.stdout
    encoding=getattr(stream,'encoding',None) or 'ascii'
    stream.write(line.encode(encoding,'replace').decode(encoding,'replace')+'\n')
    stream.flush()

def get_questions():
    with urllib.request.urlopen(API+'/questions',timeout=30) as response:return json.load(response)

def run_evaluation(output, agent=None, indices=None):
    agent=agent or AtlasAgent()
    questions=get_questions()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    run=json.loads(output.read_text()) if output.exists() else {'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':agent.model,'results':[]}
    done={r['task_id'] for r in run['results'] if not r.get('error')}
    for index,item in enumerate(questions):
        if indices is not None and index not in indices:continue
        if item['task_id'] in done:continue
        log(f"[{index+1}/{len(questions)}] {item['question'][:90]}")
        start=datetime.datetime.now(datetime.timezone.utc)
        try:
            result=agent.solve(item['question'],API+'/files/'+item['task_id'] if item.get('file_name') else '',on_event=lambda e:log('  tool:',e['tool']))
        except Exception as error:
            result={'answer':'UNKNOWN','error':str(error),'events':[]}
        result.update(task_id=item['task_id'],question=item['question'],seconds=(datetime.datetime.now(datetime.timezone.utc)-start).total_seconds())
        run['results']=[r for r in run['results'] if r['task_id']!=item['task_id']]+[result]
        output.write_text(json.dumps(run,indent=2,ensure_ascii=False),encoding='utf-8')
        log('  answer:',result['answer'],result.get('error',''))
        if 'HTTP 402' in result.get('error','') or 'HTTP 401' in result.get('error','') or 'HTTP 403' in result.get('error',''):break
    return run

def submit(run_file,username,space):
    run=json.loads(Path(run_file).read_text(encoding='utf-8'))
    questions=get_questions();results={r['task_id']:r for r in run['results']}
    payload={'username':username,'agent_code':f'https://huggingface.co/spaces/{space}/tree/main',
             'answers':[{'task_id':q['task_id'],'submitted_answer':results.get(q['task_id'],{}).get('answer','UNKNOWN')} for q in questions]}
    request=urllib.request.Request(API+'/submit',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=90) as response:score=json.load(response)
    receipt={'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_sha256':hashlib.sha256(Path(run_file).read_bytes()).hexdigest(),'result':score,'agent_code':payload['agent_code']}
    Path(run_file).with_suffix('.submission.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt

def summarize(run_file,summary_file):
    """Publishable report: per-question status and tool usage, without full page dumps."""
    run=json.loads(Path(run_file).read_text(encoding='utf-8'))
    receipt=Path(run_file).with_suffix('.submission.json')
    rows=[]
    for result in run['results']:
        rows.append({'task_id':result['task_id'],'question':result['question'][:180],
                     'answer':result['answer'],'answered':result['answer']!='UNKNOWN',
                     'tools':[event['tool'] for event in result.get('events',[])],
                     'seconds':round(result.get('seconds',0),1),'error':result.get('error','')[:200]})
    rows.sort(key=lambda row:row['task_id'])
    summary={'model':run.get('model'),'created_at':run.get('created_at'),
             'questions':len(rows),'answered':sum(1 for row in rows if row['answered']),
             'results':rows,
             'submission':json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else None}
    Path(summary_file).write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='work/evaluation.json');parser.add_argument('--submit',action='store_true');parser.add_argument('--username');parser.add_argument('--space');parser.add_argument('--summary');args=parser.parse_args()
    if args.summary:
        print(json.dumps({k:v for k,v in summarize(args.output,args.summary).items() if k!='results'},indent=2))
    elif args.submit:
        if not args.username or not args.space:parser.error('--username and --space required')
        print(json.dumps(submit(args.output,args.username,args.space),indent=2))
    else:run_evaluation(args.output)
