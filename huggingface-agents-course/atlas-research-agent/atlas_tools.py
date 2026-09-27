"""Bounded research tools. No shell execution or private-file access."""
import ast
import io
import os
import ipaddress
import json
import operator
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_BYTES = 8_000_000
# Document budgets. Small local models need tighter windows than hosted ones.
DOC_CHARS = int(os.getenv('ATLAS_DOC_CHARS', '16000'))
SNIPPET_BEFORE = int(os.getenv('ATLAS_SNIPPET_BEFORE', '500'))
SNIPPET_AFTER = int(os.getenv('ATLAS_SNIPPET_AFTER', '1200'))
MAX_SNIPPETS = int(os.getenv('ATLAS_MAX_SNIPPETS', '6'))
MAX_LINKS = int(os.getenv('ATLAS_MAX_LINKS', '30'))

def public_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Only public HTTP(S) URLs are supported')
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)):
        if not ipaddress.ip_address(result[4][0]).is_global:
            raise ValueError('Private and local addresses are not allowed')
    return url

class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def download(url):
    request = urllib.request.Request(public_url(url), headers={'User-Agent': 'AtlasResearchAgent/1.0 (educational research)'})
    with urllib.request.build_opener(PublicRedirect).open(request, timeout=30) as response:
        content = response.read(MAX_BYTES + 1)
        if len(content) > MAX_BYTES:
            raise ValueError('Document exceeds 8 MB limit')
        return content, response.headers.get('Content-Type', '')

def search_web(query):
    """Search the public web, retrying because free search endpoints rate-limit."""
    from ddgs import DDGS
    # Exclude answer banks: research the underlying sources instead.
    filtered = query + ' -site:huggingface.co/datasets -site:github.com'
    problem = ''
    for attempt, text in enumerate((filtered, filtered, query)):
        try:
            hits = list(DDGS(timeout=25).text(text, max_results=6))
            if hits:
                return hits
            problem = 'no results'
        except Exception as error:
            problem = str(error)[:200]
        time.sleep(2 * (attempt + 1))
    return {'error': 'search unavailable: ' + problem, 'hint': 'Try a shorter query or a known source URL.'}

def read_url(url, find=''):
    data, kind = download(url)
    if 'pdf' in kind or urllib.parse.urlsplit(url).path.lower().endswith('.pdf'):
        from pypdf import PdfReader
        text = '\n'.join(page.extract_text() or '' for page in PdfReader(io.BytesIO(data)).pages[:100])
    else:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(data, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        text = soup.get_text(' ', strip=True)
        links = [{'text':a.get_text(' ',strip=True)[:140], 'url':urllib.parse.urljoin(url,a['href'])} for a in soup.find_all('a',href=True)]
        if find:
            links = [a for a in links if find.casefold() in (a['text']+' '+a['url']).casefold()]
        text += '\nLINKS: '+json.dumps(links[:MAX_LINKS])
    if find:
        low = text.casefold(); needle = find.casefold(); start = 0; pieces = []
        while len(pieces) < MAX_SNIPPETS:
            index = low.find(needle,start)
            if index < 0: break
            pieces.append(text[max(0,index-SNIPPET_BEFORE):index+SNIPPET_AFTER]); start=index+len(needle)
        if pieces:
            text='\n...\n'.join(pieces)
        else:
            text='NO MATCH for '+repr(find)+' on this page. Page start:\n'+text[:DOC_CHARS//2]
    return {'url':url,'text':text[:DOC_CHARS]}

def read_attachment(url):
    try:
        data, kind = download(url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {'unavailable': 'The evaluation server returned 404 for this attachment: it serves no file for this task. '
                                   'The attachment cannot be inspected. Do not search the web for its contents.'}
        raise
    if data[:2] == b'PK':
        from openpyxl import load_workbook
        book=load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        result={s.title:list(s.iter_rows(values_only=True))[:1500] for s in book.worksheets[:5]}
        book.close()
        return result
    if data.startswith(b'%PDF'):
        from pypdf import PdfReader
        return '\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(data)).pages[:60])[:22000]
    if data.startswith((b'ID3', b'\xff\xfb', b'\xff\xf3', b'\x89PNG', b'\xff\xd8')) or 'audio' in kind or 'image' in kind:
        return {'unsupported':'This text backend cannot inspect audio/images. Do not claim to have seen or heard them.'}
    return data.decode('utf-8',errors='replace')[:22000]

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod}

def calculate(expression):
    """Evaluate a small arithmetic AST, not arbitrary Python."""
    if len(expression)>12000: raise ValueError('Expression too long')
    def walk(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            if abs(node.value)>1e15: raise ValueError('Number too large')
            return node.value
        if isinstance(node,(ast.List,ast.Tuple)): return [walk(e) for e in node.elts]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            return -walk(node.operand) if isinstance(node.op,ast.USub) else walk(node.operand)
        if isinstance(node,ast.BinOp) and type(node.op) in OPS:
            left,right=walk(node.left),walk(node.right)
            if not isinstance(left,(int,float)) or not isinstance(right,(int,float)): raise ValueError('Numeric operands required')
            value=OPS[type(node.op)](left,right)
            if abs(value)>1e18: raise ValueError('Result too large')
            return value
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords and node.func.id in ('sum','min','max','round','abs','len'):
            functions={'sum':sum,'min':min,'max':max,'round':round,'abs':abs,'len':len}
            return functions[node.func.id](*[walk(a) for a in node.args])
        raise ValueError('Unsupported expression; use arithmetic, lists, sum/min/max/round/abs/len')
    return walk(ast.parse(expression,mode='eval').body)

def reverse_text(text):
    return text[::-1]

REGISTRY={'search_web':search_web,'read_url':read_url,'read_attachment':read_attachment,'calculate':calculate,'reverse_text':reverse_text}

def schema(name,description,properties,required):
    return {'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':{k:{'type':'string','description':v} for k,v in properties.items()},'required':required,'additionalProperties':False}}}

TOOLS=[
    schema('search_web','Search public web sources. Do not search benchmark answer banks.',{'query':'Search query'},['query']),
    schema('read_url','Read a webpage or PDF; optionally find passages and links mentioning a term.',{'url':'Public URL','find':'Optional phrase to find'},['url']),
    schema('read_attachment','Read an attached spreadsheet, PDF, text or source code. Source code is read, never executed.',{'url':'Attachment URL'},['url']),
    schema('calculate','Evaluate arithmetic with numeric lists, sum/min/max/round/abs/len. No arbitrary code.',{'expression':'Arithmetic expression'},['expression']),
    schema('reverse_text','Reverse text characters, useful for encoded/reversed text.',{'text':'Text to reverse'},['text']),
]
