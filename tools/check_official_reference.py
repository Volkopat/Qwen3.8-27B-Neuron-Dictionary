"""Recheck preserved official metadata and weight digests online; downloads no weights."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    reference=json.loads((ROOT/'evidence/official_reference.json').read_text())
    model=reference['model_id'];revision=reference['revision'];results={}
    for name,expected in reference['metadata_sha256'].items():
        url=f'https://huggingface.co/{model}/resolve/{revision}/{name}'
        with urllib.request.urlopen(url,timeout=60) as response:digest=hashlib.sha256(response.read()).hexdigest()
        results[name]=digest==expected
    url=f'https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(url,timeout=60) as response:metadata=json.load(response)
    weights={r['rfilename']:r['lfs']['sha256'] for r in metadata['siblings'] if r['rfilename'].endswith('.safetensors') and 'lfs' in r}
    results['official_weight_digest_list']=weights==reference['weight_sha256']
    print(json.dumps({'model_id':model,'revision':revision,'checks':results,'all_match':all(results.values())},indent=2))
    return 0 if all(results.values()) else 1
if __name__=='__main__':raise SystemExit(main())
