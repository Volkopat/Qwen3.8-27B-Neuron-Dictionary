"""Write an integrity manifest after deliberately reviewing a release change."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'evidence/SHA256SUMS.json','evidence/verification_report.json'}
def main():
    files={}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(x in {'.git','__pycache__','.pytest_cache','.venv'} for x in rel.parts):continue
        name=rel.as_posix()
        if name not in EXCLUDE:files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'evidence/SHA256SUMS.json').write_text(json.dumps({'algorithm':'sha256','files':files},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'files_sealed':len(files)}))
if __name__=='__main__':main()
