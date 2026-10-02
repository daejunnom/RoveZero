#!/usr/bin/env python3
"""Generated after preservation; t1 consumes original argv, t2 is reconstructed."""
import argparse,json,os,pathlib,shutil,subprocess,tempfile
parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['t1','t2-reconstructed','stdout-log'],required=True)
args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parent
output=pathlib.Path(os.environ['ARTIFACT_ROOT'])
output.mkdir(parents=True,exist_ok=True)
work=pathlib.Path(tempfile.mkdtemp(prefix='runner-reproduction-',dir=output))
for name in ['synthetic-uci.py','prefix-opening.pgn']:
    shutil.copyfile(root/'reports'/name,work/name)
(work/'synthetic-uci.py').chmod(0o755)
argv_file='smoke-stdout-log.argv.json' if args.phase=='stdout-log' else 'smoke-t1-argv.json'
argv=json.loads((root/'reports'/argv_file).read_text())
argv[0]=os.environ['FASTCHESS']
argv=[item.replace('${ARTIFACT_ROOT}/reports/runner-research',str(work)) for item in argv]
if args.phase=='t2-reconstructed':
    argv=[item.replace('tc=0.010+0.005','st=0.1').replace('smoke-t1-pair','smoke-pair') for item in argv]
name='smoke-stdout-log' if args.phase=='stdout-log' else 'smoke-t1-pair' if args.phase=='t1' else 'smoke-pair'
(work/'generated-reproduction-argv.json').write_text(json.dumps({'phase':args.phase,'origin':'new reproduction; not original execution','argv':argv},indent=2)+'\n')
with (work/(name+'.stdout')).open('wb') as stdout,(work/(name+'.stderr')).open('wb') as stderr:
    result=subprocess.run(argv,cwd=work,stdout=stdout,stderr=stderr,timeout=30,check=False)
(work/'generated-reproduction-exit.json').write_text(json.dumps({'exit_code':result.returncode,'phase':args.phase})+'\n')
print(work)
raise SystemExit(result.returncode)
