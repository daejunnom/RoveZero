"""Re-capture this evidence only; does not modify F implementation or use GPUs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time

LOCKED_SOURCE = '36e619bae238a587e04c0a4bae24624eb5414aec'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    checkout, work = args.checkout.resolve(), args.artifact_root.resolve()
    if work == checkout or work.is_relative_to(checkout):
        parser.error('artifact-root must be outside checkout')
    work.mkdir(parents=True, exist_ok=False)
    bundle = Path(__file__).resolve().parent
    package = checkout / 'experiments/model-research'
    # The fixed source remains the input even when the evidence commit is newer.
    subprocess.run(['git','diff','--exit-code',LOCKED_SOURCE,'--',
                    'experiments/model-research/src','experiments/model-research/tests',
                    'experiments/model-research/fixtures','experiments/model-research/pyproject.toml'],
                   cwd=checkout, check=True, capture_output=True)
    (work / 'logs').mkdir()
    output = work / 'runs'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(package / 'src')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    records = []

    def logical(text):
        for actual, replacement in [(str(work),'${ARTIFACT_ROOT}'), (str(bundle),'${BUNDLE_ROOT}'),
                                    (str(checkout),'${CHECKOUT_ROOT}'), (sys.executable,'${PYTHON}'),
                                    ('/tmp/','${TMPDIR}/')]:
            text = text.replace(actual, replacement)
        return text

    def run(name, argv, *, environment=None, cwd=package):
        start = datetime.now(timezone.utc).isoformat()
        before = time.monotonic_ns()
        result = subprocess.run([str(arg) for arg in argv], cwd=cwd, env=environment or env,
                                capture_output=True, timeout=30)
        elapsed_ms = (time.monotonic_ns() - before) / 1_000_000
        item = {'id':name, 'argv':[logical(str(arg)) for arg in argv], 'cwd':logical(str(cwd)),
                'source_git_sha':LOCKED_SOURCE, 'source_binding_status':'locked_git_sources_verified_before_capture',
                'started_utc':start, 'elapsed_ms':elapsed_ms, 'exit_code':result.returncode,
                'capture':'new_preservation_recheck', 'execution':'actual_CPU_process',
                'input_kind':'synthetic_fixture_or_own_MIT_package', 'timeout_seconds':30,
                'environment_add':{'PYTHONPATH':logical((environment or env)['PYTHONPATH']),
                                   'PYTHONDONTWRITEBYTECODE':'1', 'PIP_DISABLE_PIP_VERSION_CHECK':'1'}}
        for label, raw in [('stdout', result.stdout), ('stderr', result.stderr)]:
            if len(raw) > 1024 * 1024:
                raise RuntimeError('capture exceeds 1 MiB; report size before publication')
            (work / 'logs' / (name + '.' + label + '.raw')).write_bytes(raw)
            text = logical(raw.decode('utf-8'))
            published = text.encode()
            (work / 'logs' / (name + '.' + label + '.txt')).write_bytes(published)
            item[label + '_original_sha256'] = hashlib.sha256(raw).hexdigest()
            item[label + '_published_sha256'] = hashlib.sha256(published).hexdigest()
            item[label + '_file'] = 'logs/recheck/' + name + '.' + label + '.txt'
        records.append(item)
        (work / 'commands.json').write_text(json.dumps({'commands':records},ensure_ascii=False,indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{name} failed; preserve logs instead of claiming success')

    run('regression-105', [sys.executable,'-m','unittest','discover','-s','tests','-v'])
    run('f01-split', [sys.executable,'-m','rz_data','split','--sources',bundle/'inputs/f01/sources.json',
                     '--seed','fixture-demo','--output-root',output,'--run-id','f01-plan'])
    run('f01-audit', [sys.executable,'-m','rz_data','validate','--manifest',bundle/'inputs/f01/manifest.json',
                     '--records',bundle/'inputs/f01/records.jsonl','--split-plan',output/'f01-plan/split-plan.json',
                     '--output-root',output,'--run-id','f01-audit'])
    run('reconstruct-training-audit', [sys.executable,'-m','rz_data','validate',
                     '--manifest',bundle/'inputs/f02/manifest.json','--records',bundle/'inputs/f02/records.jsonl',
                     '--split-plan',bundle/'inputs/f02/split-plan.json','--max-records','100',
                     '--max-file-bytes','1048576','--max-output-bytes','1048576',
                     '--output-root',output,'--run-id','training-audit'])
    common = ['--manifest',bundle/'inputs/f02/manifest.json','--records',bundle/'inputs/f02/records.jsonl',
              '--split-plan',bundle/'inputs/f02/split-plan.json','--features',bundle/'inputs/f02/features.json',
              '--output-root',output]
    recipe = bundle / 'inputs/f02/recipe.json'
    run('cpu-continuous', [sys.executable,'-m','rz_training','run','--recipe',recipe,*common,'--run-id','cpu-continuous'])
    run('cpu-pause', [sys.executable,'-m','rz_training','run','--recipe',recipe,*common,
                     '--run-id','cpu-pause','--stop-after','3'])
    run('cpu-resume', [sys.executable,'-m','rz_training','run','--recipe',recipe,*common,
                      '--run-id','cpu-resume','--resume',output/'cpu-pause/resume-checkpoint.json'])
    frozen = json.loads(recipe.read_text())
    frozen['freeze'] = True
    frozen_path = work / 'frozen-recipe.json'
    frozen_path.write_text(json.dumps(frozen,sort_keys=True,separators=(',',':'))+'\n')
    run('cpu-frozen', [sys.executable,'-m','rz_training','run','--recipe',frozen_path,*common,'--run-id','cpu-frozen'])
    run('export-verify', [sys.executable,'-m','rz_training','verify-export','--export',output/'cpu-resume/export.json',
                         '--output-root',output,'--run-id','export-verify'])
    source = work / 'package-source'
    source.mkdir()
    archive = work / 'source.tar'
    with archive.open('xb') as stream:
        subprocess.run(['git','archive','--format=tar',LOCKED_SOURCE,'experiments/model-research'],
                       cwd=checkout,stdout=stream,check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(source,filter='data')
    installed = work / 'installed'
    run('package-install', [sys.executable,'-m','pip','install','--no-index','--no-deps','--no-build-isolation',
                           '--no-compile','--no-cache-dir','--disable-pip-version-check','--target',installed,
                           source/'experiments/model-research'])
    installed_env = env.copy()
    installed_env['PYTHONPATH'] = str(installed)
    run('installed-rz-data', [installed/'bin/rz-data','validate','--manifest',bundle/'inputs/f01/manifest.json',
                             '--records',bundle/'inputs/f01/records.jsonl','--split-plan',output/'f01-plan/split-plan.json',
                             '--output-root',output,'--run-id','installed-f01-audit'],environment=installed_env)
    run('installed-resume', [installed/'bin/rz-train','run','--recipe',recipe,*common,'--run-id','installed-resume',
                            '--resume',output/'cpu-pause/resume-checkpoint.json'],environment=installed_env)
    run('installed-export-verify', [installed/'bin/rz-train','verify-export','--export',output/'installed-resume/export.json',
                                   '--output-root',output,'--run-id','installed-export-verify'],environment=installed_env)
    # Equality is numerical lifecycle evidence; no wall-time equality is assumed.
    captures = [json.loads((output/name/'resume-checkpoint.json').read_text())
                for name in ['cpu-continuous','cpu-resume','installed-resume']]
    fields = ['model','optimizer','sampler','best','history','provenance']
    matches = {key:all(item[key]==captures[0][key] for item in captures[1:]) for key in fields}
    assert all(matches.values()),matches
    frozen_receipt = json.loads((output/'cpu-frozen/receipt.json').read_text())
    assert frozen_receipt['weights_changed'] is False and frozen_receipt['selection']['step']==0
    original = json.loads((bundle/'original/f02/cpu-continuous-v1/resume-checkpoint.json').read_text())
    original_matches = {key:captures[0][key]==original[key] for key in fields}
    assert all(original_matches.values()),original_matches
    original_receipt = json.loads((bundle/'original/f02/cpu-resume-v1/receipt.json').read_text())
    reconstructed = json.loads((output/'training-audit/audit.json').read_text())
    assert reconstructed['digest']==original_receipt['provenance']['audit_digest']
    summary = {'schema_version':1, 'source_git_sha':LOCKED_SOURCE, 'capture_kind':'new_preservation_recheck',
               'execution':'actual_CPU_process', 'data_kind':'synthetic_numeric_fixture',
               'continuous_resume_installed_equal':matches,'original_continuous_equal':original_matches,
               'frozen_weights_unchanged':True,'reconstructed_audit_digest':reconstructed['digest'],
               'original_audit_body_was_missing':True,'original_audit_digest_matched':True,
               'python':platform.python_version(),'platform':platform.platform(),'cpu_count':os.cpu_count(),
               'gpu_training':'not_run','maia_training':'not_run','engine_arena':'not_run',
               'files_excluded':['source.tar','package-source','installed binaries/scripts/bytecode/build trees'],
               'command_count':len(records)}
    (work/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'commands':len(records),'all_exit_codes_zero':True,'comparison':matches,
                      'reconstructed_audit_matched':True}))


if __name__ == '__main__':
    main()
