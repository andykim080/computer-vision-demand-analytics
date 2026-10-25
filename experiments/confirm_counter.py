"""Isolate the crossing-confirmation repair against a saved old counting class."""
import argparse
import copy
import json
import sys
import types
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import improve_entrance as experiment
from experiments import retail_entrance as previous
import counting

PLAN = json.loads((ROOT/'configs/counter-confirmation.json').read_text())
DATA = ROOT/'data/counter-confirmation'
OUTPUT = ROOT/'outputs/counter-confirmation'
TEMP = Path('/private/tmp/propsight-counter-confirmation')
write,sha,stamp = experiment.write,experiment.sha,experiment.stamp


def manifest():
    return json.loads((DATA/'manifest.json').read_text())


def report():
    return json.loads((OUTPUT/'report.json').read_text())


experiment.DATA,experiment.OUTPUT,experiment.TEMP,experiment.PLAN = DATA,OUTPUT,TEMP,PLAN
experiment.manifest,experiment.report = manifest,report
previous.DATA,previous.OUTPUT,previous.PLAN = DATA,OUTPUT,{**PLAN,'native_fps':25}
previous.manifest,previous.report,previous.write_json = manifest,report,write


def acquire():
    if (DATA/'manifest.json').exists():
        raise RuntimeError('Preserve the existing source manifest')
    DATA.mkdir(parents=True,exist_ok=True)
    TEMP.mkdir(parents=True,exist_ok=True)
    old_data = ROOT/'data/entrance-improvement'
    old = json.loads((old_data/'manifest.json').read_text())
    prior = json.loads((ROOT/'outputs/entrance-improvement/report.json').read_text())
    r = {'started_at_utc':stamp(),'runs':{},'old_counter_source':prior['previous_implementation']['counting.py']['source'],
         'protocol':{'plan':PLAN,'profile':copy.deepcopy(prior['frozen_profile'])},
         'historical_reports':{p:sha(ROOT/p) for p in ['outputs/entrance-improvement/report.json','outputs/retail-entrance/report.json']}}
    current = copy.deepcopy(old)
    current['acquired_at_utc'],current['plan'] = stamp(),PLAN
    for seq,info in current['sequences'].items():
        info['split']='development'
        current['entrance_references']['sequences'][seq]['split']='development'
    with zipfile.ZipFile(old_data/'annotations.zip') as archive:
        labels = {seq:archive.read(seq+'.xml') for seq in old['sequences']}
    directory = DATA/'holdout'
    directory.mkdir(exist_ok=True)
    for seq,label in PLAN['holdout'].items():
        if seq in current['sequences']:
            raise RuntimeError('A reserved source was already used')
        urls = {}
        for filename,path in [(seq+'.mpg',directory/(seq+'.mpg')),(label,TEMP/label)]:
            url=f'https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2/{seq}/{filename}'
            if not path.exists():
                with urllib.request.urlopen(url,timeout=90) as response:
                    path.write_bytes(response.read())
            urls[filename]=url
        xml=(TEMP/label).read_bytes()
        labels[seq]=xml
        current['sequences'][seq]={'split':'holdout','video':str((directory/(seq+'.mpg')).resolve()),
                                    'video_sha256':sha(directory/(seq+'.mpg')),'annotation':seq+'.xml',
                                    'annotation_sha256':__import__('hashlib').sha256(xml).hexdigest(),'urls':urls}
        print('Acquired reserved take',seq,flush=True)
    with zipfile.ZipFile(DATA/'annotations.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for seq,xml in labels.items():archive.writestr(seq+'.xml',xml)
    current['annotation_archive_sha256']=sha(DATA/'annotations.zip')
    write(DATA/'manifest.json',current)
    write(OUTPUT/'report.json',r)


def evaluate(seq,profile,mode):
    original=counting.LineCounter
    if mode=='pre-fix':
        module=types.ModuleType('propsight_saved_counter')
        sys.modules[module.__name__]=module
        exec(report()['old_counter_source'],module.__dict__)
        counting.LineCounter=module.LineCounter
    try:
        return experiment.evaluate(seq,profile)
    finally:
        counting.LineCounter=original


def develop():
    r=report()
    if r.get('holdout_reference_review_opened_at_utc') or r.get('holdout_opened_at_utc'):
        raise RuntimeError('The reserved test is already opened')
    current=manifest()
    development=[s for s,i in current['sequences'].items() if i['split']=='development']
    r['development_sequences']=development
    for mode in ['pre-fix','fixed']:
        r['runs'].setdefault(mode,{})
        for seq in development:
            if seq in r['runs'][mode]:continue
            record=evaluate(seq,r['protocol']['profile'],mode)
            record.pop('observations')
            record['observations_omitted_after_scoring']='Development only; original detector cache permits replay.'
            r['runs'][mode][seq]=record
            write(OUTPUT/'report.json',r)
        r.setdefault('development_count_audit',{})[mode]=previous.count_audit_totals([r['runs'][mode][s] for s in development])
        write(OUTPUT/'report.json',r)
    r['selected']='fixed'
    r['frozen_profile']=r['protocol']['profile']
    r['protocol']['candidates']={'pre-fix':r['frozen_profile'],'fixed':r['frozen_profile']}
    r['frozen_at_utc']=stamp()
    files=['tracker.py','detection.py','counting.py','experiments/confirm_counter.py','experiments/improve_entrance.py',
           'experiments/retail_entrance.py','configs/counter-confirmation.json','configs/bytetrack-low-fps.yaml',
           'yolo11s.pt','data/counter-confirmation/annotations.zip']
    r['frozen_sha256']={p:sha(ROOT/p) for p in files}
    r['frozen_sources']={s:i['video_sha256'] for s,i in current['sequences'].items()}
    r['frozen_development_validation_references']=copy.deepcopy(current['entrance_references']['sequences'])
    write(OUTPUT/'report.json',r)
    print('COUNTER_REPAIR_FROZEN',r['development_count_audit'],flush=True)


def holdout():
    r=report()
    experiment.verify(r)
    current=manifest()
    for seq in PLAN['holdout']:
        if current['entrance_references']['sequences'][seq]['review_status']=='pending visual review':
            raise RuntimeError('Review reference source frames first')
    r['holdout_opened_at_utc']=r.get('holdout_opened_at_utc',stamp())
    r['protocol']['reference_notes']=current['entrance_references']
    r['holdout_reference_manifest_sha256']=sha(DATA/'manifest.json')
    write(OUTPUT/'report.json',r)
    for mode in ['pre-fix','fixed']:
        for seq in PLAN['holdout']:
            if seq not in r['runs'][mode]:
                r['runs'][mode][seq]=evaluate(seq,r['frozen_profile'],mode)
                write(OUTPUT/'report.json',r)
        r.setdefault('holdout_count_audit',{})[mode]=previous.count_audit_totals([r['runs'][mode][s] for s in PLAN['holdout']])
        r.setdefault('holdout_totals',{})[mode]=previous.totals([r['runs'][mode][s] for s in PLAN['holdout']])
        write(OUTPUT/'report.json',r)
    experiment.verify(r)
    r['finished_at_utc']=stamp()
    write(OUTPUT/'report.json',r)
    print('COUNTER_TEST_COMPLETE',r['holdout_count_audit'],flush=True)


def gallery():
    # Reuse the video renderer, whose baseline key is 'full-frame'. This alias
    # points to the saved old counter result, not a different detector/config.
    r=report()
    r['runs']['full-frame']=r['runs']['pre-fix']
    r['holdout_count_audit']['full-frame']=r['holdout_count_audit']['pre-fix']
    r['holdout_totals']['full-frame']=r['holdout_totals']['pre-fix']
    r['validation_totals']={}
    write(OUTPUT/'report.json',r)
    previous.gallery()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['acquire','develop','references','approve-references','holdout','gallery'])
    args=parser.parse_args()
    if args.phase=='acquire':acquire()
    elif args.phase=='develop':develop()
    elif args.phase=='references':experiment.references('holdout')
    elif args.phase=='approve-references':experiment.approve_references('holdout')
    elif args.phase=='holdout':holdout()
    elif args.phase=='gallery':gallery()
