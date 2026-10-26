#!/usr/bin/env python3
"""포털 데이터 무결성 점검.

건수를 고정하지 않는다(자료는 계속 늘어난다). 대신 다음을 확인한다.
 1. 구조: 필수 필드, 코드·rid 연결, notes 집계가 실제 배열과 일치하는지
 2. 날짜: 형식이 맞고 미래 날짜가 없는지
 3. 이전 판과 비교(--prev): 기업·IP·사건·공시가 줄었으면 실패(--allow-drop으로 허용)

  python validate_portal.py
  python validate_portal.py --prev ../분야별기업IP포털_v3.6.5/data.js
"""
from pathlib import Path
from collections import Counter
from datetime import date
import argparse, json, re, sys, zipfile, csv, io

ROOT = Path(__file__).resolve().parent
PREFIX = 'window.PORTAL_DATA = '
YMD = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def load(p):
    src = Path(p).read_text(encoding='utf-8')
    assert src.startswith(PREFIX), f'{p}: {PREFIX!r}로 시작하지 않음'
    return json.loads(src[len(PREFIX):].rstrip().rstrip(';'))


def counts(d):
    return {
        '분야': len({c['sector'] for c in d['companies']}),
        '분야×기업': len(d['companies']),
        '기업': len({c['code'] for c in d['companies']}),
        'IP': len(d['ips']),
        '활동 자료행': len(d['events']),
        '사건 ID': len({e.get('eid') or e.get('cid') for e in d['events']}),
        '공시': sum(len(v) for v in (d.get('disc') or {}).values()),
        '전략 활동': sum(len(v) for v in (d.get('strat') or {}).values()),
        '실적 분기': sum(len(v) for v in (d.get('fin') or {}).values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prev', help='이전 판 data.js (건수 감소 점검)')
    ap.add_argument('--allow-drop', action='store_true', help='건수 감소를 경고로만 처리')
    a = ap.parse_args()
    d = load(ROOT / 'data.js')
    errs, warns = [], []
    ev, ips, cos = d['events'], d['ips'], d['companies']

    # 1. 구조
    for k in ['version', 'companies', 'ips', 'events', 'sectors', 'notes']:
        if k not in d: errs.append(f'필드 없음: {k}')
    codes = {c['code'] for c in cos}; rids = {i['rid'] for i in ips}
    if len(rids) != len(ips): errs.append('rid 중복')
    # 같은 기사(cid)가 여러 분류·측면·IP에 나올 수 있으므로 (cid, 분류, 기업, rid, 측면)이 겹칠 때만 중복으로 본다
    dup = [k for k, v in Counter((e['cid'], e['category'], e['code'], e.get('rid'), e.get('aspect')) for e in ev).items() if v > 1]
    if dup: errs.append(f'활동 행 중복 {len(dup)}건: {dup[:3]}')
    bad = [i['rid'] for i in ips if i['code'] not in codes]
    if bad: errs.append(f'기업 명부에 없는 IP {len(bad)}건: {bad[:5]}')
    bad = [e['cid'] for e in ev if e['code'] not in codes]
    if bad: errs.append(f'기업 명부에 없는 활동 {len(bad)}건: {bad[:5]}')
    bad = [e['cid'] for e in ev if e.get('rid') and e['rid'] not in rids]
    if bad: errs.append(f'없는 rid를 가리키는 활동 {len(bad)}건: {bad[:5]}')
    for k in ['disc', 'strat', 'fin', 'intro', 'assets', 'newsStatus', 'secRel']:
        extra = set(d.get(k) or {}) - codes
        if extra: warns.append(f'{k}: 기업 명부에 없는 코드 {sorted(extra)[:5]}')
    n = d.get('notes') or {}
    real = {'company_count': len(cos), 'unique_company_count': len(codes), 'ip_count': len(ips), 'event_count': len(ev)}
    for k, v in real.items():
        if k in n and n[k] != v: errs.append(f'notes.{k}={n[k]} ≠ 실제 {v}')
    if 'event_categories' in n and Counter(n['event_categories']) != Counter(e['category'] for e in ev):
        errs.append('notes.event_categories가 실제 분류 건수와 다름')
    for s in d.get('sectors') or []:
        sc = {c['code'] for c in cos if c['sector'] == s['name']}
        if s.get('companies') != len(sc): errs.append(f"sectors[{s['name']}].companies={s.get('companies')} ≠ 실제 {len(sc)}")

    # 2. 날짜
    today = date.today().isoformat()
    dates = [('events', e['cid'], e.get('date')) for e in ev]
    for k in ['disc', 'strat']:
        dates += [(k, code, x.get('d')) for code, L in (d.get(k) or {}).items() for x in L]
    badfmt = [x for x in dates if x[2] and not YMD.match(str(x[2]))]
    future = [x for x in dates if x[2] and YMD.match(str(x[2])) and x[2] > today]
    if badfmt: errs.append(f'날짜 형식 오류 {len(badfmt)}건: {badfmt[:3]}')
    if future: errs.append(f'오늘({today}) 이후 날짜 {len(future)}건: {future[:3]}')
    empty = sum(1 for x in dates if not x[2])
    if empty: warns.append(f'날짜 빈 값 {empty}건(타임라인·연도 집계에서 빠짐)')

    # 3. 내려받기·파일
    for p in ['index.html', 'assets/app.js', 'assets/style.css', 'data.js', 'README.md']:
        if not (ROOT / p).is_file(): errs.append(f'파일 없음: {p}')
    for z in (ROOT / 'downloads').glob('*.zip'):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if not name.endswith('.csv'): continue
                rows = list(csv.reader(io.StringIO(zf.read(name).decode('utf-8-sig'))))
                if rows and any(len(r) != len(rows[0]) for r in rows[1:]): errs.append(f'{z.name}/{name}: 열 수가 다른 행')
    for doc in d.get('docs') or []:
        if not (ROOT / 'docs' / doc).is_file(): errs.append(f'docs 목록에 있으나 파일 없음: {doc}')

    cur = counts(d)
    print(f"{d.get('version')} · " + ' · '.join(f'{k} {v:,}' for k, v in cur.items()))
    if a.prev:
        p = load(a.prev); old = counts(p)
        print(f"이전 {p.get('version')} 대비: " + ' · '.join(f'{k} {cur[k]-old[k]:+,}' for k in cur))
        drops = [k for k in cur if cur[k] < old[k]]
        if drops: (warns if a.allow_drop else errs).append('건수 감소: ' + ', '.join(f'{k} {old[k]:,}→{cur[k]:,}' for k in drops))
    for w in warns: print('경고:', w)
    for e in errs: print('실패:', e)
    if errs: sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
