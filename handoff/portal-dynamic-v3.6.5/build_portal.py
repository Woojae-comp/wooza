#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an offline v3.4 sector/company/IP evidence browser from the supplied workbook and MD bundle.
No external API calls or third-party libraries; the source XLSX is never modified.
Usage: python build_portal.py path/to/source.xlsx path/to/pages.zip
"""
import csv
import io
import json
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets'; DOWNLOADS=HERE/'downloads'; DOCS=HERE/'docs'
SHEET_NAMES=['판정 현황','분야별 사건·9~10월','기업 개요(분야별)','IP 보유(IP 단위 판정)','IP 활용','IP 관계만(활용 아님)','해외 실행·성과','국제 계약·제휴','해외 계획·추진','국내 도입','해외 축소·철수','협업','거래 관계','플랫폼 이용·유통 성과','해외 법인·지분 처분','검토 후보','사건(기사 사례 단위)','사업 이력','사업 이력 후보 판정','기사·사건·이력 URL','v3.1 대비 변경','회귀 점검표','범위 밖 표본 비교','정답표 교정 로그(사례)','정답표 교정 로그(IP)','기업명 이력']
NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
EVENT_SHEETS=[5,7,8,9,10,11,12,13,14,15,16]
SECTOR_ORDER=['게임','음악','방송 및 영상','애니메이션','캐릭터','만화','지식정보','출판','영화','광고','콘텐츠솔루션']


def col_num(s):
 n=0
 for ch in s: n=n*26+ord(ch)-64
 return n-1

def sheet_rows(z,idx):
 """Read workbook row values, preserving empty cells and original data values."""
 with z.open(f'xl/worksheets/sheet{idx}.xml') as stream:
  for _,e in ET.iterparse(stream,events=('end',)):
   if e.tag!=NS+'row':continue
   vals={}
   for c in e:
    if c.tag!=NS+'c':continue
    match=re.match(r'[A-Z]+',c.attrib['r'])
    if not match:continue
    if c.attrib.get('t')=='inlineStr':
     v=''.join(c.itertext())
    else:
     v=c.findtext(NS+'v') or ''
    vals[col_num(match.group())]=v
   e.clear()
   yield vals

def iter_sheet(z,num):
 iterator=iter(sheet_rows(z,num)); head=next(iterator)
 header=[head.get(i,'') for i in range(max(head,default=-1)+1)]
 for row in iterator:
  yield {header[i]:row.get(i,'') for i in range(len(header)) if header[i]}

def to_date(v):
 if not v:return ''
 if re.fullmatch(r'\d{5}(?:\.\d+)?',str(v)):
  try:return (datetime(1899,12,30)+timedelta(days=float(v))).strftime('%Y-%m-%d')
  except (ValueError,OverflowError):pass
 return str(v)[:10]

def normal(s):
 s=unicodedata.normalize('NFKC',s or '').casefold()
 return re.sub(r'[\s\-·:()\[\]「」\'“”]+','',s)

def rights_group(right):
 if right.startswith('외부IP'):return '외부 IP'
 # 단서가 붙은 값(예: '자사원천(…세부 미확인)', '공동원작(지분 미확인)')도 같은 구분으로 묶는다(2026-10-22 수정)
 if right.startswith('자회사원천'):return '자회사 원천'
 if right.startswith('자사원천'):return '자사 원천'
 if right.startswith('인수보유'):return '인수 보유'   # 10-26: 인수한 원천 IP(자체 창작과 구분)
 if right.startswith('기획운영') or right.startswith('기획사보유'):return '기획·운영 IP'   # 10-26 3차: 소속사 기획·운영 그룹(권리 보유와 별도 집계, 표준전속계약서 제8·9조)
 if right.startswith('공동보유') or right.startswith('공동원작'):return '공동 보유'
 return '권리 미확인'

def rights_level(right,src,url):
 # 10-26 권리 확인 수준: 법적 권리 확인 / 사업활동 확인 / 외부 미확인 (권리 구분·사업 역할과 별도 축)
 if right.startswith('기획운영') or right.startswith('기획사보유'):return '사업활동 확인(그룹 기획·관리, 권리 개별 미확인)'
 if '자체 개발' in right:return '사업활동 확인(자체 개발)'
 if url and not right.startswith('미확인'):return '법적 권리 확인(외부 근거)'
 if url:return '외부 근거 확인(소유는 미확인)'
 if '공시' in src or '기사' in src:return '사업활동 확인'
 return '외부 미확인'

def dict_clean(d):return {k:v for k,v in d.items() if v not in ('',None)}

def compact_event(row,category):
 keys={'cid':'cid','사건ID':'eid','사업 이력ID':'bid','firm':'firm','code6':'code',
  '측면':'aspect','표시':'mark','검증':'verify','date':'date','대표IP':'ip','IP명':'ip_names',
  'IP관계':'ip_rel','활용방식명':'use','산출물':'output','단계':'stage','실행단계':'exec',
  '해외행위':'global_action','진출 국가':'country','진출시장':'market','협력사소재국':'partner_country',
  '행사개최국':'event_country','상대기업':'partner','상대산업':'partner_sector','공동행위':'joint_action',
  '실제주체':'actor','역할':'role','text':'title','대표 원문URL':'url',
  '추가 근거URL':'extra_url','원문 열람':'read','판정 근거':'basis','기간 구분':'period',
  '보도 건수':'reports','교정 근거URL':'correction_url','단계 흐름':'flow'}
 out={alias:row.get(orig,'') for orig,alias in keys.items()}
 out['date']=to_date(out.get('date',''))
 out['category']=category
 return dict_clean(out)

def find_md_name(s):
 try:return s.encode('cp437').decode('utf-8')
 except UnicodeError:
  try:return s.encode('cp437').decode('cp949')
  except UnicodeError:return s

def build(xlsx,mdzip):
 for d in [HERE,ASSETS,DOWNLOADS,DOCS]:d.mkdir(exist_ok=True,parents=True)
 if Path(xlsx).resolve() != (DOWNLOADS/'v3.4_원본데이터.xlsx').resolve():
  shutil.copyfile(xlsx,DOWNLOADS/'v3.4_원본데이터.xlsx')
 # All original sheets as a single fully-raw CSV archive: retains original source column names and values.
 csv_zip=DOWNLOADS/'v3.4_rawdata_26시트_CSV.zip'
 with zipfile.ZipFile(xlsx) as wb,zipfile.ZipFile(csv_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
  for i,name in enumerate(SHEET_NAMES,1):
   buffer=io.StringIO(newline='');writer=csv.writer(buffer,lineterminator='\n')
   source_rows=iter(sheet_rows(wb,i)); header=next(source_rows); ncols=max(header,default=-1)+1
   writer.writerow([header.get(j,'') for j in range(ncols)])
   for row in source_rows:
    writer.writerow([row.get(j,'') for j in range(ncols)])
   archive.writestr(f'{i:02d}_{name}.csv','\ufeff'+buffer.getvalue())
  company_source=list(iter_sheet(wb,3));ip_source=list(iter_sheet(wb,4))
  event_source={num:list(iter_sheet(wb,num)) for num in EVENT_SHEETS}
  history=list(iter_sheet(wb,26))

 # Keep source MD untouched as reference pages with relative README links.
 if mdzip and Path(mdzip).is_file():
  with zipfile.ZipFile(mdzip) as z:
   for entry in z.infolist():
    if entry.is_dir() or not entry.filename.endswith('.md'):continue
    filename=Path(find_md_name(entry.filename)).name
    (DOCS/filename).write_bytes(z.read(entry))

 code_names={}
 for r in company_source:code_names[r['code6']]=r['기업']
 for r in history:code_names[r['code6']]=r['표시명']

 companies=[];sector_codes=defaultdict(set)
 for r in company_source:
  code=r.get('code6','');sector=r.get('분야','')
  if sector not in SECTOR_ORDER:continue
  sector_codes[sector].add(code)
  companies.append(dict_clean({
   'sector':sector,'code':code,'name':code_names.get(code,r['기업']),
   'raw_name':r['기업'],'registered':r.get('연도별 등록 분야',''),
   'summary_owned':r.get('자사 원천 IP',''),'summary_external':r.get('외부 IP(사업 역할)',''),
   'summary_uses':r.get('주요 IP 활용 방식',''),'summary_countries':r.get('해외 실행·성과 국가',''),
   'summary_contracts':r.get('국제 계약·제휴 상대국',''),
   'summary_partners':r.get('주요 협업 상대',''),'summary_transactions':r.get('거래 관계','')
  }))

 ips=[];ip_lookup=defaultdict(list)
 for r in ip_source:
  if r.get('개체유형')!='IP':continue # exclude company, channel, event, general terms etc
  code=r.get('code6','');name=(r.get('대표 IP') or '').strip()
  if not name or not code:continue
  now=r.get('원천권리(현재)') or r.get('원천권리','')
  ip=dict_clean({'rid':r.get('rid',''),'code':code,'firm':code_names.get(code,r.get('기업','')),
   'name':name,'aliases':r.get('하위·관련 명칭',''),'parent':r.get('상위IP',''),
   'rights':now,'original_rights':r.get('원천권리',''),'rights_group':rights_group(now),'rights_level':rights_level(now,r.get('근거수준',''),r.get('교정 근거URL','')),
   'rights_period':r.get('권리 시점',''),'role':r.get('사업역할(IP 단위)',''),
   'source_level':r.get('근거수준',''),'note':r.get('IP 판정 비고',''),
   'evidence_url':r.get('교정 근거URL',''), 'article_count':r.get('기사 확인 사건',''),
   'period':r.get('기사 확인 기간','')})
  ips.append(ip)
  # Match the exact IP name only. A parent IP is not automatically the same rights-bearing item.
  ip_lookup[(code,normal(name))].append(ip['rid'])

 # 10-26 동일 개체 묶기: ip_same_entity.csv(대표 rid, 별칭 rid) — 원장 행은 그대로, 포털 IP 목록에서는 대표 1개로 집계하고 별칭 이름 기사도 대표 IP에 연결
 same=Path(__file__).resolve().parents[2]/'sector_pages'/'v34'/'ip_same_entity.csv'
 if same.exists():
  byrid={x['rid']:x for x in ips}; drop=set()
  for row in csv.DictReader(open(same,encoding='utf-8-sig')):
   a,b=byrid.get(row['rid_대표']),byrid.get(row['rid_별칭'])
   if not a or not b or b['rid'] in drop:continue
   a['aliases']=' · '.join(dict.fromkeys([x for x in (a.get('aliases','').split(' · ')+[b['name']]+b.get('aliases','').split(' · ')) if x and x!=a['name']]))
   a['same_entity']=' · '.join(filter(None,[a.get('same_entity',''),b['rid']]))
   ip_lookup[(b['code'],normal(b['name']))]=[a['rid']]
   drop.add(b['rid'])
  ips=[x for x in ips if x['rid'] not in drop]
  print('동일 개체 묶기',len(drop))
 # Unmodified original source rows are preserved in the downloadable CSV zip.
 events=[];categories=Counter()
 label={5:'IP 활용',7:'해외 실행·성과',8:'국제 계약·제휴',9:'해외 계획·추진',10:'국내 도입',
  11:'해외 축소·철수',12:'협업',13:'거래 관계',14:'플랫폼 이용·유통 성과',
  15:'해외 법인·지분 처분',16:'검토 후보'}
 unmatched=[];multi=[]
 for num in EVENT_SHEETS:
  for row in event_source[num]:
   ev=compact_event(row,label[num]);code=ev.get('code','')
   if not code:continue
   # Preserve each source row. Same CID may legitimately appear in multiple analytical facets.
   names=[ev.get('ip','')]
   # No automatic substring match; match the explicit primary IP name only.
   linked=list(dict.fromkeys(ip_lookup.get((code,normal(ev.get('ip',''))),[]))) if ev.get('ip') else []
   if len(linked)==1:ev['rid']=linked[0]
   elif len(linked)>1:multi.append(ev.get('cid'))
   elif ev.get('ip'):unmatched.append(ev.get('cid'))
   events.append(ev);categories[label[num]]+=1

 company_codes={r['code'] for r in companies}
 ips=[r for r in ips if r['code'] in company_codes]
 events=[r for r in events if r.get('code') in company_codes]
 ip_counts=Counter(r['code'] for r in ips)
 event_counts=Counter(r['code'] for r in events if r['category']=='IP 활용')
 sectors=[]
 for sector in SECTOR_ORDER:
  codes=sector_codes[sector]
  sectors.append({'name':sector,'companies':len(codes),'ips':sum(ip_counts[c] for c in codes),
   'ip_use_rows':sum(event_counts[c] for c in codes)})

 data={'version':'v3.4','date':'2026-10-21','updated':datetime.now().strftime('%Y-%m-%d'),'sectors':sectors,'companies':companies,
       'ips':ips,'events':events,'history':[r for r in history if r.get('code6')],
       'notes':{'company_count':len(companies),'unique_company_count':len(company_codes),
                'ip_count':len(ips),'event_count':len(events),'unmatched_ip_case_rows':len(unmatched),
                'ambiguous_ip_case_rows':len(multi),'event_categories':dict(categories),
                'raw_read_count':46,'raw_urls':4345}}
 # Existing enrichment snapshot is kept as a separate overlay, so rebuilding does not erase finance, disclosures or entity candidates.
 overlay=HERE/'data_enrichment_overlay.json'
 if overlay.is_file():
  enrich=json.loads(overlay.read_text(encoding='utf-8'))
  for k in ('flow','fin','sales','disc','strat','newsStatus','assets','intro','secRel','ipDerived','bidx','bidxInfo'):
   if k in enrich:data[k]=enrich[k]
 (HERE/'data.js').write_text('window.PORTAL_DATA = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
 (HERE/'data_quality.json').write_text(json.dumps(data['notes'],ensure_ascii=False,indent=2),encoding='utf-8')
 return data

if __name__=='__main__':
 xlsx=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'downloads/v3.4_원본데이터.xlsx'
 mdzip=Path(sys.argv[2]) if len(sys.argv)>2 else None
 data=build(xlsx,mdzip)
 print(json.dumps(data['notes'],ensure_ascii=False,indent=2))
 print('SECTOR',[(r['name'],r['companies'],r['ips']) for r in data['sectors']])
