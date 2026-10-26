/* Offline enterprise/IP evidence browser. No network access, tracking or external libraries. */
(() => {
  'use strict';
  const D=window.PORTAL_DATA;
  if(!D){document.getElementById('main').textContent='데이터를 읽지 못했습니다. data.js 파일을 확인하세요.';return;}
  const byId=id=>document.getElementById(id);
  const escape=s=>String(s===undefined||s===null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const val=(x,empty='확인된 항목 없음')=>x ? escape(x) : `<span class="meta">${escape(empty)}</span>`;
  const safeUrl=s=>/^https?:\/\//i.test(s||'')?s:'';
  const uniqCount=arr=>new Set(arr.map(x=>x.eid||x.cid).filter(Boolean)).size;
  /* 데이터에서 계산하는 값(10-26): 기간·분야·건수·분류·문서 목록은 data.js에 들어온 자료로 정하고 화면에 고정하지 않는다. 자료가 늘면 화면이 따라간다.
     data.js에 period{start,end}·updated·docs가 있으면 그 값을 우선한다. */
  const ymd=s=>/^\d{4}-\d{2}-\d{2}/.test(s||'')?String(s).slice(0,10):'';
  const RANGE=(()=>{const ds=[];for(const e of D.events)if(ymd(e.date))ds.push(ymd(e.date));
    for(const k of ['disc','strat'])for(const L of Object.values(D[k]||{}))for(const x of L)if(ymd(x.d))ds.push(ymd(x.d));
    ds.sort();const P=D.period||{};const start=ymd(P.start)||ds[0]||'2024-01-01',end=ymd(P.end)||ds[ds.length-1]||start;
    const y0=+start.slice(0,4),y1=+end.slice(0,4),m0=+start.slice(5,7),m1=+end.slice(5,7);
    return {start,end,y0,y1,m0,m1,years:Array.from({length:y1-y0+1},(_,i)=>String(y0+i)),q0:y0+'Q1',ko:`${y0}년 ${m0}월~${y1}년 ${m1}월`};})();
  const FIN_LAST=(()=>{let m='';for(const L of Object.values(D.fin||{}))for(const x of L)if(x.p>m)m=x.p;return m;})();   // 가장 최근 분기(예: 2026Q2)
  const finQuarters=y=>!FIN_LAST||y<FIN_LAST.slice(0,4)?4:y===FIN_LAST.slice(0,4)?+FIN_LAST.slice(-1):0;   // 그 해에 공시됐어야 할 분기 수
  const finScope=n=>n===4?'연간(4개 분기)':n===2?'상반기(1~2분기)':n===1?'1분기':`1~${n}분기 누적`;
  const UPDATED=ymd(D.updated)||ymd((D.bidxInfo||{}).at)||ymd(D.date)||RANGE.end;
  const VER=String(D.version||'').replace(/[^0-9A-Za-z.]/g,'')||'data';
  const ordered=(known,found)=>[...known.filter(k=>found.has(k)),...[...found].filter(k=>!known.includes(k)).sort((a,b)=>a.localeCompare(b,'ko'))];
  const SECTORS=(()=>{const names=[...new Set([...(D.sectors||[]).map(s=>s.name),...D.companies.map(c=>c.sector)])];
    return names.map(name=>{const codes=new Set(D.companies.filter(c=>c.sector===name).map(c=>c.code));
      return {name,companies:codes.size,ips:D.ips.filter(i=>codes.has(i.code)).length,ip_use_rows:D.events.filter(e=>codes.has(e.code)&&e.category==='IP 활용').length};}).filter(s=>s.companies);})();
  const NOTES=(()=>{const cats={};for(const e of D.events)cats[e.category]=(cats[e.category]||0)+1;
    return {...(D.notes||{}),company_count:D.companies.length,unique_company_count:new Set(D.companies.map(c=>c.code)).size,ip_count:D.ips.length,event_count:D.events.length,
      event_categories:Object.fromEntries(ordered(Object.keys((D.notes||{}).event_categories||{}),new Set(Object.keys(cats))).map(k=>[k,cats[k]])),
      raw_read_count:D.events.filter(e=>e.read==='원문 열람').length,raw_urls:new Set(D.events.map(e=>e.url).filter(Boolean)).size,
      article_count:new Set(D.events.map(e=>e.url||e.title).filter(Boolean)).size};})();
  const sectorByName=Object.fromEntries(SECTORS.map(x=>[x.name,x]));
  // 동일한 분야 설명이 여러 기업에 그대로 배정된 경우 기업별 사실로 소개하지 않는다.
  const flowOwners=new Map();
  for(const sectorFlow of Object.values(D.flow||{}))for(const [name,summary] of Object.entries(sectorFlow)){if(!summary)continue;if(!flowOwners.has(summary))flowOwners.set(summary,new Set());flowOwners.get(summary).add(name);}

  const codeCompanies=new Map();
  const ipCodes=new Map(), evCodes=new Map();
  for(const c of D.companies)if(!codeCompanies.has(c.code))codeCompanies.set(c.code,c);
  for(const ip of D.ips){if(!ipCodes.has(ip.code))ipCodes.set(ip.code,[]);ipCodes.get(ip.code).push(ip);}
  for(const ev of D.events){if(!evCodes.has(ev.code))evCodes.set(ev.code,[]);evCodes.get(ev.code).push(ev);}
  /* '해외 관련' = 해외 계열 분류 + 다른 분류로 묶인 해외 사건(e.ov, content_hub/portal_overseas.py) */
  const OVCAT='해외 관련',catOk=(e,g)=>g==='전체'||(g===OVCAT?!!e.ov:e.category===g);
  /* IP 활용 세부 활동(10-23): 사건의 활용 방식(use) 9종. 한 사건이 여러 방식에 해당할 수 있음 */
  const USES=(()=>{const known=[['게임화·게임 내 협업','게임화'],['영상화·애니메이션화','영상·애니화'],['음원·OST','음원·OST'],['공연·뮤지컬·콘서트','공연·콘서트'],['상품화·굿즈·완구','상품화·굿즈'],['웹툰·출판화','웹툰·출판화'],['팝업·전시·체험·공간사업','팝업·전시·공간'],['브랜드 협업·라이선싱','브랜드 협업·라이선싱']];
    const found=new Set();for(const e of D.events)for(const u of String(e.use||'').split(' · '))if(u.trim()&&u.trim()!=='nan')found.add(u.trim());found.delete('기타 IP 활용');
    return [...known,...[...found].filter(u=>!known.some(k=>k[0]===u)).sort((a,b)=>a.localeCompare(b,'ko')).map(u=>[u,u]),['기타 IP 활용','기타']];})();   // 새 활용 방식은 자동으로 단추에 추가
  const usesOf=e=>{const u=String(e.use||'').split(' · ').map(x=>x.trim()).filter(x=>x&&x!=='nan');return u.length?u:['기타 IP 활용'];};
  const useOk=(e,u)=>!u||u==='전체'||usesOf(e).includes(u);
  const useShort=u=>(USES.find(x=>x[0]===u)||[u,u])[1];
  const useBar=(list,cur,attr)=>{const items=USES.map(([u,l])=>[u,l,uniqCount(list.filter(e=>useOk(e,u)))]).filter(x=>x[2]);
    return `<div class="usebar"><span class="uselabel">IP 활용 세부 활동</span><button class="filterbtn sm ${!cur||cur==='전체'?'active':''}" data-${attr}="전체">전체 ${uniqCount(list)}</button>${items.map(([u,l,n])=>`<button class="filterbtn sm ${cur===u?'active':''}" data-${attr}="${escape(u)}">${escape(l)} ${n}</button>`).join('')}</div>`;};
  const categories=['전체','해외 관련',...ordered(['IP 활용','해외 실행·성과','국제 계약·제휴','해외 계획·추진','국내 도입','협업','거래 관계','플랫폼 이용·유통 성과','해외 법인·지분 처분','해외 축소·철수','검토 후보'],new Set(D.events.map(e=>e.category).filter(Boolean)))];
  const groups=['전체',...ordered(['자사 원천','자회사 원천','인수 보유','공동 보유','기획·운영 IP','외부 IP','권리 미확인'],new Set(D.ips.map(i=>i.rights_group).filter(Boolean)))];
  const hash=new URLSearchParams(location.hash.slice(1));
  const wanted=hash.get('sector');
  const ALL='전체';
  /* 뉴스 반영 상태: 기사 사건이 0건인 기업만 원인별 문구 */
  const NSL={'수집 필요':['뉴스 수집 필요','이 기업은 뉴스 기사가 아직 수집되지 않아 수집이 필요합니다'],'수집 중':['뉴스 수집 중','이 기업의 뉴스 기사는 현재 수집 중입니다'],'수집 완료·분석 반영 전':['뉴스 분석 반영 전','이 기업의 뉴스 기사는 수집됐지만 아직 분석에 반영되지 않았습니다'],'수집 완료':['해당 기사 사건 없음','수집된 뉴스 기사 중 IP 활용·해외·협업 사건으로 확인된 것이 없습니다']};
  const newsTag=c=>{if((evCodes.get(c.code)||[]).length)return null;const st=((D.newsStatus||{})[c.code]||{}).s||'수집 필요';return NSL[st]||NSL['수집 필요'];};
  const state={aggCat:'전체',aggSearch:'',aggLimit:12,aggIpGroup:'전체',aggIpSearch:'',aggIpLimit:24,sector:sectorByName[wanted]?wanted:ALL,code:'',rid:'',ipGroup:'전체',ipSearch:'',ipLimit:18,evCategory:'전체',evSearch:'',evLimit:10,companySearch:''};
  const getSectorCompanies=()=>state.sector===ALL?[...codeCompanies.values()]:D.companies.filter(x=>x.sector===state.sector);
  const getSectorCodes=()=>new Set(getSectorCompanies().map(x=>x.code));
  const company=()=>state.sector===ALL?codeCompanies.get(state.code):D.companies.find(x=>x.sector===state.sector&&x.code===state.code);
  const firmIps=()=>ipCodes.get(state.code)||[];
  const firmEvents=()=>evCodes.get(state.code)||[];
  const selectedIp=()=>firmIps().find(x=>x.rid===state.rid);
  function niceRights(group){const cls=['자사 원천','자회사 원천','인수 보유','공동 보유'].includes(group)?'own':group==='기획·운영 IP'?'external':group==='외부 IP'?'external':'warn';return `<span class="pill ${cls}">${escape(group)}</span>`;}
  function updateHash(){const q=new URLSearchParams({sector:state.sector,code:state.code});try{history.replaceState(null,'','#'+q.toString());}catch(_){location.hash=q.toString();}}
  function defaultCode(s){let c=D.companies.filter(x=>x.sector===s);c.sort((a,b)=>{
    let n=x=>(evCodes.get(x.code)||[]).filter(e=>e.category==='IP 활용').length;
    return n(b)-n(a)||a.name.localeCompare(b.name,'ko');});return c[0]?.code||'';}
  function setupSelection(){const hcode=hash.get('code');if(state.sector===ALL){state.code='';state.rid='';return;}state.code=getSectorCompanies().some(x=>x.code===hcode)?hcode:'';resetIp();}
  function resetIp(){state.assetRel='전체';state.assetKind='전체';state.assetSearch='';state.assetLimit=40;state.rid=firmIps().find(x=>x.rights_group==='자사 원천'&&(firmEvents().some(ev=>ev.rid===x.rid&&ev.category==='IP 활용')))?.rid || firmIps().find(x=>firmEvents().some(e=>e.rid===x.rid&&e.category==='IP 활용'))?.rid || firmIps()[0]?.rid || '';state.ipGroup='전체';state.ipSearch='';state.ipLimit=18;state.evCategory='전체';state.evSearch='';state.evLimit=10;}
  function cardMetric(label,n){return `<div class="stat"><span>${escape(label)}</span><strong>${Number(n).toLocaleString('ko-KR')}</strong></div>`;}
  function renderSidebar(){byId('sectors').innerHTML=`<button class="${state.sector===ALL?'active':''}" data-sector="${ALL}">전체 개요<span>${NOTES.unique_company_count}</span></button>`+SECTORS.map(s=>`<button class="${s.name===state.sector?'active':''}" data-sector="${escape(s.name)}">${escape(s.name)}<span>${s.companies}</span></button>`).join('');}
  function renderSector(){const isAll=state.sector===ALL,hasC=!!state.code&&!isAll;byId('overview').hidden=!isAll;byId('aggregate').hidden=hasC;byId('companySection').hidden=false;byId('profile').hidden=!hasC;
    {const m=byId('main'),ag=byId('aggregate'),cs=byId('companySection');if(isAll)m.insertBefore(ag,cs);else m.insertBefore(cs,ag);}   // 전체: 통합 데이터 → 전체 기업 / 분야: 기업 선택 → 분야 통합 데이터
    byId('companyEyebrow').textContent=isAll?'전체 기업':'기업 선택';byId('companyHeading').textContent=isAll?`전체 기업 ${codeCompanies.size}곳`:`${state.sector} 분야 기업 선택`;
    if(isAll){renderOverview();renderSidebar();renderAggregate();renderCompanies();return;}
    if(!hasC)renderAggregate();
    const s=sectorByName[state.sector],codes=getSectorCodes();
    const uniqueCaseCount=new Set(D.events.filter(x=>codes.has(x.code)).map(x=>x.eid||x.cid)).size;
    byId('sectorLabel').textContent=`분야별 기업 프로필  /  ${state.sector}`;
    byId('sectorTitle').textContent=`${state.sector} 분야의 기업과 IP`;
    byId('sectorStats').innerHTML=[cardMetric('등재 기업',s.companies),cardMetric('IP 분류 항목',s.ips),cardMetric('IP 활용 자료',s.ip_use_rows),cardMetric('연결 사건 ID(전체 범주)',uniqueCaseCount)].join('');
    renderSidebar();renderCompanies();if(hasC)renderProfile();}
  function renderCompanies(){const all=getSectorCompanies();const q=state.companySearch.toLowerCase().trim();
    const list=all.filter(x=>!q || [x.name,x.raw_name,x.summary_owned,x.summary_external].join(' ').toLowerCase().includes(q) || (ipCodes.get(x.code)||[]).some(ip=>ip.name.toLowerCase().includes(q)));
    byId('companyCount').textContent=`${list.length} / ${all.length}개 기업`;
    byId('companySearch').value=state.companySearch;
    const SR=c=>((D.secRel||{})[c.code]||{})[state.sector];const isSub=c=>state.sector!==ALL&&SR(c)&&SR(c).main===false;
    const mainOf=c=>Object.entries((D.secRel||{})[c.code]||{}).filter(([k,v])=>v.main).map(([k])=>k).join('·');
    const card=c=>{
      const ips=ipCodes.get(c.code)||[], evs=evCodes.get(c.code)||[], own=ips.filter(x=>['자사 원천','자회사 원천','인수 보유','공동 보유'].includes(x.rights_group));
      const top=own.map(x=>x.name).slice(0,3).join(' · ') || ips.map(x=>x.name).slice(0,2).join(' · ');
      return `<button class="companycard ${state.code===c.code?'selected':''}" data-company="${escape(c.code)}"><div class="cardtop"><h3>${escape(c.name)}</h3><span class="code">${escape(c.code)}</span></div><p>${escape(top||'IP 단위 자료 없음')}</p><p class="meta">기사 사건 ${evN(c)}건 · 공시 ${dsN(c)}건</p>${isSub(c)?`<p class="meta">주 분야: ${escape(mainOf(c)||'다른 분야')} · 이 분야 관련 기사·활동 ${SR(c).n}건${SR(c).rev!=null?` · 매출 품목 ${SR(c).rev}%`:''}</p>`:''}<div class="cardbottom"><span class="pill">IP ${ips.length}</span>${((D.assets||{})[c.code]||[]).length?`<span class="pill">자산 ${((D.assets||{})[c.code]||[]).length}</span>`:''}<span class="pill blue">IP 활용 ${uniqCount(evs.filter(x=>x.category==='IP 활용'))}</span><span class="pill">해외 실행 ${uniqCount(evs.filter(x=>x.category==='해외 실행·성과'))}</span>${newsTag(c)?`<span class="pill pending">${newsTag(c)[0]}</span>`:''}</div></button>`;
    };const evN=c=>uniqCount((evCodes.get(c.code)||[]).filter(e=>e.category!=='검토 후보')),dsN=c=>(D.disc[c.code]||[]).length;
    const sk=state.companySort||'기사';const cmp=(a,b)=>sk==='가나다'?a.name.localeCompare(b.name,'ko'):sk==='공시'?(dsN(b)-dsN(a))||(evN(b)-evN(a))||a.name.localeCompare(b.name,'ko'):(evN(b)-evN(a))||(dsN(b)-dsN(a))||a.name.localeCompare(b.name,'ko');
    const mainL=list.filter(c=>!isSub(c)).sort(cmp),subL=list.filter(isSub).sort(cmp);
    const sortBar=`<div class="sortbar"><span class="meta">정렬</span>${['기사','공시','가나다'].map(k=>`<button class="filterbtn ${sk===k?'active':''}" data-csort="${k}">${k==='가나다'?'가나다순':k+' 많은 순'}</button>`).join('')}</div>`;
    byId('companies').innerHTML=list.length?sortBar+mainL.map(card).join('')+(subL.length?`<div class="subdivider"><strong>주 분야가 다른 등재 기업 ${subL.length}곳</strong><span>이 분야에도 등재돼 있지만 이 분야 관련 기사·전략 활동과 매출 품목이 적은 기업입니다(등재는 그대로, 순서만 아래로).</span></div>`+subL.map(card).join(''):''):'<div class="empty">검색 조건에 해당하는 기업이 없습니다.</div>';
  }

  /* 전체 개요: 데이터 제작 과정 · 확인할 수 있는 것 · 사용법 · 분야 바로가기 · 읽을 때 유의점 */
  function renderOverview(){const N=NOTES,uniqEv=new Set(D.events.map(x=>x.eid||x.cid)).size;
    byId('sectorLabel').textContent='전체 개요';
    byId('sectorTitle').textContent='콘텐츠 상장사 IP·사업 현황';
    byId('sectorStats').innerHTML=[cardMetric('분석 기업(고유)',N.unique_company_count),cardMetric('IP 분석 항목',N.ip_count),cardMetric('활동 자료행',N.event_count),cardMetric('연결 사건 ID',uniqEv)].join('');
    const steps=[
      ['대상 기업','기관 명부(생애주기 원장)에 등재된 콘텐츠 기업 '+N.unique_company_count+'개사, '+SECTORS.length+'개 분야. 여러 분야에 등재된 기업은 각 분야에 모두 표시합니다(분야×기업 '+N.company_count+'건).'],
      ['원자료',RANGE.ko+' 기업 관련 기사(제목·요약·원문 URL), DART 공시·사업보고서의 보유 자산 목록, 기업–IP 관계표.'],
      ['활동 분석','기업 활동 기사 '+N.article_count.toLocaleString('ko-KR')+'건을 IP 활용·해외 진출·협업 세 측면으로 나눠 분석했습니다(확인·추정·추가 확인). 같은 활동의 반복 보도는 하나의 사건 ID로 묶었습니다.'],
      ['IP 권리 분석','IP '+N.ip_count.toLocaleString('ko-KR')+'개 항목마다 개체 유형(작품·캐릭터·게임·아티스트 / 기업·채널·매장 등)과 원천 권리(자사·자회사·공동·외부·미확인), 기업의 사업 역할을 따로 분석했습니다. 중요한 항목은 공식 자료·공시로 외부 확인했습니다.'],
      ['검증·교정','v3.1~'+VER+' 단계별 재검토와 사례별 정답표 교정을 거쳤고, 바뀐 값은 모두 교정 로그(이전 값·새 값·근거 URL·사유)로 남겼습니다. 원문을 직접 열람한 사례는 '+N.raw_read_count+'건이며 나머지는 제목·요약 기준 분석입니다.'],
      ['사업 이력','같은 사업이 계획→계약→실행→성과로 실제 진전된 경우만 사건 사이를 연결했습니다. 같은 IP의 서로 다른 활동은 따로 둡니다.']];
    const can=[['기업','분야별 등재 기업과 연도별 등재 분야, 사명 변경 이력'],['IP 권리','기업이 보유한 원천 IP와 사업에 활용한 외부 IP의 구분, 기업의 역할(제작·유통·라이선스·공간 운영 등), 권리 변동 시점'],['IP·사업 자산','공시(사업보고서 표·목록)와 기사 문맥에서 개체 판별기로 찾은 소속 아티스트(그룹·개인)·소속 배우·자사 플랫폼과 유통채널·작품. 기사 근거만 있는 항목은 소속·운영 근거 문장을 확인한 것만 수록'],['IP 활용','상품화·팝업·영상화·게임 협업·공연·음원 등 IP별 실제 활용 사례와 진행 단계'],['해외','해외 실행·성과 / 국제 계약 / 계획 / 국내 도입 / 법인 처분을 나눈 국가별 기록'],['협업','실제 공동 기획·제작·마케팅이 확인된 협업과 일반 거래·플랫폼 이용의 구분, 상대 기업과 산업'],['근거','사례마다 기사 원문 URL, 원문 열람 여부, 분석 근거 수준'],['원자료','원본 엑셀(26개 시트), 전체 CSV, 분야·기업·IP 단위 CSV 내려받기']];
    const how=['왼쪽 목록(휴대폰은 위쪽 탭)에서 <b>분야</b>를 고릅니다.','분야 화면의 <b>기업 카드</b>를 누르면 기업 상세(사업 활동 개요·IP·활동 사례)가 열립니다.','<b>IP 카드</b>를 누르면 권리 구분과 그 IP의 실제 활용 사례가 나옵니다. 권리 구분 버튼으로 자사·외부 IP만 골라 볼 수 있습니다.','아래 <b>활동 사례</b>에서 범주(IP 활용·해외·협업 등)를 고르거나 검색해 사례를 좁힙니다. 각 사례의 링크로 기사 원문을 엽니다.','<b>원자료</b>에서 지금 보는 분야·기업·IP 범위의 CSV를 내려받습니다. 주소창의 링크를 복사하면 같은 화면을 공유할 수 있습니다.'];
    const cats=Object.entries(N.event_categories).map(([k,v])=>`<tr><td>${escape(k)}</td><td class="num">${Number(v).toLocaleString('ko-KR')}</td></tr>`).join('');
    const topOf=s=>D.companies.filter(c=>c.sector===s).map(c=>[c.name,(evCodes.get(c.code)||[]).filter(e=>e.category==='IP 활용').length]).sort((a,b)=>b[1]-a[1]).slice(0,3).map(x=>x[0]).join(' · ');
    byId('overview').innerHTML=`
    <div class="subsection"><div class="subhead"><h3>분야 선택</h3><span class="subnote">분야를 누르면 그 분야의 통합 데이터와 기업 목록으로 이동합니다</span></div>
      <div class="sectorgrid">${SECTORS.map(s=>`<button class="sectorcard" data-sector="${escape(s.name)}"><strong>${escape(s.name)}</strong><span class="small">기업 ${s.companies} · IP ${s.ips} · IP 활용 자료 ${s.ip_use_rows}</span><span class="small muted">${escape(topOf(s.name))}</span></button>`).join('')}</div></div>
    <details class="guide"><summary>이 자료는 어떻게 만들었나 · 확인할 수 있는 것 · 사용법 · 유의점</summary>
    <div class="subsection"><div class="subhead"><h3>이 자료는 어떻게 만들었나</h3><span class="subnote">자료 기간 ${escape(RANGE.start)}~${escape(RANGE.end)} · 갱신 ${escape(UPDATED)} · ${escape(D.version)}</span></div>
      <ol class="steps">${steps.map(([t,d])=>`<li><b>${escape(t)}</b><span>${escape(d)}</span></li>`).join('')}</ol></div>
    <div class="ovgrid">
      <div class="subsection"><div class="subhead"><h3>이 페이지로 확인할 수 있는 것</h3></div><dl class="candl">${can.map(([t,d])=>`<div><dt>${escape(t)}</dt><dd>${escape(d)}</dd></div>`).join('')}</dl></div>
      <div class="subsection"><div class="subhead"><h3>사용법</h3></div><ol class="howto">${how.map(x=>`<li>${x}</li>`).join('')}</ol></div>
    </div>
    <div class="ovgrid">
      <div class="subsection"><div class="subhead"><h3>자료 구분별 건수</h3><span class="subnote">자료행 기준(한 사건이 여러 구분에 나올 수 있음)</span></div><table class="cattable"><tbody>${cats}</tbody></table></div>
      <div class="subsection"><div class="subhead"><h3>읽을 때 유의할 점</h3></div><ul class="cautions">
        <li>기사 사건이 없는 기업은 원인을 표시합니다: ${Object.entries(NSL).map(([k,v])=>{const n=new Set(D.companies.filter(c=>newsTag(c)&&newsTag(c)[0]===v[0]).map(c=>c.code)).size;return n?`'${v[0]}' ${n}개사`:''}).filter(Boolean).join(' · ')}. 이 기업들도 실적·사업부문·공시·전략 활동 자료는 볼 수 있으며, 기사 기반 활동 0건이 활동이 없다는 뜻은 아닙니다.</li>
        <li>건수는 기사로 확인된 활동의 수입니다. 매출·계약 건수나 사업 규모를 뜻하지 않습니다.</li>
        <li>[제목·요약 기준]으로 표시된 사례는 원문을 직접 열람하지 않은 분석입니다(원문 열람 ${N.raw_read_count}건, 원문 URL ${Number(N.raw_urls).toLocaleString('ko-KR')}건).</li>
        <li>‘권리 미확인’은 근거가 부족하다는 뜻이지 IP를 보유하지 않았다는 뜻이 아닙니다. 공동 제작·라이선스·매장 운영은 소유 근거로 쓰지 않았습니다.</li>
        <li>IP 이름이 정확히 일치하는 사례만 IP 카드에 연결했습니다(연결되지 않은 IP 활용 자료 ${Number(N.unmatched_ip_case_rows).toLocaleString('ko-KR')}행은 기업 활동 목록에 있습니다).</li>
        <li>해외 박람회 참가는 행사 개최국으로, 국제 계약은 계약 단계로만 표시하며 시장 진출로 보지 않습니다.</li></ul></div>
    </div></details>`;}

  /* 기업 큰 흐름: 분야 분석(⑦) 요약 + 연도별 주요 활동 */
  /* 기업 소개: DART 사업의 개요 요약(공시 기준) */
  function introBlock(c){const x=(D.intro||{})[c.code];
    if(!x)return `<div class="subsection introbox"><div class="subhead"><h3>기업 소개</h3><span class="subnote">공시 기준</span></div><p class="hint">사업보고서의 사업의 개요가 수집되지 않아 공식 소개를 싣지 않았습니다.</p></div>`;
    return `<div class="subsection introbox"><div class="subhead"><div><div class="eyebrow">기업 소개</div><h3>어떤 회사인가</h3></div><span class="subnote">출처: ${x.url?`<a href="${escape(x.url)}" target="_blank" rel="noopener">${escape(x.rep)}</a>`:escape(x.rep)}</span></div>
      <p class="introtext">${escape(x.t)}</p>
      ${x.lines.length?`<div class="btnrow">${x.lines.map(l=>`<span class="pill blue">${escape(l)}</span>`).join(' ')}</div>`:''}
      ${x.names.length?`<p class="meta" style="margin-top:8px">공시에 나온 자사 브랜드·IP: ${x.names.map(escape).join(' · ')}</p>`:''}
      ${x.note?`<p class="hint" style="margin-top:6px">유의: ${escape(x.note)}</p>`:''}</div>`;}
  function flowBlock(c){
    const flowCandidate=(evCodes.get(c.code)||[]).length?((D.flow[state.sector]||{})[c.name] || Object.values(D.flow).map(m=>m[c.name]).find(Boolean) || ''):'';
    const t=flowCandidate && (flowOwners.get(flowCandidate)||new Set()).size===1 ? flowCandidate : ''; // 복수 회사 동일 요약문은 자동 집계로 대체
    const evs=(evCodes.get(c.code)||[]).filter(e=>e.category!=='검토 후보');
    const MAIN=['IP 활용','해외 실행·성과','국제 계약·제휴','해외 계획·추진','협업','국내 도입','해외 법인·지분 처분'];
    const years=RANGE.years.map(y=>{const ye=evs.filter(e=>(e.date||'').startsWith(y));
      const cnt=MAIN.map(k=>[k,uniqCount(ye.filter(e=>e.category===k))]).filter(x=>x[1]>0);
      const seen=new Set();const pri=e=>({'성과':0,'해외성과':0,'실행':1,'해외실행':1,'운영':2,'계약':3,'국제계약·제휴':3}[e.exec]??4);
      const top=ye.filter(e=>MAIN.includes(e.category)).sort((a,b)=>pri(a)-pri(b)||(a.date||'').localeCompare(b.date||'')).filter(e=>{const k=e.eid||e.cid;if(seen.has(k))return false;seen.add(k);return true;}).slice(0,3).sort((a,b)=>(a.date||'').localeCompare(b.date||''));
      const fqAll=(D.fin[c.code]||[]).filter(f=>f.p.startsWith(y));const fq=fqAll.filter(f=>f.rev!=null&&f.op!=null);   // 매출·영업이익 값이 있는 분기만 합산·집계(빈 값을 0으로 더하지 않음)
      const fqKeys=new Set(fq.map(f=>f.p)),fqBases=new Set(fq.map(f=>f.b||'미확인'));
      const qN=finQuarters(y),qComplete=qN>0&&Array.from({length:qN},(_,i)=>y+'Q'+(i+1)).every(q=>fqKeys.has(q));
      let fin='';
      if(!fq.length&&fqAll.length)fin=`${y} 공시 ${fqAll.length}건 · 매출 또는 영업이익 수치 미추출(합산 제외)`;
      if(fq.length){
        const basis=[...fqBases].join('/');
        if(fqBases.size>1)fin=`${y} 연결·별도 재무제표 혼재 · 합산 제외`;
        else{
          const rev=fq.reduce((s,f)=>s+(f.rev||0),0),op=fq.reduce((s,f)=>s+(f.op||0),0);
          const scope=qComplete?finScope(qN):`확인된 ${fqKeys.size}개 분기 합계(연간 아님)`;
          fin=`${y} ${scope} · ${basis} · 매출 ${fmt억(rev)} · 영업이익 ${fmt억(op)}`;
        }
      }
      return {y,cnt,top,n:uniqCount(ye),fin,disc:(D.disc[c.code]||[]).filter(e=>e.d.startsWith(y)).length,strat:(D.strat[c.code]||[]).filter(e=>e.d.startsWith(y)&&!e.rt).length};}).filter(x=>x.n>0||x.fin||x.disc);
    const auto=()=>{const ips=(ipCodes.get(c.code)||[]);const own=ips.filter(x=>['자사 원천','자회사 원천','인수 보유','공동 보유'].includes(x.rights_group)).map(x=>x.name).slice(0,3);const ext=ips.filter(x=>x.rights_group==='외부 IP').map(x=>x.name).slice(0,3);
      const parts=[];if(own.length)parts.push(`자사 IP는 ${own.join('·')}입니다.`);if(ext.length)parts.push(`활용한 외부 IP는 ${ext.join('·')}입니다.`);
      const tot=MAIN.map(k=>[k,uniqCount(evs.filter(e=>e.category===k))]).filter(x=>x[1]>0).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([k,n])=>`${k} ${n}건`);if(tot.length)parts.push(`기사로 확인된 주요 활동은 ${tot.join(', ')}입니다.`);
      if(newsTag(c))return newsTag(c)[1]+'. 기사로 확인한 사업 활동(IP 활용·해외·협업)은 아직 표시하지 않습니다. 아래 실적·사업부문·공시·전략 활동 자료는 확인할 수 있습니다.'+(parts.length?' '+parts.join(' '):'');return parts.join(' ')||'기사로 확인된 사업 활동이 적어 요약하지 않았습니다.';};
    return `<div class="subsection flowbox"><div class="subhead"><h3>주요 사업 동향</h3><span class="subnote">기사·공시 기준 요약</span></div>
      <p class="flowtext">${escape(t||auto())}</p>
      ${years.length?`<div class="flowyears">${years.map(x=>`<div class="flowyear"><div class="fy">${x.y}<span>${newsTag(c)?newsTag(c)[0]:`<button class="evlink" data-evq="y|${escape(c.code)}|${x.y}">기사 사건 ${x.n}</button>`}</span></div>${x.fin?`<div class="ffin">${escape(x.fin)}</div>`:''}<div class="fcnt">${x.cnt.map(([k,n])=>`<button class="pill evlink" data-evq="y|${escape(c.code)}|${x.y}|${escape(k)}">${escape(k)} ${n}</button>`).join(' ')}${x.disc?`<span class="pill">공시 ${x.disc}</span>`:''}${x.strat?`<span class="pill blue">전략 활동 ${x.strat}</span>`:''}</div><ul>${x.top.map(e=>`<li><span class="meta">${escape((e.date||'').slice(5))}</span> ${escape(e.title)} <span class="meta">· ${escape(e.exec||e.stage||'')}</span></li>`).join('')}</ul></div>`).join('')}</div>`:''}
    </div>`;}

  /* 통합 데이터(전체 또는 분야): 구분별 비교표 · 활동 사례 통합 검색 · IP 통합 탐색 */
  const scopeCodes=()=>state.sector===ALL?new Set(codeCompanies.keys()):getSectorCodes();
  const scopeEvents=()=>{const cs=scopeCodes();return D.events.filter(e=>cs.has(e.code));};
  const scopeIps=()=>{const cs=scopeCodes();return D.ips.filter(x=>cs.has(x.code));};
  const firstSector=code=>{const c=D.companies.find(x=>x.code===code);return c?c.sector:SECTORS[0].name;};
  function goCompany(code,rid){if(state.sector===ALL||!getSectorCompanies().some(x=>x.code===code))state.sector=firstSector(code);
    state.code=code;state.companySearch='';resetIp();if(rid)state.rid=rid;updateHash();renderSector();byId('profile').scrollIntoView({behavior:'smooth',block:'start'});}
  /* 본문 검색: data.js의 bidx(기사 URL → 본문 낱말 목록, 본문 문장은 없음). 끝 조사 떼기는 content_hub/portal_body_index.py와 같은 규칙 */
  const BODY_PARTS=['으로','에서','에게','까지','부터','보다','처럼','은','는','이','가','을','를','에','의','와','과','도','로','만'];
  const normBodyWord=w=>{for(const p of BODY_PARTS)if(w.endsWith(p)&&w.length-p.length>=2)return w.slice(0,-p.length);return w;};
  let BODYHIT=new Set();
  /* 검색 같은 뜻 묶음(data.js syn·synName ← content_hub/portal_search_syn.py, 용어 표는 5_통합/sector_pages/v34/search_synonyms.csv).
     일반 용어(syn): 낱말이 표기와 같거나, 앞부분이 같거나(2자 이상), 끝 조사만 다르면 묶음 전체 표기로 찾는다.
     이름(synName: 아티스트·IP 한글/영문·본명·띄어쓰기 변형): 같거나 끝 조사만 다를 때만. 영문·숫자 3자 이하는 낱말 경계로(IP≠IPO) */
  const lowG=G=>(G||[]).map(g=>g.map(x=>String(x).toLowerCase()));const SYNG=lowG(D.syn),SYNN=lowG(D.synName);const nsp=x=>x.replace(/\s+/g,'');
  function synAlts(w){const out=new Set([w]),wn=nsp(w),wb=normBodyWord(wn);
    const add=g=>g.forEach(t=>{out.add(t);if(/\s/.test(t))out.add(nsp(t));});
    for(const g of SYNG)if(g.some(t=>{const tn=nsp(t);return tn===wn||tn===wb||(wn.length>=2&&tn.startsWith(wn));}))add(g);
    for(const g of SYNN)if(g.some(t=>{const tn=nsp(t);return tn===wn||tn===wb;}))add(g);
    return [...out];}
  const qWords=q=>String(q||'').trim().toLowerCase().split(/\s+/).filter(Boolean).map(synAlts);
  const qFlat=q=>[...new Set(qWords(q).flat())];
  const reEsc=x=>x.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const RECACHE=new Map();const wre=a=>{let r=RECACHE.get(a);if(!r){r=new RegExp('(^|[^a-z0-9])'+reEsc(a)+'([^a-z0-9]|$)');RECACHE.set(a,r);}return r;};
  const hasW=(f,a)=>/^[a-z0-9]{1,3}$/.test(a)?(f.includes(a)&&wre(a).test(f)):f.includes(a);
  const synHint=q=>{const W=String(q||'').trim().toLowerCase().split(/\s+/).filter(Boolean),A=qWords(q);
    const show=(a,w)=>{const m=new Map();a.forEach(t=>{const k=nsp(t);if(k!==nsp(w)&&!m.has(k))m.set(k,t);});return [...m.values()];};
    const parts=A.map((a,i)=>{const x=show(a,W[i]);return x.length?`<b>${escape(W[i])}</b> → ${escape(x.slice(0,8).join('·'))}${x.length>8?' 외 '+(x.length-8):''}`:'';}).filter(Boolean);
    return parts.length?`<p class="hint synhint">같은 뜻 함께 검색: ${parts.join(' / ')}</p>`:'';};
  /* 0=불일치, 1=제목·요약·항목 일치, 2=본문 낱말 목록으로만 일치(모든 낱말이 항목 또는 본문에 있어야 함) */
  function evMatch(e,words){const f=[e.title,e.firm,e.ip,e.country,e.partner,e.output,e.use,e.sm].join(' ').toLowerCase();
    if(words.every(A=>A.some(w=>hasW(f,w))))return 1;const b=(D.bidx||{})[e.url];if(!b)return 0;
    return words.every(A=>A.some(w=>hasW(f,w)||hasW(b,normBodyWord(nsp(w)))))?2:0;}
  function aggEventsFiltered(){const words=qWords(state.aggSearch);BODYHIT=new Set();
    return scopeEvents().filter(e=>{if(!catOk(e,state.aggCat))return false;if(state.aggCat==='IP 활용'&&!useOk(e,state.aggUse))return false;if(!words.length)return true;const m=evMatch(e,words);if(m===2)BODYHIT.add(e);return m>0;}).slice().sort((a,b)=>(b.date||'').localeCompare(a.date||''));}
  function aggIpsFiltered(){const q=state.aggIpSearch.trim().toLowerCase();return scopeIps().filter(x=>(state.aggIpGroup==='전체'||x.rights_group===state.aggIpGroup)&&(!q||[x.name,x.firm,x.aliases,x.parent].join(' ').toLowerCase().includes(q)));}
  const subK=code=>{const v=((D.secRel||{})[code]||{})[state.sector];return v&&v.main===false?1:0;};
  let AGGROWS={};
  function renderAggregate(){const isAll=state.sector===ALL,ev=scopeEvents(),MAIN=['IP 활용','해외 실행·성과','국제 계약·제휴','해외 계획·추진','국내 도입','협업','거래 관계'];
    const rows=isAll?SECTORS.map(s=>{const cs=new Set(D.companies.filter(c=>c.sector===s.name).map(c=>c.code));const e=ev.filter(x=>cs.has(x.code));return {key:s.name,label:s.name,n:s.companies,e,codes:[...cs],btn:`data-sector="${escape(s.name)}"`};})
      :getSectorCompanies().map(c=>{const e=ev.filter(x=>x.code===c.code);return {key:c.code,label:c.name,n:(ipCodes.get(c.code)||[]).length,e,codes:[c.code],btn:`data-company="${escape(c.code)}"`};}).sort((a,b)=>(subK(a.key)-subK(b.key))||uniqCount(b.e)-uniqCount(a.e));
    AGGROWS={};rows.forEach(r=>AGGROWS[r.key]={e:r.e,label:r.label});
    const tbl=`<div class="tablewrap"><table class="aggtable"><thead><tr><th>${isAll?'분야':'기업'}</th><th>${isAll?'기업':'IP'}</th>${MAIN.map(k=>`<th>${escape(k)}</th>`).join('')}<th>사건 ID 합계</th><th>공시</th><th>전략 활동</th><th>아티스트·배우</th><th>플랫폼·채널</th></tr></thead><tbody>${rows.map(r=>`<tr><td><button class="linkbtn" ${r.btn}>${escape(r.label)}</button>${!isAll&&subK(r.key)?` <span class="meta">주 분야: ${escape(Object.entries((D.secRel||{})[r.key]||{}).filter(([k,v])=>v.main).map(([k])=>k).join('·'))}</span>`:''}</td><td class="num">${r.n}</td>${MAIN.map(k=>{const n=uniqCount(r.e.filter(x=>x.category===k));return `<td class="num">${n?`<button class="evlink" data-evq="t|${escape(r.key)}|${escape(k)}">${n}</button>`:''}</td>`;}).join('')}<td class="num b"><button class="evlink" data-evq="t|${escape(r.key)}">${uniqCount(r.e)}</button></td><td class="num">${r.codes.reduce((s,c)=>s+(D.disc[c]||[]).length,0)||''}</td><td class="num">${r.codes.reduce((s,c)=>s+(D.strat[c]||[]).filter(x=>!x.rt).length,0)||''}</td><td class="num">${r.codes.reduce((s,c)=>s+((D.assets||{})[c]||[]).filter(x=>/아티스트|배우/.test(x.k)).length,0)||''}</td><td class="num">${r.codes.reduce((s,c)=>s+((D.assets||{})[c]||[]).filter(x=>x.k==='플랫폼·유통채널').length,0)||''}</td></tr>`).join('')}</tbody></table></div>`;
    byId('aggregate').innerHTML=`
    <div class="subsection"><div class="subhead"><div><div class="eyebrow">통합 데이터 · ${escape(isAll?'전체':state.sector)}</div><h3>${isAll?'분야별 비교':'기업별 비교'}</h3></div><span class="subnote">사건 ID 기준 건수 · 이름을 누르면 ${isAll?'분야':'기업 상세'}로 이동</span></div>${tbl}</div>
    <div class="subsection"><div class="subhead"><div><div class="eyebrow">통합 데이터</div><h3>활동 사례 통합 검색</h3></div><button class="tinybtn" id="aggEventsCsv">현재 조건 CSV ↓</button></div><div class="toolbarfit" id="aggFilters"></div><div id="aggEvents"></div></div>
    <div class="subsection"><div class="subhead"><div><div class="eyebrow">통합 데이터</div><h3>IP 통합 탐색</h3></div><button class="tinybtn" id="aggIpCsv">현재 조건 CSV ↓</button></div><div class="toolbarfit" id="aggIpFilters"></div><div id="aggIps"></div></div>`;
    renderAggEvents();renderAggIps();}
  /* 통합 검색: 공시(보고서명·원문 항목 내용) */
  function scopeDisc(){const cs=new Set(scopeEvents().map(e=>e.code));const codes=state.sector===ALL?D.companies.map(c=>c.code):D.companies.filter(c=>c.sector===state.sector).map(c=>c.code);
    const nm=Object.fromEntries(D.companies.map(c=>[c.code,c.name]));const out=[];[...new Set(codes)].forEach(c=>(D.disc[c]||[]).forEach(x=>out.push({...x,code:c,firm:nm[c]})));return out;}
  function aggDiscFiltered(){const W=qWords(state.aggSearch);return scopeDisc().filter(x=>{if(!W.length)return true;const f=[x.t,x.k,x.ft||'',x.firm].join(' ').toLowerCase();return W.every(A=>A.some(w=>hasW(f,w)));}).sort((a,b)=>(b.d||'').localeCompare(a.d||''));}
  const snip=(text,qs)=>{if(!qs||!text)return '';qs=[].concat(qs);const t=String(text),tl=t.toLowerCase();let i=-1,q='';for(const a of qs){const j=tl.indexOf(a);if(j>=0&&(i<0||j<i)){i=j;q=a;}}if(i<0)return '';const s0=Math.max(0,i-40),e0=Math.min(t.length,i+q.length+60);
    return `${s0>0?'…':''}${escape(t.slice(s0,i))}<mark>${escape(t.slice(i,i+q.length))}</mark>${escape(t.slice(i+q.length,e0))}${e0<t.length?'…':''}`;};
  const srcBar=n=>`<div class="srcbar">${['기사','공시'].map(k=>`<button class="filterbtn ${(state.aggSrc||'기사')===k?'active':''}" data-aggsrc="${k}">${k==='기사'?'기사 사례':'공시'}${k===(state.aggSrc||'기사')?' '+n.toLocaleString('ko-KR'):''}</button>`).join('')}</div>`;
  function renderAggEvents(){const src=state.aggSrc||'기사';const q=state.aggSearch.trim().toLowerCase(),QA=qFlat(state.aggSearch);
    if(src==='공시'){const L=aggDiscFiltered();
      byId('aggFilters').innerHTML=srcBar(L.length)+`<input class="searchinput" id="aggSearch" placeholder="공시 제목·원문 내용 검색(예: 조달청, 유상증자, 해외)" value="${escape(state.aggSearch)}">`;
      byId('aggEvents').innerHTML=`<p class="hint" style="margin:4px 0 12px">공시 ${L.length.toLocaleString('ko-KR')}건 · 최신순 · DART 원문 표 내용까지 검색합니다.</p>`+synHint(state.aggSearch)+(L.length?L.slice(0,state.aggLimit).map(x=>`<div class="aggitem"><button class="firmtag" data-gocompany="${escape(x.code)}|">${escape(x.firm)} →</button><div><span class="meta">${escape(x.d)}</span> <span class="pill">${escape(x.k)}</span> ${x.u?`<a href="${escape(x.u)}" target="_blank" rel="noopener">${escape(x.t)}</a>`:escape(x.t)}${(x.x||[]).length?`<div class="meta discline">${escape(discLine(x))}</div>`:''}${q&&snip(x.ft,QA)?`<div class="snip">원문: ${snip(x.ft,QA)}</div>`:''}</div></div>`).join(''):'<div class="empty">조건에 맞는 공시가 없습니다.</div>')+(L.length>state.aggLimit?`<button class="more" data-aggmore="1">20건 더 보기 (전체 ${L.length.toLocaleString('ko-KR')}건)</button>`:'');
      const s2=byId('aggSearch');if(s2&&document.activeElement!==s2&&state.aggSearch)s2.value=state.aggSearch;return;}
    const all=aggEventsFiltered(),base=scopeEvents();
    byId('aggFilters').innerHTML=srcBar(uniqCount(all))+categories.map(g=>{const n=g==='전체'?uniqCount(base):uniqCount(base.filter(e=>catOk(e,g)));return n||g==='전체'?`<button class="filterbtn ${state.aggCat===g?'active':''}" data-aggcat="${escape(g)}">${escape(g)} ${n}</button>`:'';}).join('')+(state.aggCat==='IP 활용'?useBar(base.filter(e=>e.category==='IP 활용'),state.aggUse,'agguse'):'')+`<input class="searchinput" id="aggSearch" placeholder="기업·IP·국가·상대·제목·기사 요약·본문 검색" value="${escape(state.aggSearch)}">`;
    byId('aggEvents').innerHTML=`<p class="hint" style="margin:4px 0 12px">${all.length.toLocaleString('ko-KR')}개 자료행 · ${uniqCount(all).toLocaleString('ko-KR')}개 사건 ID · 최신순</p>`+synHint(state.aggSearch)+(Object.keys(D.bidx||{}).length?`<p class="hint bodyhint">본문 검색은 기사 본문의 낱말 목록으로 찾습니다(본문은 원문 링크에서 확인). 본문을 받은 기사 ${((D.bidxInfo||{}).n||Object.keys(D.bidx).length).toLocaleString('ko-KR')}건</p>`:'')+(all.length?all.slice(0,state.aggLimit).map(e=>`<div class="aggitem"><button class="firmtag" data-gocompany="${escape(e.code)}|${escape(e.rid||'')}">${escape(e.firm)} →</button>${BODYHIT.has(e)?'<span class="bodyhit" title="기사 본문 낱말 목록에서 일치(본문은 원문 링크에서 확인)">본문 일치</span>':''}${evidence(e)}${q&&!QA.some(a=>String(e.title||'').toLowerCase().includes(a))&&snip(e.sm,QA)?`<div class="snip">기사 요약: ${snip(e.sm,QA)}</div>`:''}</div>`).join(''):'<div class="empty">조건에 맞는 사례가 없습니다.</div>')+(all.length>state.aggLimit?`<button class="more" data-aggmore="1">20건 더 보기 (전체 ${all.length.toLocaleString('ko-KR')}개 자료행)</button>`:'');
    const s=byId('aggSearch');if(s&&document.activeElement!==s&&state.aggSearch)s.value=state.aggSearch;}
  function renderAggIps(){const all=aggIpsFiltered(),base=scopeIps(),evs=scopeEvents();
    byId('aggIpFilters').innerHTML=groups.map(g=>{const n=g==='전체'?base.length:base.filter(x=>x.rights_group===g).length;return `<button class="filterbtn ${state.aggIpGroup===g?'active':''}" data-aggipgroup="${escape(g)}">${escape(g)} ${n}</button>`;}).join('')+`<input class="searchinput" id="aggIpSearch" placeholder="IP·기업 검색" value="${escape(state.aggIpSearch)}">`;
    const useN=new Map();for(const e of evs)if(e.rid&&e.category==='IP 활용'){useN.set(e.rid,(useN.get(e.rid)||new Set()).add(e.eid||e.cid));}
    const sorted=all.slice().sort((a,b)=>((useN.get(b.rid)||new Set()).size-(useN.get(a.rid)||new Set()).size)||a.name.localeCompare(b.name,'ko'));
    byId('aggIps').innerHTML=`<p class="hint" style="margin:4px 0 12px">${all.length.toLocaleString('ko-KR')}개 IP · 활용 사건이 많은 순 · 카드를 누르면 해당 기업의 IP 상세로 이동</p><div class="ipgrid">${sorted.slice(0,state.aggIpLimit).map(ip=>`<button class="ipcard" data-gocompany="${escape(ip.code)}|${escape(ip.rid)}">${niceRights(ip.rights_group)}<strong>${escape(ip.name)}</strong><span class="small">${escape(ip.firm)} · ${escape(ip.role||'역할 미확인')}</span><span class="ipbottom"><span class="small">활용 사건 ${(useN.get(ip.rid)||new Set()).size}개</span><span class="small">${escape(ip.source_level||'')}</span></span></button>`).join('')}</div>`+(all.length>state.aggIpLimit?`<button class="more" data-aggipmore="1">36개 더 보기 (전체 ${all.length.toLocaleString('ko-KR')}개)</button>`:'');}

  const fmt억=v=>v==null||isNaN(v)?'—':(Math.abs(v)>=10000?(v/10000).toFixed(2).replace(/\.?0+$/,'')+'조':Math.round(v).toLocaleString('ko-KR')+'억');
  const pct=v=>v==null||isNaN(v)?'':`${v>0?'+':''}${v.toFixed(1)}%`;
  /* 사업 활동 전체: 실적 · 사업부문·매출 구성 · 공시 · 전략 활동 · 통합 연혁 */
  /* IP·사업 자산: 개체 판별기(v3.3a)로 공시·기사에서 찾은 아티스트·소속 배우·플랫폼·작품 */
  const AK=['아티스트(그룹)','아티스트(개인)','소속 아티스트·배우','플랫폼·유통채널','작품','캐릭터','브랜드·레이블','제작 인력'];
  const RG=a=>{const r=a.r||'';if(r==='소속')return '소속';if(r==='과거 소속')return '과거 소속';if(/제작/.test(r))return '제작';if(/개발|퍼블리싱|유통/.test(r))return '개발·퍼블리싱·유통';if(r==='운영')return '운영';if(/참여 인력/.test(r))return '참여 인력';if(/공시상/.test(r))return '공시 목록(역할 미확인)';return '기타(주최·협력 등)';};
  const RGS=['소속','제작','개발·퍼블리싱·유통','운영','참여 인력','공시 목록(역할 미확인)','과거 소속','기타(주최·협력 등)'];
  function assetBlock(c){const all=(D.assets||{})[c.code]||[];const k=state.assetKind||'전체';const rg=state.assetRel||'전체';const q=(state.assetSearch||'').toLowerCase();
    const cnt=AK.map(x=>[x,all.filter(a=>a.k===x).length]).filter(x=>x[1]);
    const rcnt=RGS.map(x=>[x,all.filter(a=>RG(a)===x).length]).filter(x=>x[1]);
    let list=all.filter(a=>(k==='전체'||a.k===k)&&(rg==='전체'||RG(a)===rg)&&(!q||(a.n+' '+(a.g||'')).toLowerCase().includes(q)));
    list=list.slice().sort((a,b)=>AK.indexOf(a.k)-AK.indexOf(b.k)||b.o-a.o);const lim=state.assetLimit||40;
    const body=!all.length?`<div class="empty">${newsTag(c)?newsTag(c)[1]+'. ':''}공시·기사에서 판별된 아티스트·플랫폼·작품 자료가 없습니다.</div>`:
      `<div class="assetgrid">${list.slice(0,lim).map(a=>`<div class="assetitem"><div class="assettop"><strong>${escape(a.n)}</strong><span class="pill ${a.e.startsWith('공시')?'blue':''}">${escape(a.e)}</span></div><div class="btnrow" style="margin:4px 0"><span class="pill blue">${escape(a.r||'')}</span><span class="pill">${escape(a.rt||'')}</span></div><div class="meta">${escape(a.k)}${a.g?` · ${escape(a.g)}`:''}${a.y?` · 공시 ${escape(a.y)}`:''}${a.o?` · 기사 ${a.o.toLocaleString('ko-KR')}건`:''}</div>${a.m?`<div class="meta">${escape(a.m)}</div>`:''}${a.rb?`<div class="meta">역할 근거: ${escape(a.rb)}</div>`:''}${a.c?`<div class="assetctx">${escape(a.c)}</div>`:''}</div>`).join('')}</div>${list.length>lim?`<button class="more" data-assetmore="1">40건 더 보기 (전체 ${list.length.toLocaleString('ko-KR')}건)</button>`:''}`;
    return `<div class="subsection" id="assetBox"><div class="subhead"><div><h3>IP·사업 자산</h3></div></div>
      <div class="toolbarfit"><span class="meta" style="margin-right:6px">역할</span>${[['전체',all.length],...rcnt].map(([x,n])=>`<button class="filterbtn ${rg===x?'active':''}" data-assetrel="${escape(x)}">${escape(x)} ${n}</button>`).join('')}</div><div class="toolbarfit"><span class="meta" style="margin-right:6px">종류</span>${[['전체',all.length],...cnt].map(([x,n])=>`<button class="filterbtn ${k===x?'active':''}" data-assetkind="${escape(x)}">${escape(x)} ${n}</button>`).join('')}<input class="search" id="assetSearch" placeholder="이름 검색" value="${escape(state.assetSearch||'')}"></div>${body}</div>`;}
  /* 공시 원문에서 뽑은 항목 표시: 금액(원)은 억 원으로, 항목명 앞 묶음 이름은 정리 */
  const kvLab=k=>String(k).replace(/^(계약내역|해지내역|2\.\s*)\s*/,'').replace(/\s*\((원|%|주)\)/,'').replace(/^-\s*/,'').trim();
  const kvVal=(k,v)=>{const n=Number(String(v).replace(/,/g,''));if(/금액|\(원\)|총액|원\)/.test(k)&&isFinite(n)&&Math.abs(n)>=1e6)return fmt억(n/1e8);if(/%|대비|비율/.test(k)&&isFinite(n))return n+'%';return v;};
  const discKV=x=>(x.x||[]).map(([k,v])=>[kvLab(k),kvVal(k,v)]);
  const discLine=x=>discKV(x).slice(0,4).map(([k,v])=>`${k} ${v}`).join(' · ');
  /* 공시 종류별 쉬운 설명(보고서명 기준) */
  const DISCX=[[/단일판매|공급계약/,'일정 규모 이상의 판매·공급 계약을 맺었을 때 내는 공시입니다.'],[/무상증자/,'주주에게 새 주식을 무상으로 나눠 주기로 했을 때 내는 공시입니다.'],[/유상증자/,'새 주식을 발행해 자금을 조달하기로 했을 때 내는 공시입니다.'],
   [/전환사채/,'주식으로 바꿀 수 있는 사채를 발행해 자금을 조달할 때 내는 공시입니다.'],[/신주인수권부사채/,'새 주식을 살 권리가 붙은 사채를 발행할 때 내는 공시입니다.'],[/교환사채/,'회사가 가진 주식으로 바꿀 수 있는 사채를 발행할 때 내는 공시입니다.'],
   [/자기주식.*소각|주식소각/,'회사가 가진 자기 주식을 없애기로 했을 때 내는 공시입니다.'],[/자기주식.*처분/,'회사가 가진 자기 주식을 팔기로 했을 때 내는 공시입니다.'],[/자기주식/,'회사가 자기 주식을 사들이기로 했을 때 내는 공시입니다.'],
   [/타법인주식|출자증권/,'다른 회사의 주식이나 출자증권을 사거나 팔기로 했을 때 내는 공시입니다.'],[/합병/,'다른 회사와 합치기로 했을 때 내는 공시입니다.'],[/분할/,'회사를 나누거나 주식을 나누기로 했을 때 내는 공시입니다.'],
   [/영업양수|영업양도/,'다른 회사의 영업을 넘겨받거나 넘기기로 했을 때 내는 공시입니다.'],[/최대주주/,'최대주주가 바뀌었을 때 내는 공시입니다.'],[/대표이사/,'대표이사가 바뀌었을 때 내는 공시입니다.'],
   [/매출액또는손익구조|손익구조/,'매출액이나 손익이 직전 연도보다 크게(일반 법인 30%, 대규모 법인 15% 이상) 바뀌었을 때 내는 공시입니다.'],[/배당/,'배당을 결정했을 때 내는 공시입니다.'],[/소송/,'소송이 제기되거나 판결이 나왔을 때 내는 공시입니다.'],
   [/대량보유/,'누군가 회사 주식을 5% 이상 갖게 되거나 보유 비율이 바뀌었을 때 내는 공시입니다.'],[/임원|주요주주/,'임원이나 주요주주의 주식 보유가 바뀌었을 때 내는 공시입니다.'],[/투자판단/,'투자 판단에 영향을 줄 수 있는 주요 경영 사항을 알리는 공시입니다.'],
   [/유형자산/,'일정 규모 이상의 유형자산을 사거나 팔기로 했을 때 내는 공시입니다.'],[/신규시설/,'새 시설에 투자하기로 했을 때 내는 공시입니다.'],[/감자/,'자본금을 줄이기로 했을 때 내는 공시입니다.'],[/병합/,'주식을 합쳐 주식 수를 줄이기로 했을 때 내는 공시입니다.'],
   [/기업설명회|IR/,'기업설명회를 연다는 공시입니다.'],[/잠정|실적/,'분기·연간 실적을 미리 알리는 공시입니다.'],[/거래정지|관리종목|상장폐지/,'주식 거래 정지·관리종목 지정 등 상장 상태와 관련된 공시입니다.']];
  const discExplain=t=>{const x=DISCX.find(([r])=>r.test(t||''));return (x?x[1]:'DART에 제출된 공시입니다.')+(/정정/.test(t||'')?' 앞서 낸 공시의 내용을 고친 것입니다.':'');};
  /* 기업 히스토리 타임라인: 실제 날짜 위치에 사건 점, IP별 색, 마우스를 올리면 내용, 누르면 원문 */
  const TLPAL=['#2f6bd8','#e0782f','#1f9467','#8b55d0','#c2417a','#0e8fa3'];const TLNOIP='#b8c0cc',TLOTHER='#8a93a3';
  let TLI=[];
  function timelineBlock(c){
    const S=Date.parse(RANGE.start.slice(0,8)+'01'),E=Date.parse(new Date(Date.UTC(RANGE.y1,RANGE.m1,0)).toISOString().slice(0,10)),pct=d=>Math.max(0,Math.min(100,(Date.parse(d)-S)/(E-S)*100));
    const seen=new Set(),evs=[];(evCodes.get(c.code)||[]).forEach(e=>{if(e.category==='검토 후보'||!e.date)return;const k=e.eid||e.cid;if(seen.has(k))return;seen.add(k);evs.push(e);});
    const nzk=x=>String(x).replace(/[\s·:\-_.,!?~()\[\]'"]/g,'').toLowerCase();const amap=new Map();
    ((D.assets||{})[c.code]||[]).forEach(a=>{const al=String(a.g||'').split(',').map(x=>x.trim()).filter(Boolean);const ko=/[가-힣]/.test(a.n)?a.n:(al.find(x=>/[가-힣]/.test(x))||a.n);[a.n,...al].forEach(x=>amap.set(nzk(x),ko));});
    const DER=(D.ipDerived||{})[c.code]||{};const derOf=n=>DER[n]||null;
    const canonIp=n=>{const dv=derOf(n);const base=dv?dv.p:n;return amap.get(nzk(base))||base;};
    const ipLabel=e=>(e.ip_names||'').split('·').map(x=>x.trim()).filter(Boolean).map(n=>derOf(n)?`${n}(${canonIp(n)} ${derOf(n).rel})`:canonIp(n)).filter((v,i,a)=>a.indexOf(v)===i).join(' · ');
    const ipsOf=e=>[...new Set((e.ip_names||'').split('·').map(x=>x.trim()).filter(Boolean).map(canonIp))];
    const cnt=new Map();evs.forEach(e=>ipsOf(e).forEach(n=>cnt.set(n,(cnt.get(n)||0)+1)));
    const tops=[...cnt.entries()].sort((a,b)=>b[1]-a[1]).slice(0,6).map(x=>x[0]);
    const colOf=e=>{const L=ipsOf(e);const t=L.find(n=>tops.includes(n));return t?TLPAL[tops.indexOf(t)]:(L.length?TLOTHER:TLNOIP);};
    TLI=[];
    const lane=(items,shape)=>{const bins=new Map();let maxS=0;
      const marks=items.sort((a,b)=>a.d.localeCompare(b.d)).map(it=>{const x=pct(it.d);const b=Math.round(x/0.9);const k=bins.get(b)||0;bins.set(b,k+1);maxS=Math.max(maxS,k+1);
        const id=TLI.push(it)-1;return `<span class="tlmk ${shape}" data-tli="${id}" data-ip="${escape(it.ipkey||'')}" style="left:${x}%;bottom:${6+Math.min(k,9)*11}px;background:${it.col}"></span>`;}).join('');
      return [marks,Math.max(34,Math.min(maxS,10)*11+14)];};
    const A=evs.map(e=>({d:e.date,col:colOf(e),ipkey:ipsOf(e).find(n=>tops.includes(n))||'',kind:e.category,ip:ipLabel(e),t:e.title,u:e.url,stage:e.stage||'',sm:e.sm||'',src:'기사'}));
    const near=d0=>evs.filter(e=>Math.abs(Date.parse(e.date)-Date.parse(d0))<=3*864e5).sort((x,y)=>Math.abs(Date.parse(x.date)-Date.parse(d0))-Math.abs(Date.parse(y.date)-Date.parse(d0))).slice(0,2);
    const Bd=(D.disc[c.code]||[]).map(e=>({d:e.d,col:'#4b5563',kind:'공시 · '+e.k,t:e.t,u:e.u,sm:discExplain(e.t),src:'공시',kv:discKV(e),rel:near(e.d).map(x=>({d:x.date,t:x.title,u:x.url,sm:x.sm||''}))}));
    const Bs=(D.strat[c.code]||[]).filter(e=>!e.rt).map(e=>({d:e.d,col:'#0f766e',kind:'전략 활동 · '+e.k,t:e.t,u:e.u,sm:e.sm||'',src:'기사',shape:'dia'}));
    const [ma,ha]=lane(A,'');const [mb,hb]=lane([...Bd,...Bs.map(x=>({...x}))],'sq');
    const yrs=RANGE.years.flatMap(y=>RANGE.years.length>4?[y+'-01-01']:[y+'-01-01',y+'-07-01']).filter(d=>d>=RANGE.start.slice(0,8)+'01'&&d<=RANGE.end).map(d=>`<span class="tlgrid2 ${d.endsWith('01-01')?'yr':''}" style="left:${pct(d)}%"><b>${d.endsWith('01-01')?d.slice(0,4):d.slice(0,4).slice(2)+'.7'}</b></span>`).join('');
    const legend=`${tops.map((n,i)=>`<button class="tllg" data-tlip="${escape(n)}"><i style="background:${TLPAL[i]}"></i>${escape(n)} ${cnt.get(n)}</button>`).join('')}${evs.some(e=>ipsOf(e).length&&!ipsOf(e).some(n=>tops.includes(n)))?`<span class="tllg0"><i style="background:${TLOTHER}"></i>그 밖의 IP</span>`:''}${evs.some(e=>!ipsOf(e).length)?`<span class="tllg0"><i style="background:${TLNOIP}"></i>특정 IP 없음</span>`:''}`;
    if(!evs.length&&!Bd.length&&!Bs.length)return '';
    return `<div class="subsection" id="tlBox"><div class="subhead"><div><div class="eyebrow">기업 히스토리</div><h3>사업 활동 흐름</h3></div><span class="subnote">점에 마우스를 올리면 제목, 누르면 요약과 원문 링크</span></div>
      <div class="tllegend">${legend}</div>
      <div class="tlwrap2"><div class="tlinner">${yrs}
        <div class="tllane" style="height:${ha}px"><span class="tlname">사업 활동 ${evs.length}건</span>${ma}</div>
        <div class="tllane thin" style="height:${hb}px"><span class="tlname">공시 ${Bd.length} · 전략 활동 ${Bs.length}</span>${mb}</div>
      </div></div><p class="meta">사업 활동은 기사로 확인된 IP 활용·해외·협업·거래 사건(같은 사건의 반복 보도는 하나로). IP 색은 사건이 많은 IP 상위 ${tops.length}개이며, 이름을 누르면 그 IP만 강조됩니다. 아래 줄의 네모는 공시, 마름모는 전략 활동입니다.</p>
      <div id="tlCard" class="tlcard" hidden></div><div id="tlTip" class="tltip" hidden></div></div>`;}
  if(!window.__tlInit){window.__tlInit=1;
    document.addEventListener('mouseover',e=>{const m=e.target.closest&&e.target.closest('.tlmk');const tip=document.getElementById('tlTip');if(!tip)return;if(!m){tip.hidden=true;return;}
      const it=TLI[+m.dataset.tli];if(!it)return;tip.innerHTML=`<div class="meta">${escape(it.d)} · ${escape(it.kind)}${it.stage?' · '+escape(it.stage):''}</div>${it.ip?`<div class="tlipn">${escape(it.ip)}</div>`:''}<div>${escape(it.t||'')}</div><div class="meta">누르면 자세히 보기</div>`;
      const box=document.getElementById('tlBox').getBoundingClientRect(),r=m.getBoundingClientRect();tip.hidden=false;const w=Math.min(320,box.width-20);tip.style.width=w+'px';
      tip.style.left=Math.max(8,Math.min(box.width-w-8,r.left-box.left-w/2))+'px';tip.style.top=(r.top-box.top-tip.offsetHeight-10)+'px';});
    document.addEventListener('click',e=>{const m=e.target.closest&&e.target.closest('.tlmk');if(m){const it=TLI[+m.dataset.tli];const cd=document.getElementById('tlCard');if(!it||!cd)return;document.querySelectorAll('.tlmk.sel').forEach(x=>x.classList.remove('sel'));m.classList.add('sel');
      cd.innerHTML=`<div class="tlcardhead"><span class="meta">${escape(it.d)} · ${escape(it.kind)}${it.stage?' · '+escape(it.stage):''}</span><button class="evclose" data-tlclose="1" aria-label="닫기">×</button></div>${it.ip?`<div class="tlipn">${escape(it.ip)}</div>`:''}<div class="tlcardtitle">${escape(it.t||'')}</div>${it.sm?`<p class="tlcardsm">${escape(it.sm)}${it.src==='기사'&&it.sm.length>=220?'…':''}</p>`:''}${(it.kv||[]).length?`<table class="kvt">${it.kv.map(([k,v])=>`<tr><th>${escape(k)}</th><td>${escape(v)}</td></tr>`).join('')}</table><p class="meta">DART 공시 원문 표에서 옮긴 값입니다.</p>`:''}${it.u?`<a href="${escape(it.u)}" target="_blank" rel="noopener">${it.src==='공시'?'DART 공시 원문 보기':'기사 원문 보기'} ↗</a>`:'<span class="meta">원문 링크 없음</span>'}${(it.rel||[]).length?`<div class="tlrel"><div class="meta">같은 시기 기사(공시 전후 3일)</div>${it.rel.map(r=>`<div class="tlrelitem"><span class="meta">${escape(r.d)}</span> ${r.u?`<a href="${escape(r.u)}" target="_blank" rel="noopener">${escape(r.t)}</a>`:escape(r.t)}${r.sm?`<div class="tlcardsm">${escape(r.sm.slice(0,140))}…</div>`:''}</div>`).join('')}</div>`:''}`;cd.hidden=false;return;}
      if(e.target.closest&&e.target.closest('[data-tlclose]')){const cd=document.getElementById('tlCard');if(cd)cd.hidden=true;document.querySelectorAll('.tlmk.sel').forEach(x=>x.classList.remove('sel'));return;}
      const lg=e.target.closest&&e.target.closest('.tllg');if(lg){const box=document.getElementById('tlBox');const on=lg.classList.toggle('on');box.querySelectorAll('.tllg').forEach(x=>{if(x!==lg)x.classList.remove('on');});
        box.querySelectorAll('.tllane:not(.thin) .tlmk').forEach(mk=>mk.classList.toggle('dim',on&&mk.dataset.ip!==lg.dataset.tlip));}});}
  /* 분기 실적 그래프: 매출 막대 + 영업이익 막대(0 기준) 두 개를 같은 가로축으로(이중 축 사용 안 함) */
  function finChart(f){const Q=f.slice().sort((a,b)=>a.p.localeCompare(b.p));const n=Q.length;if(!n)return '';
    /* 매출·영업이익을 같은 세로축(억원)의 분기별 묶음 막대로 — 두 값 모두 금액이라 축 하나 */
    const W=Math.max(560,n*52),L=62,Rr=12,cw=(W-L-Rr)/n,bw=Math.min(16,cw*0.32),T=22,H=200;
    const vals=Q.flatMap(q=>[q.rev,q.op]).filter(v=>v!=null);const mx=Math.max(1,...vals),mn=Math.min(0,...vals);
    const y=v=>T+H*(mx-v)/(mx-mn);const z=y(0);const x=i=>L+cw*i+cw/2;const tk=v=>fmt억(v);let g='';
    [mx,mx/2,0,...(mn<0?[mn]:[])].forEach(v=>{g+=`<line x1="${L}" x2="${W-Rr}" y1="${y(v)}" y2="${y(v)}" class="${v===0?'fczero':'fcgrid'}"/><text x="${L-6}" y="${y(v)+4}" class="fcax" text-anchor="end">${v===0?'0':tk(v)}</text>`;});
    Q.forEach((q,i)=>{const tip=`${q.p} · ${q.b||'기준 미확인'} · 매출 ${tk(q.rev)} · 영업이익 ${tk(q.op)}${q.opm!=null?` · 영업이익률 ${q.opm.toFixed(1)}%`:''}`;
      if(q.rev!=null){const t=Math.min(y(q.rev),z),h=Math.max(1,Math.abs(z-y(q.rev)));g+=`<rect x="${x(i)-bw-1}" y="${t}" width="${bw}" height="${h}" rx="3" class="fcrev"><title>${escape(tip)}</title></rect>`;}
      if(q.op!=null){const t=Math.min(y(q.op),z),h=Math.max(1,Math.abs(z-y(q.op)));g+=`<rect x="${x(i)+1}" y="${t}" width="${bw}" height="${h}" rx="3" class="${q.op>=0?'fcpos':'fcneg'}"><title>${escape(tip)}</title></rect>`;}
      if(q.rev==null&&q.op==null)g+=`<text x="${x(i)}" y="${z-4}" class="fcax" text-anchor="middle">–</text>`;
      const lab=(q.p.endsWith('Q1')||i===0)?q.p.replace('Q',' Q'):q.p.slice(4);g+=`<text x="${x(i)}" y="${T+H+16}" class="fcax" text-anchor="middle">${lab}</text>`;
      if(i>0&&q.b&&Q[i-1].b&&q.b!==Q[i-1].b)g+=`<line x1="${x(i)-cw/2}" x2="${x(i)-cw/2}" y1="${T}" y2="${T+H}" class="fcbasis"/><text x="${x(i)-cw/2+3}" y="${T+10}" class="fcax">${escape(q.b)} 기준 시작</text>`;});
    const Ht=T+H+26;
    const leg=`<div class="fcleg"><span><i class="fcrev"></i>매출</span><span><i class="fcpos"></i>영업이익</span><span><i class="fcneg"></i>영업손실</span><span class="meta">단위 억원 · 막대에 마우스를 올리면 수치</span></div>`;
    return `${leg}<div class="fcwrap"><svg viewBox="0 0 ${W} ${Ht}" width="${W}" style="max-width:100%;height:auto" role="img" aria-label="분기 매출과 영업이익">${g}</svg></div>${Q.some((q,i)=>i>0&&q.b&&Q[i-1].b&&q.b!==Q[i-1].b)?'<p class="meta">세로 점선은 연결·별도 재무제표 기준이 바뀐 분기입니다.</p>':''}`;}
  function openEv(list,title){const seen=new Set();const L=list.filter(e=>{const k=(e.eid||e.cid)+'|'+e.category;if(seen.has(k))return false;seen.add(k);return true;}).sort((a,b)=>(b.date||'').localeCompare(a.date||''));
    let m=byId('evModal');if(!m){m=document.createElement('div');m.id='evModal';m.className='evmodal';document.body.appendChild(m);}
    m.innerHTML=`<div class="evbox" role="dialog" aria-modal="true" aria-label="${escape(title)}"><div class="evhead"><div><div class="eyebrow">기사 사건 원문 확인</div><h3>${escape(title)} · ${L.length}건</h3></div><button class="evclose" data-evclose="1" aria-label="닫기">×</button></div>
      <p class="hint">같은 사안을 다룬 기사는 하나의 사건으로 묶었습니다. 제목을 누르면 기사 원문이 새 창에서 열립니다. ‘분석 근거 제목·요약’으로 표시된 사건은 기사 본문을 열어 보지 않고 제목과 요약으로만 분류한 것입니다.</p>
      <ul class="evlist">${L.map(e=>`<li><div class="evmeta"><span class="meta">${escape(e.date||'')}</span> <span class="pill blue">${escape(e.category||'')}</span>${e.stage?` <span class="pill">${escape(e.stage)}</span>`:''}${e.verify?` <span class="pill">${escape(e.verify)}</span>`:''}</div>
        <div class="evtitle">${e.url?`<a href="${escape(e.url)}" target="_blank" rel="noopener">${escape(e.title||'(제목 없음)')}</a>`:escape(e.title||'(제목 없음)')}</div>${e.sm?`<div class="evsm">${escape(e.sm)}…</div>`:''}
        <div class="meta">${[e.firm,e.ip&&('IP '+e.ip),e.use,e.partner&&('상대 '+e.partner),e.basis&&('분석 근거 '+e.basis),e.read&&('원문 '+e.read),e.reports&&('보도 '+e.reports+'건')].filter(Boolean).map(escape).join(' · ')}</div></li>`).join('')}</ul></div>`;
    m.hidden=false;m.querySelector('.evclose').focus();}
  function evQuery(q){const p=q.split('|');const t=p[0];
    if(t==='y'){const evs=(evCodes.get(p[1])||[]).filter(e=>(e.date||'').startsWith(p[2])&&e.category!=='검토 후보'&&(!p[3]||e.category===p[3]));const nm=(D.companies.find(c=>c.code===p[1])||{}).name||'';return [evs,`${nm} ${p[2]}${p[3]?' · '+p[3]:''}`];}
    if(t==='c'){const nm=(D.companies.find(c=>c.code===p[1])||{}).name||'';return [(evCodes.get(p[1])||[]).filter(e=>e.category===p[2]),`${nm} · ${p[2]}`];}
    if(t==='t'){const r=AGGROWS[p[1]];if(!r)return [[],''];return [p[2]?r.e.filter(e=>e.category===p[2]):r.e,`${r.label}${p[2]?' · '+p[2]:' · 전체 범주'}`];}
    return [[],''];}
  function bizBlock(c){const t=state.bizTab||'실적';const tabs=['실적','사업부문·매출 구성','공시','전략 활동','통합 연혁'];
    return `<div class="subsection bizbox" id="bizBox"><div class="subhead"><div><h3>사업 활동</h3></div></div>
      <div class="toolbarfit">${tabs.map(x=>`<button class="filterbtn ${t===x?'active':''}" data-biztab="${escape(x)}">${escape(x)}</button>`).join('')}</div><div id="bizBody">${bizBody(c,t)}</div></div>`;}
  function bizBody(c,t){
    if(t==='실적'){const f=(D.fin[c.code]||[]).filter(x=>x.p>=RANGE.q0);if(!f.length)return '<div class="empty">분기 실적 자료가 없습니다.</div>';
      const mx=Math.max(...f.map(x=>Math.abs(x.rev||0)),1);
      return finChart(f)+`<p class="hint">분기별 DART 재무자료(억원). 연결·별도 기준이 다른 분기는 직접 비교하거나 합산하지 마세요. 각 행의 기준을 확인하세요. 막대는 매출 규모.</p><div class="tablewrap"><table class="aggtable fintable"><thead><tr><th>분기</th><th>기준</th><th>매출</th><th></th><th>전년비</th><th>영업이익</th><th>전년비</th><th>영업이익률</th><th>순이익</th><th>흐름</th></tr></thead><tbody>${f.slice().reverse().map(x=>`<tr><td><a href="${escape(safeUrl(x.url))}" target="_blank" rel="noopener">${escape(x.p)}</a></td><td>${escape(x.b||'미확인')}</td><td class="num">${fmt억(x.rev)}</td><td><span class="bar" style="width:${Math.round(100*Math.abs(x.rev||0)/mx)}%"></span></td><td class="num ${x.ry<0?'neg':''}">${pct(x.ry)}</td><td class="num ${x.op<0?'neg':''}">${fmt억(x.op)}</td><td class="num">${pct(x.oy)}</td><td class="num">${x.opm==null?'':x.opm.toFixed(1)+'%'}</td><td class="num ${x.ni<0?'neg':''}">${fmt억(x.ni)}</td><td class="small">${escape(x.flow)}</td></tr>`).join('')}</tbody></table></div>`;}
    if(t==='사업부문·매출 구성'){const s=D.sales[c.code]||[];if(!s.length)return '<div class="empty">사업보고서 매출 품목 자료가 없습니다.</div>';
      const per=[...new Set(s.map(x=>x.y+' '+x.per))].sort().reverse();
      return `<p class="hint">사업보고서 ‘매출 및 수주상황’의 사업부문·품목별 매출(억원)과 매출 대비 비중. 합계 행(비중 100%)과 세부 품목이 함께 있을 수 있습니다.</p>`+per.map(p=>{const rows=s.filter(x=>x.y+' '+x.per===p).sort((a,b)=>(b.amt||0)-(a.amt||0));
        return `<div class="sectionlabel">${escape(p)}</div><div class="tablewrap"><table class="aggtable"><thead><tr><th>사업부문·품목</th><th>매출</th><th>비중</th><th>전년비</th></tr></thead><tbody>${rows.slice(0,15).map(x=>`<tr><td>${escape(x.lab)}</td><td class="num">${fmt억(x.amt)}</td><td class="num">${x.sh==null?'':x.sh.toFixed(1)+'%'}</td><td class="num ${x.yoy<0?'neg':''}">${pct(x.yoy)}</td></tr>`).join('')}</tbody></table></div>`;}).join('');}
    if(t==='공시'){const s=(D.disc[c.code]||[]).slice().reverse();if(!s.length)return `<div class="empty">${RANGE.y0}년 이후 주요 공시 자료가 없습니다.</div>`;
      const ks=[...new Set(s.map(x=>x.k))];
      return `<p class="hint">${RANGE.y0}년 이후 DART 주요 공시(공급계약·인수합병·자금조달·지배구조·주주환원·위험·소송·종속회사 변동 등) ${s.length}건 · 최신순</p><div class="btnrow" style="margin-bottom:10px">${ks.map(k=>`<span class="pill">${escape(k)} ${s.filter(x=>x.k===k).length}</span>`).join('')}</div><ul class="tl">${s.map(x=>`<li><span class="meta">${escape(x.d)}</span> <span class="pill">${escape(x.k)}</span> ${x.u?`<a href="${escape(x.u)}" target="_blank" rel="noopener">${escape(x.t)}</a>`:escape(x.t)}${(x.x||[]).length?`<div class="meta discline">${escape(discLine(x))}</div>`:''}</li>`).join('')}</ul>`;}
    if(t==='전략 활동'){const s=(D.strat[c.code]||[]).slice().reverse();if(!s.length)return `<div class="empty">${RANGE.y0}년 이후 전략 활동(V16) 분석 자료가 없습니다.</div>`;
      const ks=[...new Set(s.map(x=>x.k))];
      return `<p class="hint">기사 기반 전략 활동 분석(V16, ${RANGE.y0}년 이후 실행 확인분) ${s.length}건. ‘통상 운영’ 표시는 산업 변화 근거로 쓰지 않는 일상 운영입니다.</p><div class="btnrow" style="margin-bottom:10px">${ks.map(k=>`<span class="pill blue">${escape(k)} ${s.filter(x=>x.k===k).length}</span>`).join('')}</div><ul class="tl">${s.map(x=>`<li><span class="meta">${escape(x.d)}</span> <span class="pill blue">${escape(x.k)}</span> ${x.u?`<a href="${escape(x.u)}" target="_blank" rel="noopener">${escape(x.t)}</a>`:escape(x.t)}${x.rt?' <span class="meta">· 통상 운영</span>':''}</li>`).join('')}</ul>`;}
    const ev=firmEvents().filter(e=>e.category!=='검토 후보');const seen=new Set();const items=[];
    for(const e of ev){const k=e.eid||e.cid;if(seen.has(k))continue;seen.add(k);items.push({d:e.date||'',type:'기사',k:e.category,t:e.title,u:e.url});}
    for(const x of D.disc[c.code]||[])items.push({d:x.d,type:'공시',k:x.k,t:x.t,u:x.u});
    for(const x of D.strat[c.code]||[])if(!x.rt)items.push({d:x.d,type:'전략 활동',k:x.k,t:x.t,u:x.u});
    for(const x of D.fin[c.code]||[])if(x.p>=RANGE.q0&&x.flow)items.push({d:x.p.replace(/Q(\d)/,(m,q)=>'-'+String(q*3).padStart(2,'0')+'-말'),type:'실적',k:x.p,t:`매출 ${fmt억(x.rev)}(${pct(x.ry)}) · 영업이익 ${fmt억(x.op)} · ${x.flow}`});
    items.sort((a,b)=>b.d.localeCompare(a.d));
    return `<p class="hint">기사 사례(사건 ID 기준)·공시·전략 활동·분기 실적을 한 줄 연혁으로 합쳤습니다(최신순, ${items.length}건).</p><ul class="tl">${items.slice(0,state.tlLimit||60).map(x=>`<li><span class="meta">${escape(x.d)}</span> <span class="pill ${x.type==='공시'?'':x.type==='전략 활동'?'blue':x.type==='실적'?'own':'warn'}">${escape(x.type)}</span> <span class="pill">${escape(x.k)}</span> ${x.u?`<a href="${escape(x.u)}" target="_blank" rel="noopener">${escape(x.t)}</a>`:escape(x.t)}</li>`).join('')}</ul>${items.length>(state.tlLimit||60)?`<button class="more" data-tlmore="1">더 보기 (전체 ${items.length}건)</button>`:''}`;}
  function detail(label,value){return `<div class="detail"><dt>${escape(label)}</dt><dd>${val(value)}</dd></div>`;}
  function renderProfile(){const c=company();if(!c){byId('profile').innerHTML='<div class="empty">표시할 기업이 없습니다.</div>';return;}
    const ips=firmIps(),events=firmEvents();
    const rename=D.history.find(x=>x.code6===c.code);
    const ownership=ips.filter(x=>['자사 원천','자회사 원천','인수 보유','공동 보유'].includes(x.rights_group)).length;
    const counts=cat=>uniqCount(events.filter(x=>x.category===cat));
    byId('profile').innerHTML=`<button class="backbtn" data-backsector="1">← ${escape(state.sector)} 분야 통합 데이터로</button><div class="profiletop"><div class="eyebrow">기업 상세 소개</div><h2>${escape(c.name)}</h2><p>기업코드 ${escape(c.code)} · ${escape(state.sector)} 분야 등재 기업 · 사건·IP DB 기준 활동 요약</p><span class="badge">${escape(c.registered||'연도별 등재 분야 확인 필요')}</span>${newsTag(c)?`<span class="badge pending">${newsTag(c)[0]} · 기사 기반 활동 미반영</span>`:''}${rename?`<span class="badge">구 ${escape(rename['이전 명칭'])} · 변경 승인 ${escape(rename['변경 승인일'])}</span>`:''}<div class="profilestats"><div class="profilemetrics"><strong>${ips.length}</strong><small>IP 분류 항목</small></div><div class="profilemetrics"><strong>${ownership}</strong><small>원천·공동 관련 IP</small></div><div class="profilemetrics"><strong><button class="evlink big" data-evq="c|${escape(c.code)}|IP 활용">${counts('IP 활용')}</button></strong><small>IP 활용 사건 ID</small></div><div class="profilemetrics"><strong><button class="evlink big" data-evq="c|${escape(c.code)}|해외 실행·성과">${counts('해외 실행·성과')}</button></strong><small>해외 실행·성과 사건 ID</small></div></div></div>
    ${introBlock(c)}${flowBlock(c)}${timelineBlock(c)}${assetBlock(c)}${bizBlock(c)}<div class="subsection"><div class="subhead"><h3>기사·공시 분석으로 본 사업 활동</h3><span class="subnote">사건·IP 자료를 요약한 것으로 공식 사업 소개나 재무 실적이 아닙니다.</span></div><dl class="profiledesc">${detail('자사 원천 IP(기업 개요 원문)',c.summary_owned)}${detail('외부 IP와 기업 역할',c.summary_external)}${detail('IP 활용 유형',c.summary_uses)}${detail('해외 실행·성과 국가',c.summary_countries)}${detail('국제 계약·제휴 상대국',c.summary_contracts)}${detail('주요 협업 상대',c.summary_partners)}${detail('거래 관계',c.summary_transactions)}${detail('분야 등재 이력',c.registered)}</dl></div>
    <div class="subsection" id="ipSection"><div class="subhead"><div><div class="eyebrow">03 · IP 소개</div><h3>원천·공동·외부 IP 및 권리 관계</h3></div><span class="subnote">현재 권리 분석 우선 · 기업 역할 별도</span></div><p class="hint" style="margin-bottom:15px">'보유 IP'와 '사업에 활용한 외부 IP'는 다릅니다. 아래 목록에는 권리 미확인 사례도 포함됩니다.</p><div class="toolbarfit" id="ipFilters"></div><div id="ipGrid" class="ipgrid"></div><div id="ipMore"></div><div id="ipFocus"></div></div>
    <div class="subsection activities" id="activitySection"><div class="subhead"><div><div class="eyebrow">04 · 활동 사례</div><h3>실제 활동·해외·협업 사례</h3></div><span class="subnote">같은 사건이 여러 분석 측면에 등장할 수 있습니다.</span></div><div class="toolbarfit" id="eventFilters"></div><div id="eventList"></div><div id="eventMore"></div></div>`;
    renderIpFilters();renderIps();renderIpFocus();renderEventFilters();renderEvents();}
  function renderIpFilters(){byId('ipFilters').innerHTML=groups.map(g=>`<button class="filterbtn ${state.ipGroup===g?'active':''}" data-ipgroup="${escape(g)}">${escape(g)}</button>`).join('')+`<input class="searchinput" id="ipSearch" aria-label="IP 검색" placeholder="IP명 검색" value="${escape(state.ipSearch)}">`;}
  function getFilteredIps(){const q=state.ipSearch.trim().toLowerCase();return firmIps().filter(ip=>(state.ipGroup==='전체'||ip.rights_group===state.ipGroup)&&(!q||[ip.name,ip.aliases,ip.parent].join(' ').toLowerCase().includes(q)));}
  function renderIps(){const all=getFilteredIps(),current=all.slice(0,state.ipLimit);const evs=firmEvents();
    byId('ipGrid').innerHTML=current.length?current.map(ip=>{
      const n=uniqCount(evs.filter(e=>e.rid===ip.rid&&e.category==='IP 활용'));
      return `<button class="ipcard ${state.rid===ip.rid?'selected':''}" data-ip="${escape(ip.rid)}">${niceRights(ip.rights_group)}<strong>${escape(ip.name)}</strong><span class="small">${escape(ip.role||'기업 역할 미확인')}</span>${ip.rights_level?`<span class="small" title="권리 확인 수준"> · ${escape(ip.rights_level)}</span>`:''}<span class="ipbottom"><span class="small">활용 사건 ${n}개</span><span class="small">${escape(ip.source_level||'근거 수준 미기재')}</span></span></button>`;
    }).join(''):'<div class="empty">해당 조건의 IP가 없습니다.</div>';
    byId('ipMore').innerHTML=state.ipLimit<all.length?`<button class="more" data-more-ip>IP ${Math.min(24,all.length-state.ipLimit)}개 더 보기 (${all.length}개 중 ${current.length}개 표시)</button>`:'';}
  function evidence(e){const detail=[e.use,e.output,e.global_action,e.country,e.market,e.partner,e.joint_action].filter(Boolean);
    const url=safeUrl(e.url),urlLabel=e.read==='원문 열람'?'열람 표시된 근거':'원문 URL(본문 미열람)';
    const read=e.read==='원문 열람'?'원문 열람':'제목·요약 기준';
    return `<article class="evidence"><div class="evline"><span class="pill blue">${escape(e.category)}</span>${(e.category==='IP 활용'||e.use)&&e.use?usesOf(e).map(u=>`<span class="pill usepill">${escape(useShort(u))}</span>`).join(''):''}${e.ov&&!/해외|국제/.test(e.category||'')?'<span class="pill ovpill" title="분류는 다르지만 국가·제목에 해외 단서가 있는 사건">해외</span>':''}<span class="pill">${escape(e.date||'날짜 미기재')}</span>${e.stage?`<span class="pill">단계: ${escape(e.stage)}</span>`:''}${e.exec?`<span class="pill blue">실행: ${escape(e.exec)}</span>`:''}${e.verify?`<span class="pill ${e.verify==='추정'?'warn':''}">${escape(e.verify)}</span>`:''}</div><h5>${escape(e.title||e.output||'제목 미기재')}</h5>${detail.length?`<p>${escape(detail.join(' · '))}</p>`:''}<div class="bottom"><span>사례 ${escape(e.cid||'—')} / 사건 ${escape(e.eid||'—')}</span><span>${escape(read)}</span>${e.role?`<span>${escape(e.role)}</span>`:''}${url?`<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${urlLabel} ↗</a>`:''}</div></article>`;}
  function renderIpFocus(){const box=byId('ipFocus'),ip=selectedIp();if(!ip){box.innerHTML='<div class="empty">기업의 IP를 선택해 관련 근거를 확인하세요.</div>';return;}
    const linked=firmEvents().filter(e=>e.rid===ip.rid),uses=linked.filter(e=>e.category==='IP 활용'),other=linked.filter(e=>e.category!=='IP 활용');
    const related=firmIps().filter(x=>x.rid!==ip.rid&&((x.parent&&x.parent===ip.name)||(ip.parent&&ip.parent===x.name)||(ip.parent&&ip.parent!==ip.name&&x.parent===ip.parent)));
    const description=ip.note||'원본의 IP 권리·역할 분류와 연결된 사건을 표시합니다.';
    box.innerHTML=`<div class="ipfocus"><div class="ipfocushead"><div><div class="eyebrow">IP 상세 · ${escape(ip.rid)}</div><h4>${escape(ip.name)}</h4>${niceRights(ip.rights_group)} <span class="pill">${escape(ip.role||'사업 역할 미확인')}</span>${ip.rights_level?`<span class="pill" title="권리 확인 수준">${escape(ip.rights_level)}</span>`:''}</div><button class="tinybtn" id="selectedIpCsv">선택 IP 사례 CSV ↓</button></div>
      <p class="hint" style="margin-top:11px">${escape(description)}</p><dl><div><dt>현재 권리 분석</dt><dd>${escape(ip.rights)}</dd></div><div><dt>원장 기준 권리</dt><dd>${escape(ip.original_rights||'미기재')}</dd></div><div><dt>권리 시점·변동</dt><dd>${escape(ip.rights_period||'별도 기록 없음')}</dd></div><div><dt>근거 수준</dt><dd>${escape(ip.source_level||'별도 기록 없음')}</dd></div><div><dt>하위·관련 명칭</dt><dd>${escape(ip.aliases||'별도 기록 없음')}</dd></div><div><dt>IP 단위 분석 역할</dt><dd>${escape(ip.role||'별도 기록 없음')}</dd></div></dl>
      ${safeUrl(ip.evidence_url)?`<a target="_blank" rel="noopener noreferrer" href="${escape(ip.evidence_url)}">권리 교정 근거 확인 ↗</a>`:''}
      ${(()=>{const L=linked.filter(e=>e.category==='IP 활용'||e.ip);if(!L.length)return '';const c=USES.map(([u,l])=>[l,uniqCount(L.filter(e=>useOk(e,u)))]).filter(x=>x[1]),ov=uniqCount(L.filter(e=>e.ov));
        return `<div class="sectionlabel">IP 활용 세부 활동 · ${uniqCount(L)}개 사건 ID</div><div class="btnrow">${c.map(([l,n])=>`<span class="pill usepill">${escape(l)} ${n}</span>`).join('')}${ov?`<span class="pill ovpill">해외 관련 ${ov}</span>`:''}</div>`;})()}
      ${related.length?`<div class="sectionlabel">계보상 관련 IP (각각 별도 권리 분석)</div><div class="btnrow">${related.map(r=>`<button class="tinybtn" data-ip="${escape(r.rid)}">${escape(r.name)} · ${escape(r.rights_group)}</button>`).join('')}</div>`:''}<div class="sectionlabel">실제 IP 활용 사례 · ${uniqCount(uses)}개 사건 ID</div>${uses.length?uses.slice(0,16).map(evidence).join(''):'<div class="empty">이 IP와 정확한 이름으로 연결된 활용 사건이 없습니다. 활용되지 않았음을 뜻하지 않습니다.</div>'}${uses.length>16?`<p class="hint">16건까지만 미리 보여줍니다. 해당 IP 사례 CSV를 내려받으면 연결된 모든 기록을 확인할 수 있습니다.</p>`:''}
      <div class="sectionlabel">연결된 해외·협업 및 기타 기록 · ${uniqCount(other)}개 사건 ID</div>${other.length?other.slice(0,10).map(evidence).join(''):'<p class="hint">정확한 대표 IP명으로 연결된 추가 자료가 없습니다.</p>'}${other.length>10?`<p class="hint">나머지 자료는 선택 IP 사례 CSV에 포함됩니다.</p>`:''}
    </div>`;}
  function renderEventFilters(){byId('eventFilters').innerHTML=categories.map(g=>`<button class="filterbtn ${state.evCategory===g?'active':''}" data-evcat="${escape(g)}">${escape(g)}</button>`).join('')+(state.evCategory==='IP 활용'?useBar(firmEvents().filter(e=>e.category==='IP 활용'),state.evUse,'evuse'):'')+`<input class="searchinput" id="eventSearch" aria-label="활동 검색" placeholder="사건명·국가·협업사 검색" value="${escape(state.evSearch)}">`;}
  function getFilteredEvents(){const W=qWords(state.evSearch);return firmEvents().filter(e=>catOk(e,state.evCategory)&&(state.evCategory!=='IP 활용'||useOk(e,state.evUse))&&(!W.length||(f=>W.every(A=>A.some(w=>hasW(f,w))))([e.title,e.ip,e.country,e.partner,e.output,e.use,e.eid,e.sm].join(' ').toLowerCase()))).slice().sort((a,b)=>(b.date||'').localeCompare(a.date||''));}
  function renderEvents(){const all=getFilteredEvents();byId('eventList').innerHTML=`<p class="hint" style="margin:8px 0 14px">${all.length.toLocaleString('ko-KR')}개 자료행 · ${uniqCount(all).toLocaleString('ko-KR')}개 사건 ID (선택 조건 내)</p>` + (all.length?all.slice(0,state.evLimit).map(evidence).join(''):(firmEvents().length?'<div class="empty">해당 활동이 없습니다.</div>':`<div class="empty">${(newsTag(company())||NSL['수집 필요'])[1]}.</div>`));byId('eventMore').innerHTML=state.evLimit<all.length?`<button class="more" data-more-event>20건 더 보기 (전체 ${all.length}개 자료행)</button>`:'';}
  const eventCols=[['category','자료구분'],['code','기업코드'],['firm','기업'],['date','날짜'],['cid','사례ID'],['eid','사건ID'],['bid','사업이력ID'],['ip','대표IP'],['ip_names','IP명'],['ip_rel','IP관계'],['use','활용유형'],['output','산출물'],['stage','단계'],['exec','실행단계'],['global_action','해외행위'],['country','진출국가'],['market','진출시장'],['event_country','행사개최국'],['partner_country','협력사소재국'],['partner','상대기업'],['partner_sector','상대산업'],['role','역할'],['verify','검증'],['mark','표시'],['title','제목'],['url','대표원문URL'],['read','원문열람'],['basis','판정근거'],['correction_url','교정근거URL']];
  const ipCols=[['code','code6'],['firm','기업'],['rid','IP_ID'],['name','대표IP'],['aliases','하위관련명칭'],['parent','상위IP'],['rights_group','권리구분'],['rights','현재권리'],['original_rights','원권리'],['rights_period','권리시점'],['role','사업역할'],['source_level','근거수준'],['note','판정비고'],['evidence_url','교정근거URL']];
  const companyCols=[['sector','분야'],['code','code6'],['name','기업표시명'],['raw_name','명부기업명'],['registered','연도별등록분야'],['summary_owned','자사원천IP_개요'],['summary_external','외부IP역할_개요'],['summary_uses','주요IP활용방식'],['summary_countries','해외실행성과국가'],['summary_contracts','국제계약제휴국가'],['summary_partners','주요협업상대'],['summary_transactions','거래관계']];
  function csvDownload(filename,rows,cols){const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"';const csv='\ufeff'+[cols.map(x=>quote(x[1])).join(','),...rows.map(row=>cols.map(x=>quote(row[x[0]])).join(','))].join('\r\n');const blob=new Blob([csv],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1500);}
  const codes=()=>getSectorCodes();
  function actionDownload(id){const sec=state.sector.replaceAll(' ','_'),c=company(),ip=selectedIp();
    if(!c&&['companyIpCsv','companyEventsCsv','selectedIpCsv'].includes(id)){alert('기업을 먼저 선택하세요. 전체 개요에서는 분야·전체 단위 CSV만 내려받을 수 있습니다.');return;}
    if(id==='downloadSectorCompany')csvDownload(`${sec}_기업개요_${VER}.csv`,getSectorCompanies(),companyCols);
    if(id==='sectorIpCsv')csvDownload(`${sec}_IP_${VER}.csv`,D.ips.filter(x=>codes().has(x.code)),ipCols);
    if(id==='sectorEventsCsv')csvDownload(`${sec}_활동_${VER}.csv`,D.events.filter(x=>codes().has(x.code)),eventCols);
    if(id==='companyIpCsv')csvDownload(`${sec}_${c.name}_IP_${VER}.csv`,firmIps(),ipCols);
    if(id==='companyEventsCsv')csvDownload(`${sec}_${c.name}_활동_${VER}.csv`,firmEvents(),eventCols);
    if(id==='selectedIpCsv'&&ip)csvDownload(`${sec}_${c.name}_${ip.name}_활동_${VER}.csv`,firmEvents().filter(x=>x.rid===ip.rid),eventCols);
  }
  function renderDocs(){const names=Array.isArray(D.docs)&&D.docs.length?D.docs:['00_색인.md','01_v3.4_재검토반영보고.md','01_게임.md','02_음악.md','03_방송및영상.md','04_애니메이션.md','05_캐릭터.md','06_만화.md','07_지식정보.md','08_출판.md','09_영화.md','10_광고.md','11_콘텐츠솔루션.md'];byId('sourceDocs').innerHTML=names.map(n=>`<a href="./docs/${encodeURIComponent(n)}" download>${escape(n)}</a>`).join('');}
  document.addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;
    if(b.hasAttribute('data-sector')){state.sector=b.dataset.sector;state.code='';Object.assign(state,{aggCat:'전체',aggSearch:'',aggLimit:12,aggIpGroup:'전체',aggIpSearch:'',aggIpLimit:24});state.companySearch='';resetIp();updateHash();renderSector();window.scrollTo({top:0,behavior:'smooth'});return;}
    if(b.hasAttribute('data-company')){goCompany(b.dataset.company,'');return;}
    if(b.hasAttribute('data-gocompany')){const [c,r]=b.dataset.gocompany.split('|');goCompany(c,r||'');return;}
    if(b.hasAttribute('data-aggsrc')){state.aggSrc=b.dataset.aggsrc;state.aggLimit=12;renderAggEvents();return;}
    if(b.hasAttribute('data-csort')){state.companySort=b.dataset.csort;renderCompanies();return;}
    if(b.hasAttribute('data-evq')){const [L,t]=evQuery(b.dataset.evq);openEv(L,t);return;}
    if(b.hasAttribute('data-evclose')){byId('evModal').hidden=true;return;}
    if(b.hasAttribute('data-tl')){const k=b.dataset.tl;const L=(TLDATA[k]||[]).slice().sort((a,b)=>a.d.localeCompare(b.d));const [r,mo]=[k.slice(0,k.lastIndexOf('|')),k.slice(k.lastIndexOf('|')+1)];byId('tlDetail').innerHTML=`<h4>${escape(mo)} · ${escape(r.replace('IP:','IP '))} · ${L.length}건</h4><ul class="tl">${L.map(i=>`<li><span class="meta">${escape(i.d)}</span> <span class="pill">${escape(i.g)}</span> ${i.u?`<a href="${escape(i.u)}" target="_blank" rel="noopener">${escape(i.t)}</a>`:escape(i.t)} <span class="meta">${escape(i.x||'')}</span></li>`).join('')}</ul>`;return;}
    if(b.hasAttribute('data-assetrel')){state.assetRel=b.dataset.assetrel;state.assetLimit=40;byId('assetBox').outerHTML=assetBlock(company());return;}
    if(b.hasAttribute('data-assetkind')){state.assetKind=b.dataset.assetkind;state.assetLimit=40;byId('assetBox').outerHTML=assetBlock(company());return;}
    if(b.hasAttribute('data-assetmore')){state.assetLimit=(state.assetLimit||40)+40;byId('assetBox').outerHTML=assetBlock(company());return;}
    if(b.hasAttribute('data-biztab')){state.bizTab=b.dataset.biztab;state.tlLimit=60;const c=company();byId('bizBox').outerHTML=bizBlock(c);return;}
    if(b.hasAttribute('data-tlmore')){state.tlLimit=(state.tlLimit||60)+80;byId('bizBody').innerHTML=bizBody(company(),state.bizTab||'실적');return;}
    if(b.hasAttribute('data-backsector')){state.code='';updateHash();renderSector();window.scrollTo({top:0,behavior:'smooth'});return;}
    if(b.hasAttribute('data-agguse')){state.aggUse=b.dataset.agguse;state.aggLimit=12;renderAggEvents();return;}
    if(b.hasAttribute('data-evuse')){state.evUse=b.dataset.evuse;state.evLimit=10;renderEventFilters();renderEvents();return;}
    if(b.hasAttribute('data-aggcat')){state.aggCat=b.dataset.aggcat;state.aggUse='전체';state.aggLimit=12;renderAggEvents();return;}
    if(b.hasAttribute('data-aggipgroup')){state.aggIpGroup=b.dataset.aggipgroup;state.aggIpLimit=24;renderAggIps();return;}
    if(b.hasAttribute('data-aggmore')){state.aggLimit+=20;renderAggEvents();return;}
    if(b.hasAttribute('data-aggipmore')){state.aggIpLimit+=36;renderAggIps();return;}
    if(b.id==='aggEventsCsv'){csvDownload(`${state.sector.replaceAll(' ','_')}_통합활동_${VER}.csv`,aggEventsFiltered(),eventCols);return;}
    if(b.id==='aggIpCsv'){csvDownload(`${state.sector.replaceAll(' ','_')}_통합IP_${VER}.csv`,aggIpsFiltered(),ipCols);return;}
    if(b.hasAttribute('data-ipgroup')){state.ipGroup=b.dataset.ipgroup;state.ipLimit=18;renderIpFilters();renderIps();return;}
    if(b.hasAttribute('data-ip')){state.rid=b.dataset.ip;renderIps();renderIpFocus();byId('ipFocus').scrollIntoView({behavior:'smooth',block:'start'});return;}
    if(b.hasAttribute('data-more-ip')){state.ipLimit+=24;renderIps();return;}
    if(b.hasAttribute('data-evcat')){state.evCategory=b.dataset.evcat;state.evUse='전체';state.evLimit=10;renderEventFilters();renderEvents();return;}
    if(b.hasAttribute('data-more-event')){state.evLimit+=20;renderEvents();return;}
    if(['downloadSectorCompany','sectorIpCsv','sectorEventsCsv','companyIpCsv','companyEventsCsv','selectedIpCsv'].includes(b.id)){actionDownload(b.id);return;}
  });
  document.addEventListener('input',e=>{if(e.target.id==='assetSearch'){state.assetSearch=e.target.value;state.assetLimit=40;byId('assetBox').outerHTML=assetBlock(company());const i=byId('assetSearch');i.focus();i.setSelectionRange(i.value.length,i.value.length);return;}if(e.target.id==='aggSearch'){state.aggSearch=e.target.value;state.aggLimit=12;renderAggEvents();return;}if(e.target.id==='aggIpSearch'){state.aggIpSearch=e.target.value;state.aggIpLimit=24;renderAggIps();return;}if(e.target.id==='companySearch'){state.companySearch=e.target.value;renderCompanies();}
    if(e.target.id==='ipSearch'){state.ipSearch=e.target.value;state.ipLimit=18;renderIps();}
    if(e.target.id==='eventSearch'){state.evSearch=e.target.value;state.evLimit=10;renderEvents();}});
  window.addEventListener('hashchange',()=>{const h=new URLSearchParams(location.hash.slice(1)),s=h.get('sector'),c=h.get('code')||'';const ns=sectorByName[s]?s:ALL;if(ns===state.sector&&c===state.code)return;state.sector=ns;state.code=ns===ALL?'':(D.companies.some(x=>x.sector===ns&&x.code===c)?c:'');resetIp();renderSector();});
  /* 머리글·바닥글의 기간·버전·분야 수도 데이터에서 */
  function renderChrome(){document.title=`콘텐츠 상장사 IP·사업 현황 | ${D.version||''}`;
    const set=(id,t)=>{const el=byId(id);if(el)el.textContent=t;};
    set('dataPeriod',RANGE.y0===RANGE.y1?String(RANGE.y0):`${RANGE.y0}–${RANGE.y1}`);set('dataVersion','DATA '+(D.version||''));
    set('sectorCountLabel',`전체 + ${SECTORS.length}개 분야`);
    set('footerNote',`출처: 사용자 제공 「분야별정리_IP·글로벌·협업_2024-2026_v3.4」와 이후 갱신 자료 · 자료 기간 ${RANGE.start}~${RANGE.end} · 갱신 ${UPDATED} · 원자료 및 교정 로그는 다운로드 파일 참조`);}
  renderChrome();setupSelection();renderDocs();renderSector();
})();

document.addEventListener('keydown',e=>{if(e.key==='Escape'){const m=document.getElementById('evModal');if(m)m.hidden=true;}});
document.addEventListener('click',e=>{const m=document.getElementById('evModal');if(m&&e.target===m)m.hidden=true;});
