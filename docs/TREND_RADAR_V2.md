# 콘텐츠산업 뉴스 트렌드 분석 시스템 v2 — 구축 설계

근거: 「콘텐츠산업 뉴스 기반 트렌드 분석 시스템 상세 기획안」(2026-09-28, v1.0). 이 문서는 그 기획안을
**현재 원자료와 저장소에 맞춰 구체화**한 것이다. v1 구현 기록은 [TREND_RADAR.md](TREND_RADAR.md), 최초 설계는
[TREND_RADAR_DESIGN.md](TREND_RADAR_DESIGN.md).

---

## 0. 확정 사항 (2026-09-28)

| 항목 | 결정 |
|---|---|
| 분야 범위 | 핵심 6개(게임, 음악, 방송 및 영상, 만화, 애니메이션, 캐릭터) + **연관산업**(대상 밖 분야 전용 기업의 기사, 한 단계 낮은 층위) |
| 시계열 단위 | **월**. 비교 이력 12기간 |
| 정답셋 | 학습 라벨은 **약한 지도학습(규칙 라벨 함수 + 라벨 모델)**. 평가는 **기사 단위 임의 판정(사람·LLM) 없이** 외부 라벨(네이버 섹션 코드·언론사 URL 경로), 독립 방법 간 교차 확인, 통계적 임계값, 부트스트랩 안정성, 외부 사실(출시·방영·수상일) 백테스트로 한다 (검토 중, 사람은 규칙 정의만 확인) |
| LLM 역할 | 기획안 10장대로 **비교·설명 계층**. 지금까지의 LLM 라벨·이슈 추출은 **도전자 모델**로만 등록 (정답 아님) |
| 착수 | E0 기반 → 정답셋 지침·표본 → E2 관련성 → E3 핵심어 → E5 시계열 |

## 1. 현재 원자료의 제약

| 기획안 필드 | 원자료 | 처리 |
|---|---|---|
| body | **없음** (요약 약 150자만) | 제목+요약 기반 분석으로 표시 (`text_scope: title_summary`) |
| collected_at | 없음 | 스냅샷 생성 시각으로 대체 |
| content_hash / 중복 | 원자료에 `중복기사군ID`·`중복판정`으로 이미 처리됨 | 그대로 사용, 중복률은 품질표에 보고 |
| sector_source | `분야목록` (기업 단위) | 기사 분야 = 기업 분야의 합집합 |
| rights_status | 없음 | LLM 전송은 제목·요약만, 원문 링크만 보존 |

## 2. 계층과 폴더

```text
configs/            trend_radar.yaml (분석), lexicon.yaml (승인 사전)
registry/           dataset_registry.jsonl · run_registry.jsonl · experiment_registry.jsonl
                    decision_log.jsonl · prompt_registry.jsonl      ← 커밋 (결정 이력)
goldset/            label_guide.md · goldset_v0.csv                   ← 커밋
out/logs/{run_id}/  run_manifest.json · events.jsonl                  ← 커밋 안 함 (manifest 요약은 run_registry로)
out/runs/{run_id}/  01~09 산출물
```

식별자: `data_snapshot_id = ds_{최종 기사일}_{입력 파일 해시 8자}`, `run_id = run_{YYYYMMDD_HHMMSS}_{4자}`,
`config_hash = sha256(정규화 설정)`, `experiment_id`, `model_id`, `artifact_id = art_{산출물}_{체크섬 8자}`.

## 3. 단계별 설계

### E0 데이터 기반
- 스냅샷: 입력 파일 목록·크기·sha256 → `dataset_registry`.
- 품질 점검(기획안 4.2): 요약 누락률, 중복률(행 대비 기사, 중복기사군), 분야 미분류율, 날짜 이상,
  매체 편중(분야별 상위 매체 점유 > 35%), 대형 사건·기업 편중(분야별 상위 기업 점유 > 20%), 월별 기사 수.
- 산출물: `01_data_quality.xlsx`, `run_manifest.json`.

### 정답셋 v0
- `goldset/label_guide.md`: INCLUDE / REVIEW / EXCLUDE 정의, 사유 코드, 경계 사례(실적 기사, 교육, 연예 가십, 교차 업종).
- 관련성 평가 300건: 약한 지도 확률 구간 × 분야(6+연관) × 연도 층화. 판정자 1인 + 100건 교차(또는 50건 1주 후 재판정).
- 알려진 이슈 목록(백테스트용) 약 20건: 이슈, 시작 월, 관련 분야.

### E2 기사 관련성
| 모델 | 내용 |
|---|---|
| m_rel_seed (기준선) | 시드 사전 점수 (v1 층 분류) |
| m_rel_ws_lr | **약한 지도학습** 라벨 함수 → 라벨 모델 → 로지스틱 회귀 |
| m_rel_llm (도전자) | LLM 판정 (기존 라벨) |

판정: ≥ 0.80 INCLUDE, 0.45~0.80 REVIEW, < 0.45 EXCLUDE (정답셋 PR 곡선으로 확정). 강제 포함·제외 규칙, `exclusion_reason` 기록.
지표: Macro F1, 분야별 F1, EXCLUDE 거짓 음성률, REVIEW 물량.

### E3 핵심어 사전
후보: 1~3그램 명사구(Kiwi), 작품명(제목 따옴표), 기업·인물 개체 구분.
지표(개별 저장): DomainSpecificity(가중 로그오즈, 비교 말뭉치 = EXCLUDE 기사), DocumentFrequency, Termhood(C-value),
NetworkAssociation(시드 NPMI), Persistence(월 등장 비율), SectorEntropy.
등급: CORE / EXTENDED / EMERGING / REVIEW / DROP. 모델: 기준선(TF-IDF+로그오즈) vs C-value 복합어 vs LLM 이슈 추출(도전자).

### E4 주제·네트워크 (1차는 기준선만)
키워드 네트워크 NPMI + Leiden, 문서 군집 TF-IDF K-means. 월별 계보 continued / split / merged / new / ended.

### E5 시계열 판정 (월)
지표: Frequency, Growth, Z-score, Burst(Kleinberg), Persistence, Novelty, SectorSpread, Centrality, EntityConcentration.
유형: Emerging / Growing / Established / Event Spike / Declining / Cross-sector (기획안 8.2 규칙, 지표별 조건).
평가: 알려진 이슈 백테스트(탐지 선행기간, 이벤트 구분 F1, 오경보).

### LLM 비교 계층
실험마다 `comparison_pack`(지표·겹침·불일치 사례·근거 참조) → 고정 JSON 스키마 응답 → 수치·근거 자동 대조 → `08_llm_comparison.jsonl`.

### 화면
v1 레이더에 **검토함**(REVIEW 기사·신규어), **모델 비교**, **실행 이력** 탭 추가. 월 단위 시계열.

## 4. 1차 검수 기준 (기획안 15.1 중 1차 범위)
- 모든 실행에 run_manifest, 산출물 체크섬, 설정 해시가 남는다 (100%).
- 관련성: 정답셋 300건 기준 분야별 F1과 EXCLUDE 거짓 음성률을 보고한다.
- 핵심어: 등급별 후보 파일과 대표 기사 근거가 있다.
- 트렌드: 알려진 이슈 백테스트 결과표가 있다.
- 결정(불용어·병합·임계값·모델 교체)은 모두 `decision_log`에 남는다.
