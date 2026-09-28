# wooza

이 저장소에는 두 프로젝트가 있다.

- **UZA** — 먼저 말을 걸기도 하는 대화 상대 (`uza/`, 아래)
- **콘텐츠산업 Trend Radar** — 뉴스 원자료로 콘텐츠산업 트렌드를 탐지하는 분석 (`trend_radar/`)

## 콘텐츠산업 Trend Radar

설계 [`docs/TREND_RADAR_DESIGN.md`](docs/TREND_RADAR_DESIGN.md), 구현 [`docs/TREND_RADAR.md`](docs/TREND_RADAR.md).

```bash
pip install -r requirements.txt
python -m trend_radar run --input "data/raw/*.xlsx"   # → out/radar.html
python -m pytest tests/trend_radar
```

키워드 정제 규칙은 자동 적용하지 않는다. 후보는 `out/review/lexicon_candidates.xlsx`,
Claude의 제안은 `proposals/`, 승인한 규칙은 `lexicon.yaml`에 둔다. 원자료(`data/`)와 결과(`out/`)는 커밋하지 않는다.

---

# UZA

먼저 말을 걸기도 하는 대화 상대. 성격·행동 원칙은 `UZA.md`, 구현 설계는 [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md).

## 구조

| 파일 | 설계 문서 | 역할 |
|---|---|---|
| `uza/scheduler.py` | 2장 | 매시 선제 판단, 23:30 하루 정리, 일요일 검토 |
| `uza/gate.py` | 4장 | 하드 규칙 (시간, 횟수, 무응답, backoff, 방해 금지) |
| `uza/store.py` | 5장 | SQLite: messages, episodes, hypotheses, current_state, proactive_log, user_settings |
| `uza/context.py` | 6장 | 모델에 넘길 맥락 조립 |
| `uza/prompts.py` | 7장 | UZA.md 핵심부 + 용도별 지시 + JSON 스키마 |
| `uza/llm.py` | — | Claude 호출 (structured outputs로 JSON 강제) |
| `uza/engine.py` | 2, 8, 9장 | 선제 대화, 사용자 대화, 반응 추적, 하루 정리, 검토, 기억 통제 |

모델은 판단과 문장만 맡고, 횟수·시간·저장·반응 기록은 전부 코드가 한다.

## 실행

```bash
pip install -r requirements.txt
cp ../other-project/.env .env   # 또는 .env.example을 복사해 ANTHROPIC_API_KEY 입력
python -m uza                   # 콘솔 대화 + 백그라운드 스케줄러
python -m uza --env ../other-project/.env   # 복사하지 않고 경로로 지정
```

`.env`는 커밋되지 않는다 (`.gitignore`). 이미 설정된 환경 변수가 `.env`보다 우선한다.

대화 중 `/tick`(선제 판단 즉시 실행), `/reflect`, `/review`, `/memory`, `/quit`.

`UZA.md`의 1~4, 10~12, 16장이 모든 프롬프트 앞에 붙는다. 장 제목은 `## 숫자. 제목` 형식을 지켜야 한다.
설정은 `config.yaml` (모델, effort, DB 경로 포함).

## 테스트

```bash
python -m pytest
```

11장 시나리오는 `tests/test_scenarios.py`에 있다. 모델은 가짜로 대체해 코드가 지키는 규칙만 검증한다.
