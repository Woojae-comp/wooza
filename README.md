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
export ANTHROPIC_API_KEY=...
python -m uza            # 콘솔 대화 + 백그라운드 스케줄러
```

대화 중 `/tick`(선제 판단 즉시 실행), `/reflect`, `/review`, `/memory`, `/quit`.

`UZA.md`를 저장소 루트에 두면 1~4, 10~12, 16장이 모든 프롬프트 앞에 붙는다. 없으면 용도별 지시만으로 동작한다.
설정은 `config.yaml` (모델, effort, DB 경로 포함).

## 테스트

```bash
python -m pytest
```

11장 시나리오는 `tests/test_scenarios.py`에 있다. 모델은 가짜로 대체해 코드가 지키는 규칙만 검증한다.
