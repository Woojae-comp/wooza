# UZA 구현 설계

UZA.md(성격·행동 원칙)를 실제로 동작시키기 위한 구현 설계 문서.
원칙은 UZA.md에, **"누가 무엇을 하는가"**는 이 문서에 둔다.

---

## 1. 역할 분담

| 영역 | 담당 | 이유 |
|---|---|---|
| 매시간 깨어나기 | 코드 (스케줄러) | 모델은 스스로 실행되지 않는다 |
| 하드 규칙 (횟수, 시간, 무응답) | 코드 (게이트) | 모델 판단은 SEND 쪽으로 기운다 |
| 기억 저장·검색 | 코드 (DB) | 모델은 호출 사이에 아무것도 기억하지 않는다 |
| 반응 기록 | 코드 | 답장 여부·시간은 측정값이다 |
| 맥락 조립 | 코드 (컨텍스트 빌더) | 필요한 정보만 골라 프롬프트에 넣는다 |
| SEND / SKIP 판단 | 모델 | 자연스러움 판단은 언어 모델이 잘한다 |
| 메시지 작성 | 모델 | 말투와 톤 |
| 하루 정리·가설 갱신 | 모델 → 코드 저장 | 모델이 요약하고 코드가 저장한다 |

---

## 2. 전체 흐름

### 2.1 선제 대화 (매시간)

```
스케줄러 (매시 정각)
 ↓
게이트: 하드 규칙 검사 ── 실패 → 종료 (SKIP 기록)
 ↓ 통과
컨텍스트 빌더: DB에서 맥락 수집
 ↓
모델 호출 (proactive 프롬프트)
 ↓
JSON 응답 파싱 ── decision: SKIP → 기록 후 종료
 ↓ SEND
메시지 전송
 ↓
proactive_log 기록 (반응 대기 상태)
```

### 2.2 사용자 대화 (사용자가 먼저 말할 때)

```
사용자 메시지 수신
 ↓
대기 중인 선제 메시지가 있으면 반응 기록 갱신
 ↓
컨텍스트 빌더
 ↓
모델 호출 (chat 프롬프트)
 ↓
응답 전송 + messages 저장
```

### 2.3 하루 정리 (매일 23:30)

```
오늘 대화 전체 로드
 ↓
모델 호출 (reflection 프롬프트)
 ↓
JSON 응답 → episodes, current_state, hypotheses 갱신
```

### 2.4 주기적 사용자 모델 갱신 (매주 일요일 밤)

```
최근 7일 reflection + 전체 hypotheses 로드
 ↓
모델 호출 (review 프롬프트)
 ↓
가설 레벨 조정, 오래된 current_state 만료, active_hours 조정 제안
```

---

## 3. 설정값

```yaml
active_hours:
  start: "09:00"
  end: "23:00"
check_interval: "1h"
timezone: "Asia/Seoul"

limits:
  max_proactive_per_day: 4          # 하루 최대 선제 메시지
  min_gap_after_any_message: 90m    # 마지막 대화 후 최소 간격
  min_gap_after_unanswered: 4h      # 선제 메시지 무응답 시 대기
  max_consecutive_unanswered: 2     # 연속 무응답 N회면 당일 중단

backoff:
  # 최근 7일 선제 메시지 응답률에 따라 하루 최대치 조정
  - response_rate_below: 0.2
    max_proactive_per_day: 1
  - response_rate_below: 0.5
    max_proactive_per_day: 2

reflection_time: "23:30"
review_day: "sunday"

context:
  recent_messages: 20               # 프롬프트에 넣을 최근 메시지 수
  max_episodes: 5
  max_hypotheses: 8
```

---

## 4. 게이트 (하드 규칙)

모델을 호출하기 **전에** 코드가 검사한다. 하나라도 걸리면 모델을 부르지 않는다.

1. 현재 시각이 active_hours 밖이다.
2. 오늘 보낸 선제 메시지가 하루 최대치 이상이다 (backoff 반영).
3. 마지막 메시지(누가 보냈든)로부터 `min_gap_after_any_message`가 지나지 않았다.
4. 마지막 선제 메시지에 답이 없고, `min_gap_after_unanswered`가 지나지 않았다.
5. 오늘 연속 무응답이 `max_consecutive_unanswered` 이상이다.
6. 사용자가 "오늘은 말 걸지 마" 같은 요청을 했다 (`do_not_disturb_until`).

게이트가 막은 경우에도 `proactive_log`에 `decision: SKIP, reason: gate_*`로 남긴다.

---

## 5. 데이터 구조

### 5.1 messages

```yaml
id:
timestamp:
sender: user | uza
text:
is_proactive: bool
```

### 5.2 episodes (8.1 Episode)

```yaml
id:
date:
summary: "화요일에 중요한 회의가 있다고 했다"
related_date:        # 사건이 일어날/일어난 날짜 (있으면)
resolved: bool       # 결과를 들었는가
importance: low | mid | high
expires_at:          # 이후로는 컨텍스트에 넣지 않음
```

### 5.3 hypotheses (8.2 Long-term, 9장)

숫자 신뢰도 대신 **관찰 가능한 단계**를 쓴다.

```yaml
id:
statement: "전시를 좋아하는 편이다"
level: mentioned | repeated | confirmed
#  mentioned : 한 번 언급
#  repeated  : 서로 다른 날 2회 이상 관찰
#  confirmed : 사용자가 직접 말함
evidence: [episode_id, ...]
contradictions: [episode_id, ...]
last_seen:
sensitive: bool      # 건강, 인간관계, 재정 등
```

규칙:
- `mentioned`는 대화에서 확정적으로 언급하지 않는다.
- `contradictions`가 `evidence`보다 많아지면 review에서 삭제 또는 수정한다.
- `sensitive: true`인 가설은 사용자가 먼저 꺼내지 않는 한 선제 메시지에 쓰지 않는다.

### 5.4 current_state (8.3 Current State)

```yaml
id:
statement: "이번 주 프로젝트 마감으로 바쁨"
created_at:
expires_at:          # 기본 3일, 최대 14일
```

만료되면 자동으로 컨텍스트에서 빠진다. 영구 특성으로 승격하지 않는다.

### 5.5 proactive_log (13장)

```yaml
id:
timestamp:
decision: SEND | SKIP
reason:              # 모델이 준 이유 또는 gate_*
message_type:        # follow_up, observation, recall ...
message_id:
replied: bool
reply_delay_minutes:
response_length: short | normal | long
conversation_continued: bool   # 3턴 이상 이어졌는가
```

### 5.6 user_settings

```yaml
active_hours:        # review로 조정될 수 있음
do_not_disturb_until:
preferred_frequency: low | normal | high   # 사용자가 직접 말한 경우
```

---

## 6. 컨텍스트 빌더

모델에 넘기는 입력. UZA.md 7장 순서를 따른다.

```yaml
now: "2026-09-27 (일) 15:00"
last_message_at: "2026-09-27 12:40"
last_message_by: user
today_messages: [...]            # 오늘 대화
recent_episodes: [...]           # 미해결 우선, 최대 5개
unresolved_topics: [...]         # resolved: false 이고 related_date가 가까운 것
current_state: [...]             # 만료 안 된 것만
hypotheses: [...]                # repeated 이상 우선, 최대 8개, sensitive 제외
recent_proactive:                # 최근 5개
  - type: recall
    at: "2026-09-27 11:00"
    replied: false
today_proactive_count: 1
last_message_types: [recall, observation]   # 같은 유형 반복 방지용
```

원칙: **많이 넣지 않는다.** 모델은 받은 정보를 쓰고 싶어 한다. 넣은 만큼 "아는 척"이 늘어난다.

---

## 7. 프롬프트 구성

모든 호출은 `[UZA.md 핵심부] + [용도별 지시] + [컨텍스트]`로 구성한다.

UZA.md 핵심부 = 1~4장, 10~12장, 16장. (시간 규칙, 저장 구조 등 코드가 처리하는 장은 넣지 않는다.)

### 7.1 proactive 프롬프트

```
아래 맥락을 보고, 지금 사용자에게 먼저 말을 거는 것이 자연스러운지 판단해.

판단 기준:
- 이어갈 이야기, 결과를 못 들은 일, 시간대와 관련된 맥락이 있으면 SEND 쪽.
- 단순 안부밖에 할 말이 없으면 SKIP.
- 최근 보낸 메시지와 같은 주제나 같은 유형이면 SKIP.
- 확신이 없으면 SKIP. 침묵은 정상이다.

SEND라면:
- 친구가 메시지 하나 보내듯 1~3문장.
- 질문으로 끝낼 필요 없다.
- 기억을 억지로 끌어오지 않는다.
- 사용자를 분석하는 표현을 쓰지 않는다.
- last_message_types에 있는 유형은 피한다.

JSON만 출력:
{
  "decision": "SEND" | "SKIP",
  "reason": "한 줄",
  "message_type": "follow_up | observation | recall | sharing | check_in | playful | recommendation",
  "message": "SEND일 때만"
}
```

### 7.2 chat 프롬프트

```
사용자와 평소처럼 대화해.

- 기본은 1~3문장.
- 단, 사용자가 구체적인 설명이나 도움을 요청하면 필요한 만큼 길게 답해도 된다.
- 사용자가 힘든 상황을 털어놓으면 짧고 가벼운 톤보다 진지하게 듣는 것을 우선한다.
- 자해, 자살, 위험 신호가 보이면 말투 규칙보다 안전을 우선하고, 도움받을 수 있는 곳을 안내한다.
- 사용자가 "그거 잊어줘", "나에 대해 뭘 기억해?"라고 하면 아래 memory_command를 출력한다.

출력:
{
  "message": "...",
  "memory_command": null | { "action": "forget" | "list", "target": "..." },
  "do_not_disturb_until": null | "ISO 시각"
}
```

### 7.3 reflection 프롬프트 (14장)

```
오늘 대화를 보고 다음 대화의 연속성을 위해 짧게 정리해.
일기가 아니다. 추측을 사실처럼 쓰지 않는다.
한 번의 대화로 성격을 단정하지 않는다.

JSON만 출력:
{
  "date": "",
  "important_events": [{ "summary": "", "related_date": null, "importance": "low|mid|high" }],
  "resolved_episode_ids": [],
  "current_state": [{ "statement": "", "expires_in_days": 3 }],
  "hypothesis_updates": [
    { "id": null, "statement": "", "observation": "support | contradict | new", "sensitive": false }
  ],
  "unresolved_topics": [],
  "conversation_reaction": ""
}
```

### 7.4 review 프롬프트 (15장)

```
지난 7일 정리와 현재 가설 목록을 검토해.

- 관심사가 변했는가
- 예전에 중요했던 일이 지금도 중요한가
- 반박이 쌓인 가설이 있는가
- 선제 메시지 빈도가 적절했는가 (응답률 참고)
- 실제 활동 시간이 active_hours와 다른가

유지할 이유가 없는 판단은 수정한다.

JSON만 출력:
{
  "hypothesis_changes": [{ "id": "", "action": "keep | upgrade | downgrade | delete | rewrite", "new_statement": null }],
  "active_hours_suggestion": null | { "start": "", "end": "" },
  "notes": ""
}
```

---

## 8. 기억에 대한 사용자 통제

16장("감시하는 인상 주지 않기")을 실제로 지키기 위한 장치.

- **"그거 잊어줘"** → 해당 episode / hypothesis 삭제. 삭제했다고 짧게 말한다.
- **"나에 대해 뭘 기억해?"** → `repeated` 이상 가설과 최근 미해결 episode를 평범한 말로 보여준다. 숨기지 않는다.
- **"오늘은 말 걸지 마"** → `do_not_disturb_until` 설정. 다음 날 09:00까지 게이트가 막는다.
- **"자주 말 걸어도 돼" / "좀 덜 걸어"** → `preferred_frequency` 갱신, 하루 최대치에 반영.

---

## 9. 반응 추적 규칙

- 선제 메시지 후 사용자 메시지가 오면 → `replied: true`, `reply_delay_minutes` 기록.
- 이후 양쪽 합계 3턴 이상 이어지면 → `conversation_continued: true`.
- 다음 선제 판단 시점까지 답이 없으면 → `replied: false` 확정.
- 한 번의 무응답이나 짧은 답에는 의미를 두지 않는다. backoff는 **7일 응답률**로만 계산한다.

---

## 10. 구현 순서 제안

1. messages 저장 + chat 프롬프트만으로 일반 대화 동작
2. reflection으로 episodes / current_state 쌓기
3. 컨텍스트 빌더에 기억 연결 (대화에서 자연스럽게 회상되는지 확인)
4. 스케줄러 + 게이트 + proactive 프롬프트
5. proactive_log + backoff
6. hypotheses + review
7. 기억 통제 명령

선제 대화(4번)는 기억이 어느 정도 쌓인 뒤에 켜는 게 좋다. 기억 없이 먼저 말을 걸면 단순 안부밖에 할 말이 없다.

---

## 11. 확인해볼 것 (테스트 시나리오)

- 하루 종일 무응답일 때 선제 메시지가 몇 번 나가는가 → 2회 이하여야 한다.
- "화요일에 발표 있어"라고 한 뒤 화요일 저녁에 follow_up이 나오는가.
- "오늘 사람 만나기 싫다" 한 번으로 "내향적" 가설이 생기지 않는가.
- 같은 유형(recall 등)이 연속으로 나오지 않는가.
- "그거 잊어줘" 이후 해당 내용이 다시 등장하지 않는가.
- 힘든 이야기를 할 때 1~3문장 규칙에 묶여 가볍게 넘기지 않는가.

---

## 12. 구현 메모 (설계 대비 추가·해석한 부분)

실제 코드(`uza/`)를 짜면서 설계에 없던 부분을 이렇게 채웠다.

- **chat `memory_command`에 `target_ids` 추가.** 컨텍스트에 episode/hypothesis id가 들어가므로, 모델이 지울 대상을 id로 짚게 했다. id가 없으면 `target` 문구로 부분 일치 검색한다.
- **chat 출력에 `preferred_frequency` 추가.** 8장의 "자주 말 걸어도 돼 / 좀 덜 걸어"를 처리할 통로가 필요했다.
- **잊은 내용 목록(`forgotten`).** 삭제만 하면 그날 밤 reflection이 같은 대화를 다시 읽고 되살린다. 잊은 문구를 따로 남겨 reflection·chat 입력에 "다시 기록하거나 꺼내지 말 것"으로 넣고, 코드에서도 같은 문구의 재저장을 막는다.
- **reflection `hypothesis_updates`에 `user_stated` 추가.** `confirmed`(사용자가 직접 말함) 단계로 올릴 근거가 필요했다.
- **가설 evidence는 관찰 날짜로 쌓는다.** `repeated` 판정이 "서로 다른 날 2회 이상"이므로 날짜 단위가 직접적이다. 같은 날 여러 번은 1회로 센다.
- **새 가설은 항상 `mentioned`에서 시작한다.** 모델이 무엇을 주든 코드가 강제한다 (11장 "한 번으로 내향적 가설이 생기지 않는가").
- **같은 유형 연속 방지는 코드에서도 한 번 더 막는다.** 모델이 직전과 같은 `message_type`으로 SEND하면 `SKIP (repeat_type)`으로 바꾼다.
- **backoff 최소 표본.** 7일 동안 선제 메시지가 3개 미만이면 응답률 backoff를 적용하지 않는다 (9장 "한 번의 무응답에 의미를 두지 않는다").
- **preferred_frequency 반영.** `low`=하루 2회, `normal`=설정값, `high`=설정값+2. 그 다음 backoff 상한을 적용한다.
- **episode 해결 여부 기본값.** `related_date`가 오늘 이후면 `resolved: false`(결과를 아직 못 들음), 없거나 지났으면 `resolved: true`.
- **episode 만료 기본값.** `related_date`가 있으면 그 날 + 7일, 없으면 기록일 + 30일 (`high`는 60일).
