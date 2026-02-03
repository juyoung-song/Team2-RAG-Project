# 🤝 협업 가이드 (Collaboration Guide)

우리 팀의 일관된 코드 품질과 효율적인 협업을 위해 아래 워크플로우를 준수합니다.

## 🔄 작업 프로세스 (Standard Workflow)

1. **Issue 생성**: 작업할 내용을 [Issues] 탭에 템플릿에 맞춰 내용을 등록하고 담당자를 지정합니다.
2. **Branch 생성**: `dev` 브랜치에서 분기하여 기능 단위 브랜치를 만듭니다.
   - 형식: `feature/이슈번호-기능명` 또는 `fix/이슈번호-버그명` (예: `feature/12-ingestion-pdf`) 
3. **코드 작성 및 커밋**: 로컬에서 작업 후 커밋 컨벤션에 맞춰 커밋을 남깁니다.
4. **Push & Pull Request(PR)**: 작업 브랜치를 원격에 올리고 `dev` 브랜치로 PR을 보냅니다.
5. **Code Review**: 팀원의 리뷰를 거쳐 `Approve`를 획득합니다.
6. **Merge**: 리뷰가 완료된 코드를 `dev`에 반영하고 브랜치를 삭제합니다.
7. **Release to main**: 모든 기능이 완성되어 배포 가능한 상태가 되면, **dev → main**으로 최종 PR을 보내 병합합니다.

---

## 📝 이슈 생성 가이드 (How to Create an Issue)

1. 템플릿 선택: Issues -> New issue에서 카테고리에 맞는 템플릿을 선택합니다.

2. 정보 입력: Assignees에 담당자를 지정하고, 아래의 3가지 필수 레이블을 설정합니다.

Type: 템플릿에 따라 자동 지정 (feat, bug, exp 등)

Priority: 작업의 우선순위 (Urgent 시 팀 채널에 공유 필수!)

Status: Pending(대기), In Progress(진행 중), Completed(완료/닫기), Aborted(중단)

작성 원칙: 너무 완벽하게 적으려 고생하지 마세요! 팀원이 이해할 수 있을 정도로 간단하고 핵심만 적어주셔도 충분합니다.

---

## 🌿 브랜치 네이밍 규칙

브랜치 이름은 다음 두 가지 유형 중 하나를 따릅니다.

**`feature` 브랜치**

- 새로운 기능 개발에 사용합니다.
- 형식: `feat/{issue-number}-short-description`
    - `{issue-number}`: 해당 작업의 이슈 번호
    - `short-description`: 기능을 간단히 설명하는 이름
- 예시:
    - `feat/123-user-login`
    - `feat/45-add-item-to-cart`

**`hotfix` 브랜치**

- 긴급한 수정 작업에 사용합니다.
- 형식: `hotfix-short-description`
    - `short-description`: 수정할 문제를 간단히 설명
- 예시:
    - `hotfix-fix-login-error`
    - `hotfix-correct-typo`

**세부 규칙**

- 구분자 사용
    - `feature` 뒤에는 `/`를 사용해 폴더 구조처럼 구분합니다.
    - 기능 이름은 로 단어를 연결합니다.
- 작성 원칙
    - 브랜치 이름은 **작업 내용을 명확히 표현**하도록 간결하게 작성합니다.
    - 이슈 번호와 연동하여 작업 내역을 추적합니다.

대부분의 경우에는 `feature` 브랜치를 사용하시면 됩니다.

---

## 💬 커밋 메시지 규칙 (Commit Convention)

메시지는 **`Type: 요약`** 형식을 사용하며, 필요시 상세 내용을 본문에 적습니다.

| Type | 설명 | 예시 |
| :--- | :--- | :--- |
| **feat** | 새로운 기능 추가 | `feat: PDF 파싱 및 텍스트 추출 로직 구현` |
| **fix** | 버그 수정 | `fix: 벡터 DB 연결 타임아웃 오류 수정` |
| **data** | 원본 데이터(RFP, CSV 등) 추가/변경 | `data: 2차 가공된 RFP 텍스트 데이터셋 추가` |
| **prompt** | 프롬프트 엔지니어링 관련 수정 | `prompt: RAG 답변 생성을 위한 시스템 프롬프트 최적화` |
| **refactor** | 코드 리팩토링 (기능 변화 없음) | `refactor: 전처리 함수 모듈화 및 가독성 개선` |
| **docs** | 문서 수정 (README, 주석 등) | `docs: README 내 협업 가이드 업데이트` |
| **chore** | 단순 환경 설정, 패키지 설정 | `chore: poetry 라이브러리 추가 (.gitignore 수정)` |
| **perf** | 성능 개선 (검색 속도 최적화 등) | `perf: FAISS 인덱싱 성능 최적화` |
| **test** | 테스트 코드 추가 및 수정 | `test: 데이터 로더 단위 테스트 추가` |

---

## 🔍 Pull Request(PR) 규칙

- **리뷰어 지정**: 작업 내용과 관련된 팀원을 반드시 리뷰어로 지정합니다.
- **이슈 연결**: PR 본문에 `Closes #이슈번호`를 포함하여 머지 시 이슈가 자동 종료되도록 합니다.
- **Self-Review**: PR 생성 후 자신의 코드를 다시 한 번 검토하며 변경 사항을 요약합니다.

---

## 📁 폴더 및 파일 관리

- **.env**: `.env` 파일은 절대 커밋하지 않으며, `example.env`를 통해 필요한 키 목록만 공유합니다.
- **Notebooks**: 실험용 노트북은 `notebooks/YYMMDD-작성자-주제.ipynb` 형식으로 저장합니다.
- **Conflict 해결**: 충돌 발생 시, 로컬에서 `dev` 브랜치를 `pull` 받은 뒤 작업 브랜치에 `merge` 하여 해결한 후 다시 푸시합니다.
```bash
  git checkout feat/이슈번호-기능명
  git pull origin dev
  # 충돌 해결 후
  git push origin feat/이슈번호-기능명
```