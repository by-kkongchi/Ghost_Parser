"""
③ Git Log Parser — 깃 로그 + Pull Request 파서
====================================================

[역할]
  두 가지 소스에서 개발 활동 데이터를 가져와 파싱한다.

  (A) parse_local_git_log()  — 로컬에 clone된 저장소에서 `git log` 실행 결과 파싱
  (B) fetch_github_data()    — GitHub REST API로 커밋 + Pull Request 수집
                               ★ PR 정보는 요구사항상 필수이므로 (B)가 메인.
                               PR(리뷰/머지/설명)은 로컬 git log에 없고
                               GitHub 서버에만 있는 데이터이기 때문.

[출력 형태] (노멀라이저가 받아가는 중간 포맷)
  커밋:
  {
      "type": "commit",
      "sha": "abc123...",
      "author": "kim", "email": "kim@x.com",
      "date": "2026-08-20T10:00:00+09:00",
      "subject": "feat: 로그인 구현",
      "body": "상세 설명...",
      "files": [{"path": "src/auth.ts", "added": 30, "deleted": 2}],
  }
  PR:
  {
      "type": "pull_request",
      "number": 12,
      "title": "로그인 기능",
      "body": "PR 설명...",
      "author": "kim",
      "state": "merged" | "open" | "closed",
      "created_at": ..., "merged_at": ...,
      "base": "main", "head": "feature/login",
      "labels": [...], "commit_shas": [...],
  }
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# (A) 로컬 git log 파싱
# ---------------------------------------------------------------------------

# git log 출력에서 필드를 안전하게 자르기 위한 구분자.
# 커밋 메시지에 쉼표/줄바꿈이 들어가도 깨지지 않도록
# 일반 텍스트에 절대 안 나오는 ASCII 제어문자를 쓴다.
FIELD_SEP = "\x1f"   # unit separator  : 필드 사이
RECORD_SEP = "\x1e"  # record separator: 커밋 레코드 사이

# %H=전체 SHA, %an=작성자, %ae=이메일, %aI=ISO8601 날짜, %s=제목, %b=본문
# RECORD_SEP을 레코드 "앞"에 붙이는 이유:
#   --numstat 줄들은 pretty format 출력 "뒤"에 이어서 출력된다.
#   구분자를 뒤에 붙이면 numstat이 다음 커밋 레코드에 섞여 버리므로,
#   앞에 붙여서 "구분자 ~ 다음 구분자 = 메타데이터 + 그 커밋의 numstat"이 되게 한다.
PRETTY_FORMAT = RECORD_SEP + FIELD_SEP.join(["%H", "%an", "%ae", "%aI", "%s", "%b"])


def parse_local_git_log(repo_path: str | Path, max_commits: int = 500) -> list[dict]:
    """
    로컬 저장소에서 `git log --numstat`을 실행해 커밋 리스트로 파싱한다.

    --numstat을 쓰는 이유:
      커밋마다 "어떤 파일을 몇 줄 추가/삭제했는지"가 나온다.
      → 팀원별 기여도 분석(누가 어떤 파일을 얼마나 만졌나)에 핵심 데이터.

    Args:
        repo_path: git 저장소 경로 (.git이 있는 폴더)
        max_commits: 가져올 최대 커밋 수 (해커톤 데모용 안전장치)
    """
    cmd = [
        "git", "-C", str(repo_path), "log",
        f"--max-count={max_commits}",
        f"--pretty=format:{PRETTY_FORMAT}",
        "--numstat",  # 각 커밋 아래에 "추가수 \t 삭제수 \t 파일경로" 줄들이 붙음
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    commits: list[dict] = []

    # RECORD_SEP 기준으로 커밋 레코드를 자른다.
    # 각 레코드 = "메타데이터(FIELD_SEP 구분)" + 그 뒤에 numstat 줄들
    for record in out.split(RECORD_SEP):
        if not record.strip():
            continue

        fields = record.split(FIELD_SEP)
        if len(fields) < 6:
            continue  # 형식이 깨진 레코드는 건너뜀 (방어적 처리)

        sha, author, email, date, subject = fields[0].strip(), *fields[1:5]
        # 마지막 필드(본문)에는 numstat 줄들이 이어붙어 있으므로 분리한다
        body_and_stats = fields[5]
        body_lines: list[str] = []
        files: list[dict] = []

        for line in body_and_stats.splitlines():
            parts = line.split("\t")
            # numstat 줄 형식: "30\t2\tsrc/auth.ts" (바이너리 파일은 "-\t-\t경로")
            if len(parts) == 3 and (parts[0].isdigit() or parts[0] == "-"):
                files.append({
                    "path": parts[2],
                    "added": int(parts[0]) if parts[0].isdigit() else 0,
                    "deleted": int(parts[1]) if parts[1].isdigit() else 0,
                })
            else:
                body_lines.append(line)

        commits.append({
            "type": "commit",
            "sha": sha,
            "author": author,
            "email": email,
            "date": date,
            "subject": subject,
            "body": "\n".join(body_lines).strip(),
            "files": files,
        })

    return commits


# ---------------------------------------------------------------------------
# (B) GitHub API — 커밋 + Pull Request (★필수)
# ---------------------------------------------------------------------------

API_BASE = "https://api.github.com"


def _github_get(url: str, token: str | None) -> list | dict:
    """
    GitHub REST API GET 요청 헬퍼.

    - token 없이도 공개 저장소는 조회 가능하지만 시간당 60회 제한이라
      해커톤 시연 중 막히기 쉽다. → Personal Access Token 발급을 권장.
    - 표준 라이브러리(urllib)만 사용해서 의존성을 없앴다.
    """
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _paginate(url: str, token: str | None, max_pages: int = 10) -> list[dict]:
    """
    GitHub API는 한 번에 최대 100개까지만 주므로 page 파라미터로 반복 조회한다.
    max_pages로 상한을 둬서 대형 저장소에서도 폭주하지 않게 한다.
    """
    results: list[dict] = []
    for page in range(1, max_pages + 1):
        sep = "&" if "?" in url else "?"
        batch = _github_get(f"{url}{sep}per_page=100&page={page}", token)
        if not batch:
            break
        results.extend(batch)
        if len(batch) < 100:  # 마지막 페이지
            break
    return results


def fetch_github_data(
    owner: str,
    repo: str,
    token: str | None = None,
    include_pr_commits: bool = True,
) -> dict:
    """
    GitHub 저장소에서 커밋 + PR을 모두 수집해 파싱한다.

    Args:
        owner: 저장소 소유자 (예: "ghost-member-team")
        repo:  저장소 이름   (예: "backend")
        token: GitHub Personal Access Token (권장)
        include_pr_commits: PR마다 포함된 커밋 SHA 목록까지 추가 조회할지.
                            PR 수만큼 API 호출이 늘어나므로 저장소가 크면 False로.

    Returns:
        {"commits": [커밋 dict...], "pull_requests": [PR dict...]}
    """
    # ---- 1) 커밋 목록 --------------------------------------------------
    raw_commits = _paginate(f"{API_BASE}/repos/{owner}/{repo}/commits", token)
    commits: list[dict] = []
    for c in raw_commits:
        commit = c.get("commit", {})
        subject, _, body = commit.get("message", "").partition("\n")
        commits.append({
            "type": "commit",
            "sha": c.get("sha", ""),
            # author가 GitHub 계정과 연결 안 된 경우(None) 커밋 서명의 이름으로 대체
            "author": (c.get("author") or {}).get("login")
                      or commit.get("author", {}).get("name", "unknown"),
            "email": commit.get("author", {}).get("email", ""),
            "date": commit.get("author", {}).get("date", ""),
            "subject": subject.strip(),
            "body": body.strip(),
            "files": [],  # 목록 API에는 파일 정보가 없음 (상세 API는 커밋당 1호출이라 생략)
        })

    # ---- 2) Pull Request 목록 (★요구사항 필수 항목) ---------------------
    # state=all : open + closed + merged 전부 가져온다.
    # 팀원 기여 분석에는 "머지된 PR"이 제일 중요하지만
    # 열려만 두고 방치된 PR도 활동 신호이므로 전부 수집한다.
    raw_prs = _paginate(f"{API_BASE}/repos/{owner}/{repo}/pulls?state=all", token)
    pull_requests: list[dict] = []
    for pr in raw_prs:
        number = pr.get("number")

        # merged 여부는 state 필드만으로는 알 수 없다.
        # closed + merged_at 존재 → 실제로 머지됨 / closed + merged_at 없음 → 그냥 닫힘
        if pr.get("merged_at"):
            state = "merged"
        else:
            state = pr.get("state", "unknown")

        entry = {
            "type": "pull_request",
            "number": number,
            "title": pr.get("title", ""),
            "body": pr.get("body") or "",   # 설명 없는 PR은 null이 옴 → 빈 문자열로
            "author": (pr.get("user") or {}).get("login", "unknown"),
            "state": state,
            "created_at": pr.get("created_at", ""),
            "merged_at": pr.get("merged_at"),
            "base": pr.get("base", {}).get("ref", ""),   # 머지 대상 브랜치 (보통 main)
            "head": pr.get("head", {}).get("ref", ""),   # 작업 브랜치
            "labels": [l.get("name") for l in pr.get("labels", [])],
            "commit_shas": [],
        }

        # PR ↔ 커밋 연결: "이 PR에 어떤 커밋들이 들어갔나"
        # → 노멀라이저에서 PR과 커밋을 서로 참조할 수 있게 해주는 연결고리
        if include_pr_commits and number is not None:
            try:
                pr_commits = _paginate(
                    f"{API_BASE}/repos/{owner}/{repo}/pulls/{number}/commits", token)
                entry["commit_shas"] = [c.get("sha") for c in pr_commits]
            except Exception:
                pass  # 개별 PR 조회 실패가 전체 수집을 막지 않도록

        pull_requests.append(entry)

    return {"commits": commits, "pull_requests": pull_requests}


if __name__ == "__main__":
    # 단독 실행 테스트용:
    #   python git_parser.py /path/to/local/repo          → 로컬 git log 파싱
    #   python git_parser.py owner/repo [토큰]             → GitHub API 수집
    import sys

    arg = sys.argv[1]
    if "/" in arg and not Path(arg).exists():
        owner, repo = arg.split("/", 1)
        token = sys.argv[2] if len(sys.argv) > 2 else None
        data = fetch_github_data(owner, repo, token)
        print(f"commits: {len(data['commits'])}, PRs: {len(data['pull_requests'])}")
    else:
        commits = parse_local_git_log(arg)
        print(f"commits: {len(commits)}")
        print(json.dumps(commits[:2], ensure_ascii=False, indent=2))
