"""
④ 공통 JSON Normalizer — 파싱 결과 → Qdrant 업로드용 공통 청크
====================================================

[왜 노멀라이저가 필요한가?]
  세 파서(마크다운/코드/깃)의 출력 형태가 전부 다르다.
  그런데 Qdrant에 올릴 때는:
    1. 임베딩할 텍스트가 "한 개의 문자열"로 정해져 있어야 하고
    2. 검색 후 필터링("코드만", "PR만", "김OO 것만")을 하려면
       payload 필드 이름이 소스와 무관하게 통일돼 있어야 한다.
  → 그래서 모든 소스를 아래 "공통 청크 스키마" 하나로 변환한다.

[공통 청크 스키마]
  {
      "chunk_id":    "uuid",             # Qdrant point ID로 그대로 사용
      "source_type": "requirement_doc"   # 검색 필터의 핵심 축
                    | "code"
                    | "git_commit"
                    | "pull_request",
      "title":       "사람이 읽는 제목",   # 검색 결과 표시용
      "content":     "본문 텍스트",        # 임베딩 대상의 몸통
      "author":      "작성자" | None,     # 팀원별 기여 분석의 핵심 축
      "created_at":  "ISO8601" | None,
      "file_path":   "원본 파일 경로" | None,
      "metadata":    { ...소스별 고유 정보... },  # 통일 불가능한 나머지는 여기로
  }

[사용 흐름]
  md_chunks   = parse_markdown("설계문서.md")
  code_chunks = parse_ts("Login.tsx")
  git_data    = fetch_github_data("owner", "repo", token)

  chunks  = (normalize_markdown(md_chunks, "설계문서.md")
           + normalize_code(code_chunks, "Login.tsx")
           + normalize_git(git_data))

  points  = to_qdrant_points(chunks, embed_fn=내_임베딩_함수)
  client.upsert(collection_name="ghost_member", points=points)
"""

from __future__ import annotations

import uuid


# ===========================================================================
# 1) 소스별 노멀라이즈 함수들
# ===========================================================================

def normalize_markdown(md_chunks: list[dict], file_path: str) -> list[dict]:
    """
    ① markdown_parser.parse_markdown() 출력 → 공통 청크

    포인트: heading_path(상위 제목 경로)를 title에 합쳐 넣는다.
    "요구사항 > 3. 핵심 기능 > F1. 자동 가계부" 처럼 경로가 임베딩에 포함되면
    "핵심 기능 중 가계부 관련" 같은 질의에 걸릴 확률이 크게 올라간다.
    """
    out = []
    for c in md_chunks:
        # heading 경로를 " > "로 이어 문맥이 살아있는 제목을 만든다
        title = " > ".join(c["heading_path"]) if c["heading_path"] else c["heading"]
        out.append({
            "chunk_id": str(uuid.uuid4()),
            "source_type": "requirement_doc",
            "title": title,
            "content": c["content"],
            "author": None,  # md 파일 자체에는 작성자 정보가 없음 (필요 시 업로더가 지정)
            "created_at": None,
            "file_path": file_path,
            "metadata": {
                "heading_level": c["level"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                # frontmatter가 있었다면 보존 (제목/작성자 등이 들어있을 수 있음)
                **({"frontmatter": c["frontmatter"]} if "frontmatter" in c else {}),
            },
        })
    return out


def normalize_code(code_chunks: list[dict], file_path: str,
                   author: str | None = None) -> list[dict]:
    """
    ② code_parser.parse_ts() 출력 → 공통 청크

    Args:
        author: 업로드한 사람 (파일 자체에는 작성자가 없으므로
                업로드 시점에 웹에서 받은 사용자명을 넣어준다)

    포인트: title을 "파일명 :: 종류 이름" 형태로 만들어
    검색 결과만 봐도 어떤 코드인지 알 수 있게 한다.
    예) "Login.tsx :: component LoginForm"
    """
    out = []
    for c in code_chunks:
        out.append({
            "chunk_id": str(uuid.uuid4()),
            "source_type": "code",
            "title": f"{file_path} :: {c['kind']} {c['name']}",
            "content": c["content"],
            "author": author,
            "created_at": None,
            "file_path": file_path,
            "metadata": {
                "kind": c["kind"],              # function/class/component ...
                "exported": c["exported"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "imports": c["imports"],        # 의존성 분석용
                "hooks_used": c["hooks_used"],  # React 훅 사용 분석용
            },
        })
    return out


def normalize_git(git_data: dict | list) -> list[dict]:
    """
    ③ git_parser 출력 → 공통 청크

    두 형태를 모두 받는다:
      - fetch_github_data() 결과: {"commits": [...], "pull_requests": [...]}
      - parse_local_git_log() 결과: [커밋 dict ...]

    포인트:
      - 커밋과 PR을 별도 source_type으로 나눈다.
        ("커밋만 검색" vs "PR 논의만 검색"이 가능해야 하므로)
      - content에는 제목+본문+변경파일 목록을 합쳐 넣는다.
        커밋 메시지 한 줄만으로는 임베딩 정보량이 부족하기 때문.
    """
    # 입력 형태 통일: 리스트로 오면 커밋 목록으로 간주
    if isinstance(git_data, list):
        commits, prs = git_data, []
    else:
        commits = git_data.get("commits", [])
        prs = git_data.get("pull_requests", [])

    out = []

    # ---- 커밋 → 공통 청크 -------------------------------------------------
    for c in commits:
        # 변경 파일 목록을 텍스트로 요약해 임베딩에 포함
        # ("auth 관련 작업한 사람?" 같은 질의가 파일 경로에 걸리게 하기 위함)
        files_text = "\n".join(
            f"- {f['path']} (+{f['added']}/-{f['deleted']})" for f in c.get("files", [])
        )
        content = c["subject"]
        if c.get("body"):
            content += "\n\n" + c["body"]
        if files_text:
            content += "\n\n[변경 파일]\n" + files_text

        out.append({
            "chunk_id": str(uuid.uuid4()),
            "source_type": "git_commit",
            "title": c["subject"],
            "content": content,
            "author": c.get("author"),
            "created_at": c.get("date"),
            "file_path": None,
            "metadata": {
                "sha": c.get("sha"),
                "email": c.get("email"),
                "files_changed": len(c.get("files", [])),
                "lines_added": sum(f["added"] for f in c.get("files", [])),
                "lines_deleted": sum(f["deleted"] for f in c.get("files", [])),
            },
        })

    # ---- PR → 공통 청크 (★필수 데이터) -------------------------------------
    for pr in prs:
        # PR은 제목+설명이 곧 "이 작업이 무엇이었는지"의 최고 요약본이다
        content = pr["title"]
        if pr.get("body"):
            content += "\n\n" + pr["body"]
        content += f"\n\n[브랜치] {pr.get('head', '?')} → {pr.get('base', '?')}"
        if pr.get("labels"):
            content += "\n[라벨] " + ", ".join(pr["labels"])

        out.append({
            "chunk_id": str(uuid.uuid4()),
            "source_type": "pull_request",
            "title": f"PR #{pr['number']}: {pr['title']}",
            "content": content,
            "author": pr.get("author"),
            "created_at": pr.get("created_at"),
            "file_path": None,
            "metadata": {
                "pr_number": pr.get("number"),
                "state": pr.get("state"),           # merged/open/closed — 기여도 판단 핵심
                "merged_at": pr.get("merged_at"),
                "base": pr.get("base"),
                "head": pr.get("head"),
                "commit_shas": pr.get("commit_shas", []),  # 커밋 청크와의 연결고리
            },
        })

    return out


# ===========================================================================
# 2) 임베딩 텍스트 생성 + Qdrant 포인트 변환
# ===========================================================================

def build_embedding_text(chunk: dict, max_chars: int = 4000) -> str:
    """
    공통 청크 하나를 "임베딩 모델에 넣을 최종 문자열"로 만든다.

    형식:  [소스타입] 제목\n본문
    - 소스타입/제목을 본문 앞에 붙이면 짧은 커밋 메시지도 문맥을 갖는다.
    - max_chars로 자르는 이유: 임베딩 모델 입력 토큰 한도 초과 방지.
      (대부분의 임베딩 모델이 512 토큰 내외 — 한글 기준 여유 있게 4000자)
    """
    label = {
        "requirement_doc": "요구사항 문서",
        "code": "코드",
        "git_commit": "커밋",
        "pull_request": "Pull Request",
    }.get(chunk["source_type"], chunk["source_type"])

    text = f"[{label}] {chunk['title']}\n{chunk['content']}"
    return text[:max_chars]


def to_qdrant_points(chunks: list[dict], embed_fn) -> list[dict]:
    """
    공통 청크 리스트 → Qdrant upsert용 포인트 리스트

    Args:
        embed_fn: 문자열 리스트를 받아 벡터 리스트를 반환하는 함수.
                  예) watsonx.ai 임베딩:
                      def embed_fn(texts):
                          return model.embed_documents(texts=texts)

    Returns:
        [{"id": ..., "vector": [...], "payload": {...}}, ...]
        → qdrant_client의 upsert(points=...)에 그대로 넣으면 된다.
          (PointStruct(**p) 형태로 감싸도 됨)
    """
    # 임베딩은 한 건씩 호출하면 느리고 비싸므로 반드시 배치로 호출한다
    texts = [build_embedding_text(c) for c in chunks]
    vectors = embed_fn(texts)

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append({
            "id": chunk["chunk_id"],
            "vector": vector,
            # payload = 검색 결과와 함께 돌려받을 원본 정보 + 필터링 축.
            # source_type / author 필드에 Qdrant 인덱스를 걸면
            # "PR 중에서" "김OO이 작성한" 같은 필터 검색이 빨라진다.
            "payload": {
                "source_type": chunk["source_type"],
                "title": chunk["title"],
                "content": chunk["content"],
                "author": chunk["author"],
                "created_at": chunk["created_at"],
                "file_path": chunk["file_path"],
                **chunk["metadata"],  # 소스별 고유 정보도 평탄화해서 포함
            },
        })
    return points


if __name__ == "__main__":
    # 파서 3종 → 노멀라이저 전체 흐름 데모 (임베딩은 더미 벡터로 대체)
    import json
    from parsers.markdown_parser import parse_markdown  # noqa: F401 (사용 예시)

    sample_git = {
        "commits": [{
            "type": "commit", "sha": "abc123", "author": "kim",
            "email": "kim@x.com", "date": "2026-08-20T10:00:00+09:00",
            "subject": "feat: 로그인 구현", "body": "",
            "files": [{"path": "src/auth.ts", "added": 30, "deleted": 2}],
        }],
        "pull_requests": [{
            "type": "pull_request", "number": 1, "title": "로그인 기능",
            "body": "JWT 기반 로그인", "author": "kim", "state": "merged",
            "created_at": "2026-08-20T12:00:00Z", "merged_at": "2026-08-21T09:00:00Z",
            "base": "main", "head": "feature/login", "labels": [], "commit_shas": ["abc123"],
        }],
    }
    chunks = normalize_git(sample_git)
    points = to_qdrant_points(chunks, embed_fn=lambda ts: [[0.0] * 8 for _ in ts])
    print(json.dumps(points, ensure_ascii=False, indent=2))
