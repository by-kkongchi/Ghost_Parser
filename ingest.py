"""
인제스트 스크립트 — 프로젝트 폴더 하나를 통째로 청크 JSON으로 만든다
====================================================

사용법:
    python ingest.py "sample_data/frontend_handover_employee_sample"
    → 같은 위치에 chunks_<폴더명>.json 생성

하는 일 (RAG 인덱싱의 앞부분):
    폴더 안을 훑으며
      - *.md        → ① markdown_parser  (heading 청킹)
      - *.ts *.tsx  → ② code_parser      (선언 단위 청킹)
      - .git 존재 시 → ③ git_parser       (커밋 히스토리)
    를 모두 실행하고 ④ normalizer로 공통 스키마로 합쳐서 저장한다.

    ★ 이 JSON은 "임베딩 직전" 상태다.
      실제 RAG를 쓰려면 embed_and_upload.py 로 Qdrant에 올려야 한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from parsers import parse_markdown, parse_ts, parse_local_git_log
from normalizer import normalize_markdown, normalize_code, normalize_git, build_embedding_text

# 파싱에서 제외할 폴더들 (의존성/빌드 산출물은 팀원 기여물이 아님)
# __MACOSX: 맥에서 zip으로 압축하면 자동으로 생기는 메타데이터 폴더
SKIP_DIRS = {".git", "node_modules", "dist", "build", "__pycache__", ".next", "__MACOSX"}


def ingest_project(project_dir: str | Path, author: str | None = None) -> list[dict]:
    """
    프로젝트 폴더 하나를 파싱해서 공통 청크 리스트로 반환한다.

    Args:
        author: 업로드한 팀원 이름 — 코드 청크의 author 필드에 들어간다.
                (파일 자체에는 작성자 정보가 없으므로 업로드 시점에 받아야 함)
    """
    root = Path(project_dir)
    chunks: list[dict] = []

    # ---- ①② 문서 + 코드 파일 순회 -------------------------------------
    for path in sorted(root.rglob("*")):
        # 제외 폴더 아래 파일은 건너뛴다
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue

        rel = str(path.relative_to(root))  # 청크에는 프로젝트 기준 상대경로 저장

        try:
            if path.suffix == ".md":
                chunks += normalize_markdown(parse_markdown(path), rel)
            elif path.suffix in (".ts", ".tsx"):
                chunks += normalize_code(parse_ts(path), rel, author=author)
        except Exception as e:
            # 파일 하나가 깨져도 전체 인제스트는 계속 진행
            print(f"  ⚠️ 파싱 실패 (건너뜀): {rel} — {e}")

    # ---- ③ git 히스토리 (폴더가 git 저장소인 경우) ----------------------
    if (root / ".git").exists():
        try:
            commits = parse_local_git_log(root)
            chunks += normalize_git(commits)
            print(f"  git 커밋 {len(commits)}개 파싱")
        except Exception as e:
            print(f"  ⚠️ git 파싱 실패: {e}")

    return chunks


if __name__ == "__main__":
    target = Path(sys.argv[1])
    chunks = ingest_project(target)

    # 소스 타입별 개수 요약 출력 (뭐가 얼마나 들어갔는지 한눈에 확인용)
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c["source_type"]] = counts.get(c["source_type"], 0) + 1
    print(f"\n총 {len(chunks)}개 청크: {counts}")

    # 각 청크에 "임베딩에 들어갈 최종 텍스트"도 미리 계산해서 같이 저장
    # → JSON만 열어봐도 모델에 뭐가 들어가는지 검수할 수 있다
    for c in chunks:
        c["embedding_text"] = build_embedding_text(c)

    out_path = target.parent / f"chunks_{target.name.replace(' ', '_')}.json"
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out_path}")
