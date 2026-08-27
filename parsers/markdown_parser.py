"""
① Markdown Parser — 요구사항/설계 문서(.md) 파서
====================================================

[역할]
  요구사항 설계 문서를 heading(#, ##, ### ...) 기준으로 잘라서
  "섹션 청크" 리스트로 만든다.

[왜 heading 기준으로 자르나?]
  - 설계 문서는 보통 heading 하나가 하나의 주제(기능, 요구사항 항목)를 담는다.
  - 임베딩 검색 시 "문서 전체"가 아니라 "관련 섹션"만 검색되게 하려면
    의미 단위(heading)로 잘라야 검색 품질이 좋아진다.

[출력 형태] (노멀라이저가 받아가는 중간 포맷)
  {
      "heading": "F1. 자동 가계부",          # 이 섹션의 제목
      "level": 2,                            # heading 레벨 (# = 1, ## = 2 ...)
      "heading_path": ["3. 핵심 기능", "F1. 자동 가계부"],
                                             # 상위 heading 경로 (문맥 보존용)
      "content": "섹션 본문 텍스트...",
      "start_line": 42,                      # 원본에서의 시작 줄 번호
      "end_line": 60,
  }
"""

from __future__ import annotations

import re
from pathlib import Path


# heading 판별용 정규식: "# 제목", "## 제목" ... (# 뒤에 공백 필수 — 마크다운 표준)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# 코드펜스( ``` 또는 ~~~ ) 판별용. 코드블록 안의 "# 주석"을 heading으로
# 착각하지 않기 위해 반드시 추적해야 한다.
CODE_FENCE_RE = re.compile(r"^(```|~~~)")


def _strip_frontmatter(lines: list[str]) -> tuple[dict, list[str], int]:
    """
    문서 맨 위의 YAML frontmatter(--- ... ---)를 분리한다.

    노션 export나 정적 사이트용 md에는 메타데이터 블록이 붙어 있는 경우가 많다.
    본문 파싱에 섞이면 안 되므로 미리 떼어내고,
    (frontmatter dict, 남은 본문 lines, 본문 시작 줄번호)를 돌려준다.
    """
    if not lines or lines[0].strip() != "---":
        return {}, lines, 0

    meta: dict[str, str] = {}
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            # frontmatter 내부는 "key: value" 형태만 단순 파싱 (외부 yaml 의존성 제거)
            for raw in lines[1:i]:
                if ":" in raw:
                    k, _, v = raw.partition(":")
                    meta[k.strip()] = v.strip()
            return meta, lines[i + 1:], i + 1

    # 닫는 ---가 없으면 frontmatter가 아니라고 판단하고 원본 그대로 반환
    return {}, lines, 0


def parse_markdown(
    file_path: str | Path,
    min_chunk_chars: int = 30,
) -> list[dict]:
    """
    md 파일을 heading 단위 청크 리스트로 파싱한다.

    Args:
        file_path: 파싱할 .md 파일 경로
        min_chunk_chars: 본문이 이 글자 수보다 짧은 섹션은 버린다.
                         (빈 heading, 목차용 heading 등 노이즈 제거 목적)

    Returns:
        청크 dict 리스트 (모듈 상단 docstring의 출력 형태 참고)
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    frontmatter, lines, offset = _strip_frontmatter(lines)

    chunks: list[dict] = []

    # heading_stack: 현재 위치의 상위 heading 경로를 유지하는 스택.
    # 예: [("1. 문제 정의", 1), ("배경", 2)] 상태에서 level 2 heading을 만나면
    #     level >= 2 인 항목을 pop 하고 새 heading을 push 한다.
    # 이렇게 하면 각 청크가 "어느 대제목 아래의 소제목인지" 문맥을 갖게 되어
    # 임베딩 텍스트에 경로를 포함시킬 수 있다. (검색 품질에 중요!)
    heading_stack: list[tuple[str, int]] = []

    current: dict | None = None   # 지금 본문을 모으고 있는 청크
    buffer: list[str] = []        # 현재 청크의 본문 줄들
    in_code_fence = False         # 코드블록 내부인지 여부

    def flush(end_line: int) -> None:
        """모아둔 buffer를 청크로 확정해서 chunks에 추가한다."""
        nonlocal current, buffer
        if current is None:
            # 첫 heading 이전의 서문(preamble)도 버리지 않고 청크로 만든다.
            body = "\n".join(buffer).strip()
            if len(body) >= min_chunk_chars:
                chunks.append({
                    "heading": "(서문)",
                    "level": 0,
                    "heading_path": [],
                    "content": body,
                    "start_line": offset + 1,
                    "end_line": end_line,
                })
        else:
            body = "\n".join(buffer).strip()
            if len(body) >= min_chunk_chars:
                current["content"] = body
                current["end_line"] = end_line
                chunks.append(current)
        buffer = []

    for idx, line in enumerate(lines):
        line_no = offset + idx + 1  # 사람이 보는 1-based 줄번호

        # 코드펜스 상태 토글 — 코드블록 안에서는 heading 판별을 건너뛴다
        if CODE_FENCE_RE.match(line.strip()):
            in_code_fence = not in_code_fence
            buffer.append(line)
            continue

        m = None if in_code_fence else HEADING_RE.match(line)
        if m:
            # 새 heading을 만났으므로 직전까지 모은 본문을 청크로 확정
            flush(end_line=line_no - 1)

            level = len(m.group(1))          # '#' 개수 = heading 레벨
            title = m.group(2).strip()

            # 스택 정리: 지금 레벨보다 깊거나 같은 heading은 더 이상 상위가 아님
            while heading_stack and heading_stack[-1][1] >= level:
                heading_stack.pop()

            current = {
                "heading": title,
                "level": level,
                # 상위 경로 + 자기 자신 → ["3. 핵심 기능", "F1. 자동 가계부"]
                "heading_path": [h for h, _ in heading_stack] + [title],
                "content": "",
                "start_line": line_no,
                "end_line": line_no,
            }
            heading_stack.append((title, level))
        else:
            buffer.append(line)

    # 파일 끝: 마지막 청크 확정
    flush(end_line=offset + len(lines))

    # frontmatter가 있었다면 모든 청크에 참고 정보로 붙여준다
    if frontmatter:
        for c in chunks:
            c["frontmatter"] = frontmatter

    return chunks


if __name__ == "__main__":
    # 단독 실행 테스트용: python markdown_parser.py <파일.md>
    import json
    import sys

    result = parse_markdown(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
