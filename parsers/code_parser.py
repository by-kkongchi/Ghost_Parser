"""
② TS/TSX Parser — 개발 코드 파일 파서
====================================================

[역할]
  업로드된 TypeScript / TSX 코드 파일을 "최상위 선언 단위"로 잘라서
  청크 리스트로 만든다.

[자르는 단위 = 최상위 선언(top-level declaration)]
  - import 묶음
  - function 선언 / const 화살표 함수
  - class / interface / type / enum
  - export default ...
  코드 파일을 통째로 임베딩하면 검색이 뭉개지고,
  줄 수로 기계적으로 자르면 함수가 중간에 잘린다.
  → "선언 하나 = 의미 하나"가 코드에서의 자연스러운 의미 단위다.

[파싱 방식: 브레이스 깊이 추적(heuristic scanner)]
  진짜 AST 파서(tree-sitter, tsc)는 네이티브 의존성/노드 런타임이 필요해서
  해커톤 파이썬 백엔드에 붙이기 무겁다.
  대신 문자열/템플릿 리터럴/주석을 건너뛰면서 중괄호 { } 깊이를 추적해
  "깊이 0에서 시작하는 선언"의 시작~끝 줄을 찾는다.
  → 의존성 0으로 충분히 정확하게 동작한다. (극단적인 문법은 못 볼 수 있음)

[출력 형태] (노멀라이저가 받아가는 중간 포맷)
  {
      "name": "useAuth",              # 선언 이름
      "kind": "function",             # function|class|interface|type|enum|component|import|other
      "exported": True,               # export 여부
      "content": "코드 원문...",
      "start_line": 10,
      "end_line": 42,
      "imports": [...],               # (파일 단위 정보) 이 파일이 import 하는 모듈
      "hooks_used": ["useState"],     # React hook 사용 목록 (컴포넌트 분석용)
  }
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) 최상위 선언의 "시작"을 알아보는 정규식들
#    줄 맨 앞(공백 허용 X — 최상위이므로 들여쓰기 없음)에서 매칭한다.
# ---------------------------------------------------------------------------
DECL_PATTERNS: list[tuple[str, re.Pattern]] = [
    # export async function Foo(...)  /  function Foo(...)
    ("function", re.compile(
        r"^(?P<export>export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)")),
    # export const Foo = (...) => / const Foo = async () =>  (화살표 함수·상수 모두)
    ("const", re.compile(
        r"^(?P<export>export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)")),
    # export class Foo / abstract class Foo
    ("class", re.compile(
        r"^(?P<export>export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)")),
    # export interface Foo
    ("interface", re.compile(
        r"^(?P<export>export\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)")),
    # export type Foo = ...
    ("type", re.compile(
        r"^(?P<export>export\s+)?type\s+(?P<name>[A-Za-z_$][\w$]*)")),
    # export enum Foo
    ("enum", re.compile(
        r"^(?P<export>export\s+)?(?:const\s+)?enum\s+(?P<name>[A-Za-z_$][\w$]*)")),
]

# import ... from '...' — 파일이 무엇에 의존하는지는 파일 단위 메타데이터로 수집
IMPORT_RE = re.compile(r"""^import\s+(?:[\s\S]*?\s+from\s+)?['"](?P<module>[^'"]+)['"]""")

# React hook 호출 (useXxx( ) — 컴포넌트가 어떤 훅을 쓰는지 분석용
HOOK_RE = re.compile(r"\buse[A-Z]\w*(?=\()")

# JSX 태그 존재 여부 — "이 선언은 React 컴포넌트다"라고 판단하는 근거
JSX_RE = re.compile(r"<[A-Za-z][\w.]*(\s|/?>)")


def _scan_line_depths(source: str) -> list[int]:
    """
    소스 전체를 한 글자씩 훑으며 "각 줄이 끝난 시점의 중괄호 깊이"를 계산한다.

    핵심: 문자열('...', "...", `...`), 주석(//, /* */), 정규식 리터럴 안의
    중괄호는 세면 안 된다. 그래서 상태 머신으로 현재 어떤 문맥인지 추적한다.

    반환: depths[i] = i번째 줄이 끝났을 때의 { } 중첩 깊이
          → depth 0에서 0으로 끝나는 구간이 "하나의 최상위 블록"
    """
    depths: list[int] = []
    depth = 0
    state = "code"      # code | line_comment | block_comment | s_quote | d_quote | template
    prev = ""           # 직전 문자 (이스케이프 \" 판별용)

    for line in source.splitlines():
        i = 0
        while i < len(line):
            ch = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""

            if state == "code":
                if ch == "/" and nxt == "/":
                    state = "line_comment"
                    break  # 줄 주석은 그 줄 끝까지 무시
                elif ch == "/" and nxt == "*":
                    state = "block_comment"
                    i += 1
                elif ch == "'":
                    state = "s_quote"
                elif ch == '"':
                    state = "d_quote"
                elif ch == "`":
                    state = "template"
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth = max(0, depth - 1)  # 문법 오류 파일도 죽지 않게 방어
            elif state == "block_comment":
                if ch == "*" and nxt == "/":
                    state = "code"
                    i += 1
            elif state == "s_quote":
                if ch == "'" and prev != "\\":
                    state = "code"
            elif state == "d_quote":
                if ch == '"' and prev != "\\":
                    state = "code"
            elif state == "template":
                # 템플릿 리터럴은 여러 줄 가능. ${ } 내부 중첩까지는 안 세고
                # 백틱이 닫힐 때까지 통째로 무시한다 (실용적 타협).
                if ch == "`" and prev != "\\":
                    state = "code"

            prev = ch
            i += 1

        # 줄이 끝나면 줄 주석 상태는 해제된다 (블록 주석/템플릿은 유지)
        if state == "line_comment":
            state = "code"
        prev = ""
        depths.append(depth)

    return depths


def parse_ts(file_path: str | Path) -> list[dict]:
    """
    TS/TSX 파일을 최상위 선언 단위 청크 리스트로 파싱한다.

    Args:
        file_path: .ts / .tsx 파일 경로

    Returns:
        청크 dict 리스트 (모듈 상단 docstring의 출력 형태 참고)
    """
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    depths = _scan_line_depths(source)
    is_tsx = path.suffix == ".tsx"

    # ---- 파일 단위 메타데이터: import 목록 ------------------------------
    imports: list[str] = []
    for line in lines:
        m = IMPORT_RE.match(line)
        if m:
            imports.append(m.group("module"))

    chunks: list[dict] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        # 최상위(직전 줄 기준 depth 0)에서만 선언 시작을 인정한다.
        at_top = (i == 0) or (depths[i - 1] == 0)

        if not at_top or not line.strip() or line.startswith(("//", "/*", " ", "\t")):
            i += 1
            continue

        # import 문은 선언 청크로 만들지 않고 메타데이터로만 쓴다
        if line.startswith("import"):
            i += 1
            continue

        matched = None
        for kind, pattern in DECL_PATTERNS:
            m = pattern.match(line)
            if m:
                matched = (kind, m)
                break

        if not matched:
            i += 1
            continue

        kind, m = matched
        name = m.group("name")
        exported = bool(m.group("export"))

        # ---- 선언의 끝 찾기 ------------------------------------------
        # 이 줄부터 시작해서 "depth가 다시 0으로 돌아오는 줄"까지가 한 블록.
        # 단, `type Foo = string;` 처럼 중괄호 없이 한 줄로 끝나는 선언은
        # 깊이 변화가 없으므로 세미콜론 기준으로 끝을 잡는다.
        end = i
        opened = False  # 블록( { )이 실제로 열렸는지
        for j in range(i, n):
            if depths[j] > 0:
                opened = True
            if opened and depths[j] == 0:
                end = j
                break
            if not opened and lines[j].rstrip().endswith(";"):
                end = j
                break
        else:
            end = n - 1  # 파일 끝까지 (닫는 괄호가 없는 비정상 파일 방어)

        body = "\n".join(lines[i:end + 1])

        # ---- React 컴포넌트 판별 --------------------------------------
        # 근거: (1) .tsx 파일이고 (2) 이름이 대문자로 시작하며 (3) JSX가 들어있다
        if is_tsx and name[0].isupper() and JSX_RE.search(body):
            kind = "component"

        chunks.append({
            "name": name,
            "kind": kind,
            "exported": exported,
            "content": body,
            "start_line": i + 1,
            "end_line": end + 1,
            "imports": imports,                       # 파일 공통 정보
            "hooks_used": sorted(set(HOOK_RE.findall(body))),
        })

        i = end + 1

    # ---- 아무 선언도 못 찾은 경우 --------------------------------------
    # (설정 파일, 스크립트성 코드 등) 파일 전체를 청크 1개로 만들어
    # 정보 유실을 막는다. "못 자르면 통째로"가 안전한 기본값.
    if not chunks and source.strip():
        chunks.append({
            "name": path.stem,
            "kind": "other",
            "exported": False,
            "content": source,
            "start_line": 1,
            "end_line": n,
            "imports": imports,
            "hooks_used": sorted(set(HOOK_RE.findall(source))),
        })

    return chunks


if __name__ == "__main__":
    # 단독 실행 테스트용: python code_parser.py <파일.tsx>
    import json
    import sys

    result = parse_ts(sys.argv[1])
    # content는 길어서 미리보기만 출력
    for c in result:
        preview = {**c, "content": c["content"][:80] + "..."}
        print(json.dumps(preview, ensure_ascii=False))
