"""
evals/test_retrieval.py — Evaluation script for BNS RAG system.

Tests retrieval quality and answer accuracy against 15 ground-truth
question/answer pairs covering both BNS and BNSS.

Run:  python -m evals.test_retrieval

Outputs:
  - Per-question pass/fail with scores
  - Overall retrieval accuracy (did the right section appear in top-5?)
  - Overall answer accuracy (does the answer contain expected keywords?)
"""

import sys
import time
import requests
from dataclasses import dataclass

BACKEND_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Ground truth test cases
# Format:
#   question        : what a user would ask
#   expected_section: section number(s) that MUST appear in retrieved chunks
#   expected_keywords: keywords that MUST appear in the LLM answer
#   source          : "BNS" or "BNSS" (for labeling only)
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    question: str
    expected_sections: list[str]   # e.g. ["Section 101", "Section 103"]
    expected_keywords: list[str]   # case-insensitive substring match in answer
    source: str
    description: str               # human-readable label


TEST_CASES = [
    TestCase(
        question="What is the punishment for murder under BNS?",
        expected_sections=["Section 101", "Section 103"],
        expected_keywords=["death", "imprisonment for life"],
        source="BNS",
        description="Murder punishment",
    ),
    TestCase(
        question="What constitutes culpable homicide?",
        expected_sections=["Section 100"],
        expected_keywords=["death", "intention", "knowledge"],
        source="BNS",
        description="Culpable homicide definition",
    ),
    TestCase(
        question="What is the definition of theft in BNS?",
        expected_sections=["Section 303"],
        expected_keywords=["movable property", "dishonest", "consent"],
        source="BNS",
        description="Theft definition",
    ),
    TestCase(
        question="What are the punishments available under BNS?",
        expected_sections=["Section 4"],
        expected_keywords=["death", "imprisonment", "fine", "community service"],
        source="BNS",
        description="Types of punishments",
    ),
    TestCase(
        question="What is the right of private defence of body?",
        expected_sections=["Section 35", "Section 38"],
        expected_keywords=["private defence", "body", "harm"],
        source="BNS",
        description="Right of private defence",
    ),
    TestCase(
        question="When does the right of private defence of the body extend to causing death?",
        expected_sections=["Section 38"],
        expected_keywords=["death", "grievous hurt", "assault", "rape"],
        source="BNS",
        description="Private defence extending to causing death",
    ),
    TestCase(
        question="What is the punishment for rape?",
        expected_sections=["Section 64"],
        expected_keywords=["imprisonment", "fine"],
        source="BNS",
        description="Rape punishment",
    ),
    TestCase(
        question="What does the BNS say about defamation?",
        expected_sections=["Section 356"],
        expected_keywords=["defamation", "imputation", "reputation"],
        source="BNS",
        description="Defamation",
    ),
    TestCase(
        question="What are the provisions for bail in BNSS?",
        expected_sections=["Section 478", "Section 479", "Section 480"],
        expected_keywords=["bail", "bailable", "court"],
        source="BNSS",
        description="Bail provisions",
    ),
    TestCase(
        question="How is a First Information Report (FIR) filed under BNSS?",
        expected_sections=["Section 173"],
        expected_keywords=["information", "officer", "cognizable"],
        source="BNSS",
        description="FIR filing procedure",
    ),
    TestCase(
        question="What is the procedure for arrest without warrant?",
        expected_sections=["Section 35"],
        expected_keywords=["arrest", "warrant", "police officer"],
        source="BNSS",
        description="Arrest without warrant",
    ),
    TestCase(
        question="What constitutes abetment under BNS?",
        expected_sections=["Section 45", "Section 46"],
        expected_keywords=["abets", "instigates", "conspiracy", "aids"],
        source="BNS",
        description="Abetment definition",
    ),
    TestCase(
        question="What is the punishment for dacoity?",
        expected_sections=["Section 310"],
        expected_keywords=["dacoity", "imprisonment", "life"],
        source="BNS",
        description="Dacoity punishment",
    ),
    TestCase(
        question="What does BNS say about criminal conspiracy?",
        expected_sections=["Section 61"],
        expected_keywords=["conspiracy", "agree", "illegal"],
        source="BNS",
        description="Criminal conspiracy",
    ),
    TestCase(
        question="What is the age of criminal responsibility under BNS?",
        expected_sections=["Section 20", "Section 21"],
        expected_keywords=["seven", "child", "offence"],
        source="BNS",
        description="Age of criminal responsibility",
    ),
]


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def check_retrieval(chunks_used: int, answer: str, citations: str, tc: TestCase) -> bool:
    """Check if any expected section appears in the citations."""
    citations_lower = citations.lower()
    for sec in tc.expected_sections:
        if sec.lower() in citations_lower:
            return True
    return False


def check_answer(answer: str, tc: TestCase) -> tuple[bool, list[str]]:
    """Check if expected keywords appear in the answer."""
    answer_lower = answer.lower()
    missing = [kw for kw in tc.expected_keywords if kw.lower() not in answer_lower]
    return len(missing) == 0, missing


def run_evals():
    print("=" * 65)
    print("BNS RAG — Evaluation Suite")
    print("=" * 65)

    # Check backend
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        if not health.get("index_loaded"):
            print("ERROR: Index not loaded. Run POST /ingest first.")
            sys.exit(1)
        print(f"Backend OK — {health['total_chunks']} chunks indexed\n")
    except Exception as e:
        print(f"ERROR: Cannot reach backend at {BACKEND_URL}: {e}")
        sys.exit(1)

    retrieval_pass = 0
    answer_pass = 0
    total = len(TEST_CASES)
    results = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i:02d}/{total}] {tc.description} ({tc.source})")
        print(f"       Q: {tc.question}")

        start = time.time()
        try:
            resp = requests.post(
                f"{BACKEND_URL}/query",
                json={"question": tc.question, "top_k": 5},
                timeout=30,
            )
            elapsed = time.time() - start

            if resp.status_code != 200:
                print(f"       ❌ HTTP {resp.status_code}: {resp.text}\n")
                results.append({"retrieval": False, "answer": False})
                continue

            data = resp.json()
            answer = data["answer"]
            citations = data["citations"]

            ret_ok = check_retrieval(data["chunks_used"], answer, citations, tc)
            ans_ok, missing_kws = check_answer(answer, tc)

            retrieval_icon = "✅" if ret_ok else "❌"
            answer_icon = "✅" if ans_ok else "⚠️"

            print(f"       Retrieval: {retrieval_icon}  |  Answer: {answer_icon}  |  {elapsed:.1f}s")
            if not ret_ok:
                print(f"       Expected sections: {tc.expected_sections}")
                print(f"       Got citations: {citations[:120]}")
            if not ans_ok:
                print(f"       Missing keywords: {missing_kws}")

            if ret_ok:
                retrieval_pass += 1
            if ans_ok:
                answer_pass += 1

            results.append({"retrieval": ret_ok, "answer": ans_ok})

        except Exception as e:
            print(f"       ❌ Exception: {e}")
            results.append({"retrieval": False, "answer": False})

        print()

    # Summary
    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"Retrieval accuracy : {retrieval_pass}/{total} = {retrieval_pass/total*100:.1f}%")
    print(f"Answer accuracy    : {answer_pass}/{total} = {answer_pass/total*100:.1f}%")
    print()

    if retrieval_pass / total >= 0.80:
        print("✅ Retrieval: PASS (≥80%)")
    else:
        print("❌ Retrieval: FAIL (<80%) — consider tuning chunk size or similarity threshold")

    if answer_pass / total >= 0.70:
        print("✅ Answers:   PASS (≥70%)")
    else:
        print("❌ Answers:   FAIL (<70%) — consider improving prompt or lowering temperature")

    print("=" * 65)
    return retrieval_pass, answer_pass, total


if __name__ == "__main__":
    run_evals()