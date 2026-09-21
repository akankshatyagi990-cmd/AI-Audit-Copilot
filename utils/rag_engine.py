from pathlib import Path
from collections import Counter
import math
import re

from pypdf import PdfReader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_DIRECTORY = (
    PROJECT_ROOT
    / "knowledge_base"
    / "policies"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text):

    text = text.lower()

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text,
    )

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "by",
        "from",
        "as",
        "at",
        "it",
        "its",
        "should",
        "may",
        "can",
        "must",
        "will",
        "has",
        "have",
        "had",
    }

    return [
        word
        for word in words
        if word not in stop_words
        and len(word) > 1
    ]


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_path):

    pages = []

    try:

        reader = PdfReader(
            str(pdf_path)
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            try:

                text = page.extract_text()

            except Exception:

                text = ""

            text = clean_text(
                text
            )

            if text:

                pages.append(
                    {
                        "source":
                            pdf_path.name,

                        "page":
                            page_number,

                        "text":
                            text,
                    }
                )

    except Exception as error:

        print(
            f"Could not read {pdf_path.name}: {error}"
        )

    return pages


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=1200,
    overlap=200,
):

    text = clean_text(
        text
    )

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(
        text
    )

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[
            start:end
        ]


        # Try to end at a sentence boundary
        if end < text_length:

            sentence_end = max(
                chunk.rfind(". "),
                chunk.rfind("? "),
                chunk.rfind("! "),
            )

            if (
                sentence_end
                > chunk_size * 0.60
            ):

                end = (
                    start
                    + sentence_end
                    + 1
                )

                chunk = text[
                    start:end
                ]


        chunk = clean_text(
            chunk
        )

        if chunk:

            chunks.append(
                chunk
            )


        if end >= text_length:
            break


        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


# ============================================================
# KNOWLEDGE BASE LOADING
# ============================================================

def load_policy_chunks(
    policy_directory=POLICY_DIRECTORY,
):

    policy_directory = Path(
        policy_directory
    )


    if not policy_directory.exists():

        raise FileNotFoundError(
            f"Policy directory not found: "
            f"{policy_directory}"
        )


    pdf_files = sorted(
        policy_directory.glob(
            "*.pdf"
        )
    )


    if not pdf_files:

        raise FileNotFoundError(
            "No PDF policy files were found in "
            f"{policy_directory}"
        )


    all_chunks = []

    chunk_id = 1


    for pdf_file in pdf_files:

        pages = extract_pdf_pages(
            pdf_file
        )


        for page_data in pages:

            page_chunks = chunk_text(
                page_data[
                    "text"
                ]
            )


            for page_chunk_index, chunk in enumerate(
                page_chunks,
                start=1,
            ):

                all_chunks.append(
                    {
                        "chunk_id":
                            chunk_id,

                        "source":
                            page_data[
                                "source"
                            ],

                        "page":
                            page_data[
                                "page"
                            ],

                        "page_chunk":
                            page_chunk_index,

                        "text":
                            chunk,

                        "tokens":
                            tokenize(
                                chunk
                            ),
                    }
                )

                chunk_id += 1


    return all_chunks


# ============================================================
# IDF CALCULATION
# ============================================================

def calculate_idf(
    chunks,
):

    document_count = len(
        chunks
    )

    document_frequency = (
        Counter()
    )


    for chunk in chunks:

        unique_tokens = set(
            chunk[
                "tokens"
            ]
        )

        for token in unique_tokens:

            document_frequency[
                token
            ] += 1


    idf = {}


    for token, frequency in (
        document_frequency.items()
    ):

        idf[
            token
        ] = (
            math.log(
                (
                    document_count
                    + 1
                )
                /
                (
                    frequency
                    + 1
                )
            )
            + 1
        )


    return idf


# ============================================================
# RELEVANCE SCORING
# ============================================================

def score_chunk(
    query,
    chunk,
    idf,
):

    query_tokens = tokenize(
        query
    )

    if not query_tokens:
        return 0.0


    chunk_tokens = (
        chunk[
            "tokens"
        ]
    )


    chunk_counter = Counter(
        chunk_tokens
    )


    score = 0.0


    # --------------------------------------------------------
    # TOKEN OVERLAP WITH IDF
    # --------------------------------------------------------

    for token in query_tokens:

        if token in chunk_counter:

            term_frequency = (
                chunk_counter[
                    token
                ]
            )


            score += (
                idf.get(
                    token,
                    1.0,
                )
                *
                (
                    1
                    + math.log(
                        term_frequency
                    )
                )
            )


    # --------------------------------------------------------
    # PHRASE BONUS
    # --------------------------------------------------------

    query_lower = (
        query.lower()
    )

    chunk_lower = (
        chunk[
            "text"
        ].lower()
    )


    important_phrases = [
        "billed amount",
        "allowed amount",
        "billing variance",
        "financial variance",
        "supporting documentation",
        "audit review",
        "audit escalation",
        "units",
        "utilization",
        "risk score",
        "anomaly",
        "provider",
        "procedure",
        "claim",
        "human review",
        "artificial intelligence",
        "generative ai",
        "decision support",
    ]


    for phrase in important_phrases:

        if (
            phrase in query_lower
            and
            phrase in chunk_lower
        ):

            score += 3.0


    # --------------------------------------------------------
    # EXACT QUERY TERM DENSITY BONUS
    # --------------------------------------------------------

    matched_tokens = sum(
        1
        for token in set(
            query_tokens
        )
        if token in chunk_counter
    )


    if query_tokens:

        coverage = (
            matched_tokens
            / len(
                set(
                    query_tokens
                )
            )
        )

        score += (
            coverage
            * 2.0
        )


    return score


# ============================================================
# POLICY RETRIEVAL
# ============================================================

def retrieve_relevant_policy_chunks(
    query,
    top_k=4,
    policy_directory=POLICY_DIRECTORY,
):

    chunks = load_policy_chunks(
        policy_directory
    )


    if not chunks:
        return []


    idf = calculate_idf(
        chunks
    )


    scored_chunks = []


    for chunk in chunks:

        score = score_chunk(
            query,
            chunk,
            idf,
        )


        if score > 0:

            result = chunk.copy()

            result[
                "score"
            ] = score

            scored_chunks.append(
                result
            )


    scored_chunks.sort(
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )


    return scored_chunks[
        :top_k
    ]


# ============================================================
# FORMAT CONTEXT FOR LLM
# ============================================================

def format_policy_context(
    results,
):

    if not results:

        return (
            "No relevant policy context "
            "was retrieved."
        )


    sections = []


    for index, result in enumerate(
        results,
        start=1,
    ):

        section = f"""
SOURCE {index}
Policy: {result['source']}
Page: {result['page']}
Relevance Score: {result['score']:.2f}

Policy Text:
{result['text']}
"""

        sections.append(
            section.strip()
        )


    return "\n\n---\n\n".join(
        sections
    )


# ============================================================
# HUMAN-READABLE SOURCES
# ============================================================

def get_policy_sources(
    results,
):

    sources = []

    seen = set()


    for result in results:

        source_key = (
            result[
                "source"
            ],
            result[
                "page"
            ],
        )


        if source_key not in seen:

            sources.append(
                {
                    "source":
                        result[
                            "source"
                        ],

                    "page":
                        result[
                            "page"
                        ],

                    "score":
                        result[
                            "score"
                        ],
                }
            )

            seen.add(
                source_key
            )


    return sources


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

def get_policy_library_status(
    policy_directory=POLICY_DIRECTORY,
):

    policy_directory = Path(
        policy_directory
    )


    if not policy_directory.exists():

        return {
            "available":
                False,

            "pdf_count":
                0,

            "chunk_count":
                0,

            "files":
                [],
        }


    pdf_files = sorted(
        policy_directory.glob(
            "*.pdf"
        )
    )


    if not pdf_files:

        return {
            "available":
                False,

            "pdf_count":
                0,

            "chunk_count":
                0,

            "files":
                [],
        }


    try:

        chunks = load_policy_chunks(
            policy_directory
        )

    except Exception:

        chunks = []


    return {
        "available":
            True,

        "pdf_count":
            len(
                pdf_files
            ),

        "chunk_count":
            len(
                chunks
            ),

        "files":
            [
                file.name
                for file in pdf_files
            ],
    }


# ============================================================
# SIMPLE TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nAI Audit Copilot - Policy Retrieval Test"
    )

    print(
        "=" * 55
    )


    status = (
        get_policy_library_status()
    )


    print(
        f"\nPolicy PDFs: "
        f"{status['pdf_count']}"
    )

    print(
        f"Policy chunks: "
        f"{status['chunk_count']}"
    )


    if not status[
        "available"
    ]:

        print(
            "\nNo policy PDFs found."
        )

        raise SystemExit


    print(
        "\nLoaded policies:"
    )


    for file_name in status[
        "files"
    ]:

        print(
            f" - {file_name}"
        )


    test_query = (
        "What policy guidance applies when "
        "the billed amount is much higher than "
        "the allowed amount and the claim has "
        "a large financial variance?"
    )


    print(
        "\nTest Query:"
    )

    print(
        test_query
    )


    results = (
        retrieve_relevant_policy_chunks(
            query=test_query,
            top_k=4,
        )
    )


    print(
        "\nRetrieved Policy Context"
    )

    print(
        "=" * 55
    )


    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nResult {index}"
        )

        print(
            f"Policy: "
            f"{result['source']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Score: "
            f"{result['score']:.2f}"
        )

        print(
            "\nText:"
        )

        print(
            result[
                "text"
            ][:700]
        )

        print(
            "\n"
            + "-" * 55
        )