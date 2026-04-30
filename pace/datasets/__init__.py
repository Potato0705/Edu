"""Dataset loaders for PACE-AES (ASAP / ASAP++ / TOEFL11).

All loaders return records of the form
    {"essay_id": str, "essay_text": str, "domain1_score": int, "meta": dict}
so downstream code (including SimpleVectorStore.search) is unchanged.
"""
