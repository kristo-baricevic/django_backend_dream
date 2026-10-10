"""Per-request HTTP profile adapter; no ORM, signing secret or profile cache.

Use only the authenticated APIClient owned by the current chat request. The
analyzer resolves the saved settings authoritatively; this snapshot makes chat
tool behavior explicit and prevents the former empty-settings default.
"""


async def run_analysis(client, operation, *, content=None, question=None, entries=None):
    if operation not in {"single", "cumulative", "question"}:
        raise ValueError("invalid_analysis_operation")
    settings = await client.get_settings()
    if not isinstance(settings, dict):
        raise ValueError("invalid_settings_response")
    if operation == "single":
        return await client.analyze_dream(content, settings=settings)
    if operation == "cumulative":
        return await client.get_cumulative_analysis_workflow(entries, settings=settings)
    return await client.ask_custom_question_workflow(question, entries, settings=settings)


def format_grounded_text(answer, metadata):
    """Display validated claims/citations directly, without another model call."""
    claims, evidence = metadata.get("claims"), metadata.get("evidence")
    if not isinstance(claims, list) or not claims or not isinstance(evidence, list):
        return answer + "\n\nSource attribution unavailable. Confidence unavailable."
    if " ".join(answer.split()) != " ".join(" ".join(claim["text"] for claim in claims).split()):
        raise ValueError("invalid_display_claim_inventory")
    registry = {item["evidence_id"]: item for item in evidence}
    used = list(dict.fromkeys(key for claim in claims for key in claim["evidence_ids"]))
    if any(key not in registry for key in used):
        raise ValueError("invalid_display_evidence")
    labels = {key: index + 1 for index, key in enumerate(used)}
    lines = [claim["text"] + (" " + " ".join(f"[{labels[key]}]" for key in claim["evidence_ids"]) if claim["evidence_ids"] else "") for claim in claims]
    sources = []
    for key in used:
        item = registry[key]
        if item["kind"] == "user_entry":
            label = f"Saved record {item['entry_id']}; source span {item['start_char']}–{item['end_char']}"
        elif item["kind"] == "user_draft":
            label = "Current unsaved draft"
        else:
            label = str(item.get("source", "Reference"))[:500]
            if item.get("page") is not None:
                label += f"; PDF page index {item['page']}"
        sources.append(f"[{labels[key]}] {label}")
    return "\n\n".join(lines + (["Sources:\n" + "\n".join(sources)] if sources else []) + ["Confidence unavailable. Interpretations are reflective hypotheses."])
