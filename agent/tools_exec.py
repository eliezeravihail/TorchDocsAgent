"""One tool-execution step, shared by the manual loop and the LangGraph twin.

Both drivers (agent/loop.py, agent/graph.py) accumulate the same state —
sections, referrals, seen urls, transcript — and differ only in control flow.
Keeping the actual tool dispatch (and the seed search) here means the two
cannot drift: they get the same dedup, the same observation strings, and the
same forced first search. Budget accounting and the terminal answer/step
decision stay with each driver.
"""

from __future__ import annotations


def do_search(
    query: str, library, kind=None, *, sections: list, seen_urls: set, transcript: list
) -> None:
    """Run search_docs, dedup its sections into the accumulators, log the titles."""
    from agent.tools import search_docs

    result = search_docs(query, library, kind)
    for s in result["sections"]:
        key = s["url"] + s.get("anchor", "")
        if key not in seen_urls:
            seen_urls.add(key)
            sections.append(s)
    transcript.append(f"search_docs({result['query']!r}) → {result['titles'][:5]}")


def execute_tool(
    name: str,
    action: dict,
    question: str,
    *,
    sections: list,
    referrals: list,
    seen_urls: set,
    transcript: list,
) -> None:
    """Execute one planner action (search_docs / read_page / ask_source).

    Mutates the accumulators in place. The caller has already checked and
    decremented the budget for `name`; unknown actions are no-ops here.
    """
    from agent.tools import ask_source, read_page

    if name == "search_docs":
        do_search(
            action.get("query", question),
            action.get("library"),
            action.get("kind"),
            sections=sections,
            seen_urls=seen_urls,
            transcript=transcript,
        )
    elif name == "read_page":
        page = read_page(action.get("url", ""))
        if "content" in page:
            sections.append(
                {
                    "url": page["url"],
                    "anchor": "",
                    "heading_path": page.get("title", ""),
                    "content": page["content"],
                }
            )
            transcript.append(f"read_page({page['url']}) → full page added")
        else:
            transcript.append(f"read_page → {page.get('outline') or page.get('error')}")
    elif name == "ask_source":
        src = ask_source(action.get("question", question))
        referrals.extend(src["referrals"])
        transcript.append(f"ask_source → {len(src['referrals'])} referral links")
