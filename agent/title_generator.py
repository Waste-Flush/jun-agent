"""Auto-generate and periodically update session titles from conversation context.

Runs asynchronously after each response is delivered so it never adds latency to
the user-facing reply. Titles are refreshed every N user messages (configurable,
default 10) using a sliding window of recent messages so the title evolves with
the conversation topic.
"""

import json
import logging
import os
import threading
from typing import Callable, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# Callback signature: (task_name, exception) -> None. Used to surface
# auxiliary failures to the user through AIAgent._emit_auxiliary_failure
# so silent-drops (e.g. OpenRouter 402 exhausting the fallback chain)
# become visible instead of piling up as NULL session titles.
FailureCallback = Callable[[str, BaseException], None]
TitleCallback = Callable[[str], None]

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation based on "
    "the recent exchange below. The title should capture the current main topic or intent. "
    "Write the title in the same language the user is writing in. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)

_TITLE_PROMPT_PINNED_LANGUAGE = (
    "Generate a short, descriptive title (3-7 words) for a conversation based on "
    "the recent exchange below. The title should capture the current main topic or intent. "
    "Write the title in {language}. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)

# Title state file — tracks when we last updated, so we don't spam the LLM
_TITLE_STATE_PATH = os.path.expanduser("~/.hermes/data/title_update_counts.json")


def _title_language() -> str:
    """Return configured title language, or empty string to match the user."""
    try:
        from hermes_cli.config import load_config

        return str(
            ((load_config() or {}).get("auxiliary") or {}).get(
                "title_generation", {}
            ).get("language", "")
        ).strip()
    except Exception:
        return ""


def _load_title_state() -> dict:
    """Load title update tracking state from disk."""
    try:
        with open(_TITLE_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_title_state(state: dict) -> None:
    """Persist title update tracking state to disk."""
    os.makedirs(os.path.dirname(_TITLE_STATE_PATH), exist_ok=True)
    with open(_TITLE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


def _mark_updated(session_id: str, total_user_msgs: int) -> None:
    """Record the current user-message count as the last-update point."""
    state = _load_title_state()
    state[session_id] = total_user_msgs
    _save_title_state(state)


def generate_title_from_history(
    conversation_history: list,
    timeout: float = 30.0,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
) -> Optional[str]:
    """Generate a session title from recent conversation history.

    Uses the last N user/assistant exchanges (default: last 6 messages) to
    produce a title reflecting the current topic. Falls back to the full
    exchange if history is too short.
    """
    # Grab last N messages for context
    recent = list(conversation_history[-12:]) if conversation_history else []

    # Build a prompt from the recent messages
    messages = []
    system_language = _title_language()
    prompt = (
        _TITLE_PROMPT_PINNED_LANGUAGE.format(language=system_language)
        if system_language
        else _TITLE_PROMPT
    )
    messages.append({"role": "system", "content": prompt})

    # Format recent exchanges for the LLM
    context_parts = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            context_parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            # Handle multimodal content — extract text parts
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            text = "\n".join(text_parts)
            context_parts.append(f"{role}: {text}")
        else:
            context_parts.append(f"{role}: [non-text content]")

    user_snippet = "\n".join(context_parts)[-2000:] if context_parts else ""

    messages.append({"role": "user", "content": user_snippet})

    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            timeout=timeout,
            main_runtime=main_runtime,
        )
        title = (response.choices[0].message.content or "").strip()
        # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
        title = title.strip('"\'')
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        # Enforce reasonable length
        if len(title) > 80:
            title = title[:77] + "..."
        return title if title else None
    except Exception as e:
        logger.warning("Title generation from history failed: %s", e)
        logger.debug("Title generation traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("title generation", e)
            except Exception:
                logger.debug("Title generation failure_callback raised", exc_info=True)
        return None


def generate_title(
    user_message: str,
    assistant_response: str,
    timeout: float = 30.0,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
) -> Optional[str]:
    """Generate a session title from the first exchange.

    Uses the main runtime's model when available, falling back to the
    auxiliary LLM client (cheapest/fastest available model).
    Returns the title string or None on failure.

    ``failure_callback`` is invoked with ``(task, exception)`` when the
    auxiliary call raises — the caller typically wires this to
    ``AIAgent._emit_auxiliary_failure`` so the user sees a warning instead
    of silently accumulating untitled sessions.
    """
    # Truncate long messages to keep the request small
    user_snippet = user_message[:500] if user_message else ""
    assistant_snippet = assistant_response[:500] if assistant_response else ""

    language = _title_language()
    prompt = _TITLE_PROMPT_PINNED_LANGUAGE.format(language=language) if language else _TITLE_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"User: {user_snippet}\n\nAssistant: {assistant_snippet}"},
    ]

    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            timeout=timeout,
            main_runtime=main_runtime,
        )
        title = (response.choices[0].message.content or "").strip()
        # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
        title = title.strip('"\'')
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        # Enforce reasonable length
        if len(title) > 80:
            title = title[:77] + "..."
        return title if title else None
    except Exception as e:
        # Log at WARNING so this shows up in agent.log without debug mode.
        # Full detail at debug level for operators who need the stack.
        logger.warning("Title generation failed: %s", e)
        logger.debug("Title generation traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("title generation", e)
            except Exception:
                logger.debug("Title generation failure_callback raised", exc_info=True)
        return None


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    interval: int = 10,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
) -> None:
    """Generate or update a session title.

    On the first exchange (no title yet): generates a title from the initial
    user→assistant pair. On subsequent calls, checks if enough new messages have
    accumulated — if so, regenerates the title from recent history.

    Called in a background thread after each response completes.
    """
    if not session_db or not session_id:
        return

    # Count user messages in history
    user_msg_count = sum(
        1 for m in (conversation_history or []) if m.get("role") == "user"
    )

    # Check if title already exists
    try:
        existing = session_db.get_session_title(session_id)
    except Exception:
        existing = None

    if existing:
        # Title exists — check if we should update it
        if not _should_update_title(session_id, user_msg_count, interval):
            return
        # Enough messages accumulated — regenerate from history
        title = generate_title_from_history(
            conversation_history,
            failure_callback=failure_callback,
            main_runtime=main_runtime,
        )
        if not title:
            return
        try:
            session_db.set_session_title(session_id, title)
            logger.debug("Updated session title: %s (session: %s)", title, session_id)
            if title_callback is not None:
                try:
                    title_callback(title)
                except Exception:
                    logger.debug("Auto-title callback failed", exc_info=True)
        except Exception as e:
            logger.debug("Failed to set updated title: %s", e)
    else:
        # No title yet — generate from first exchange
        title = generate_title(
            user_message,
            assistant_response,
            failure_callback=failure_callback,
            main_runtime=main_runtime,
        )
        if not title:
            return
        try:
            session_db.set_session_title(session_id, title)
            logger.debug("Auto-generated session title: %s", title)
            if title_callback is not None:
                try:
                    title_callback(title)
                except Exception:
                    logger.debug("Auto-title callback failed", exc_info=True)
        except Exception as e:
            logger.debug("Failed to set auto-generated title: %s", e)

    # Mark that we've updated this session's title
    _mark_updated(session_id, user_msg_count)


def _should_update_title(session_id: str, current_count: int, interval: int) -> bool:
    """Check whether enough new messages have accumulated since last title update.

    Reads the stored count for this session and compares against current_count.
    Only returns True if at least `interval` new user messages have appeared
    since the last update.
    """
    state = _load_title_state()
    last_count = state.get(session_id, 0)
    if last_count == 0:
        return True
    return (current_count - last_count) >= interval


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    interval: int = 10,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
) -> None:
    """Fire-and-forget title generation/update after each exchange.

    Unlike the old version that only fired on the first exchange, this calls
    ``auto_title_session`` every time. The session DB check + message-count
    gating inside ``auto_title_session`` ensures we only hit the LLM when
    a new title is needed.

    Parameters:
        interval: How many new user messages to wait before regenerating title.
                  Default 10.
    """
    if not session_db or not session_id or not user_message or not assistant_response:
        return

    thread = threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message, assistant_response, conversation_history),
        kwargs={
            "interval": interval,
            "failure_callback": failure_callback,
            "main_runtime": main_runtime,
            "title_callback": title_callback,
        },
        daemon=True,
        name="auto-title",
    )
    thread.start()
