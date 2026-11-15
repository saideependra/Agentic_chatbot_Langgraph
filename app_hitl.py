import os
import tempfile
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command

from agentic_chatbot_hitl_backend import (
    chatbot,
    get_all_threads,
    get_document_name,
    ingest_rag_document,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_message_text(content):
    """Convert LangChain message content into displayable text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts)

    return str(content)


def get_chat_title(thread_id):
    """Title is cached so the sidebar does not hit the database on every rerun."""
    cache = st.session_state["title_cache"]

    if thread_id in cache:
        return cache[thread_id]

    for message in load_conversation(thread_id):
        if isinstance(message, HumanMessage):
            title = get_message_text(message.content).strip()
            if title:
                title = title[:35] + "..." if len(title) > 35 else title
                cache[thread_id] = title
                return title

    return "New Chat"


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


# ============================================================
# HITL help function
# ===========================================================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:

        # Display the resumed response
        with st.chat_message("assistant"):

            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...",
                    expanded=True
                )
            }

            def resumed_ai_only_stream():

                # Resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # Update tool execution status
                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    # Stream only assistant-generated text
                    if isinstance(
                        message_chunk,
                        AIMessage
                    ):

                        text = get_message_text(message_chunk.content)
                        if text:
                            yield text

            # Display the streamed final answer
            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            # Check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                # No more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        # Rerun so the response appears in normal chat order
        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )




# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Agentic Chatbot with LangGraph")


# ============================================================
# SESSION STATE
# ============================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "title_cache" not in st.session_state:
    st.session_state["title_cache"] = {}

if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

add_thread(st.session_state["thread_id"])


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💬 My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

current_pdf = get_document_name(st.session_state["thread_id"])
if current_pdf:
    st.sidebar.success(f"📄 PDF in this chat:\n\n{current_pdf}")
else:
    st.sidebar.caption("No PDF in this chat. Attach one in the input box.")

st.sidebar.divider()

for thread_id in st.session_state["chat_threads"][::-1]:
    chat_title = get_chat_title(thread_id)
    is_current = thread_id == st.session_state["thread_id"]
    button_label = f"📍 {chat_title}" if is_current else f"💬 {chat_title}"

    if st.sidebar.button(button_label, key=f"chat_{thread_id}", use_container_width=True):
        st.session_state["thread_id"] = thread_id

        temp_messages = []
        for message in load_conversation(thread_id):
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            content = get_message_text(message.content)
            if content.strip():
                temp_messages.append({"role": role, "content": content})

        st.session_state["message_history"] = temp_messages
        st.rerun()


# ============================================================
# SHOW MESSAGE FROM THE LAST PDF UPLOAD (survives st.rerun)
# ============================================================

flash = st.session_state.pop("flash", None)
if flash:
    kind, text = flash
    if kind == "success":
        st.success(text)
    else:
        st.error(text)


# ============================================================
# SHOW CHAT HISTORY
# ============================================================

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# HUMAN APPROVAL (shown when the graph is paused)
# ============================================================

# Read the saved checkpoint, so the approval box also survives refresh / chat switching
sync_pending_interrupt(st.session_state["thread_id"])

pending = st.session_state.get("pending_hitl")
approval_waiting = pending is not None

if approval_waiting:
    st.warning(f"⚠️ Approval needed: {pending['prompt']}")
    approve_col, reject_col, _ = st.columns([1, 1, 4])

    if approve_col.button("✅ Approve", key="hitl_yes", use_container_width=True):
        resume_hitl_execution("yes")

    if reject_col.button("❌ Reject", key="hitl_no", use_container_width=True):
        resume_hitl_execution("no")


# ============================================================
# CHAT INPUT WITH PDF UPLOAD
# ============================================================

submission = st.chat_input(
    "Approve or reject the purchase above first..." if approval_waiting
    else "Ask me anything or attach a PDF...",
    accept_file=True,
    file_type=["pdf"],
    disabled=approval_waiting,
)


# ============================================================
# PROCESS SUBMISSION
# ============================================================

if submission:
    user_input = (submission.text or "").strip()
    uploaded_files = submission.files

    # --------------------------------------------------------
    # PDF INGESTION (stored for THIS chat only)
    # --------------------------------------------------------

    if uploaded_files:
        uploaded_pdf = uploaded_files[0]
        temporary_file_path = None

        try:
            pdf_bytes = uploaded_pdf.getvalue()

            if not pdf_bytes:
                raise ValueError("The uploaded PDF is empty.")

            if not pdf_bytes.startswith(b"%PDF-"):
                raise ValueError("The uploaded file is not a valid PDF.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
                temporary_file.write(pdf_bytes)
                temporary_file_path = temporary_file.name

            with st.spinner(f"Processing {uploaded_pdf.name}... (first time can take a minute)"):
                info = ingest_rag_document(
                    temporary_file_path,
                    thread_id=st.session_state["thread_id"],
                    file_name=uploaded_pdf.name,
                )

            st.session_state["flash"] = (
                "success",
                f"✅ {uploaded_pdf.name} is ready: {info['pages']} pages, "
                f"{info['chunks']} searchable chunks.",
            )

        except Exception as error:
            st.session_state["flash"] = ("error", f"❌ PDF processing failed: {error}")

        finally:
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)

    # PDF only, no text: refresh so the sidebar and message show up
    if not user_input:
        st.rerun()

    # --------------------------------------------------------
    # CHAT MESSAGE
    # --------------------------------------------------------

    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None, "tools_used": []}

        def show_status(label, state="running"):
            if status_holder["box"] is None:
                status_holder["box"] = st.status(label, expanded=True)
            else:
                status_holder["box"].update(label=label, state=state, expanded=True)

        def ai_only_stream():
            try:
                for message_chunk, _metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    # The model decided to call a tool (shown BEFORE it runs)
                    if isinstance(message_chunk, AIMessageChunk) and message_chunk.tool_call_chunks:
                        for call in message_chunk.tool_call_chunks:
                            name = call.get("name")
                            if name:
                                status_holder["tools_used"].append(name)
                                show_status(f"🔧 Using {name}...")

                    # The tool finished
                    elif isinstance(message_chunk, ToolMessage):
                        name = getattr(message_chunk, "name", "tool")
                        show_status(f"⏳ {name} finished, writing answer...")

                    # Assistant text
                    if isinstance(message_chunk, AIMessage):
                        content = get_message_text(message_chunk.content)
                        if content:
                            yield content

            except Exception as error:
                yield f"\n\n⚠️ Something went wrong: {error}"

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            used = ", ".join(dict.fromkeys(status_holder["tools_used"])) or "tools"
            status_holder["box"].update(
                label=f"✅ Used: {used}", state="complete", expanded=False
            )

    if ai_message is None:
        ai_message = ""

    if ai_message.strip():
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    st.rerun()