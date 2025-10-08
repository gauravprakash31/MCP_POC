import os, sys, json, pathlib
from typing import List, Dict, Any
import streamlit as st
import anthropic

# allow imports from ../mcp-gdrive
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent / "mcp-gdrive"))

from drive_core import search_files as drv_search, get_text as drv_get_text, get_metadata as drv_meta

st.set_page_config(page_title="Claude + Drive", page_icon="💬", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "tool_events" not in st.session_state:
    st.session_state["tool_events"] = []

ANTH_MODEL_DEFAULT = "claude-3-5-sonnet-20241022"

anthropic_tools = [
    {
        "name": "gdrive_search",
        "description": "Search Google Drive using Drive v3 q syntax.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "page_size": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50}
            },
            "required": ["query"]
        }
    },
    {
        "name": "gdrive_get_text",
        "description": "Fetch text for a Drive file id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "max_chars": {"type": "integer", "default": 60000}
            },
            "required": ["file_id"]
        }
    }
]

def run_claude_with_tools(history: List[Dict[str,str]], system_prompt: str) -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return "[Paste your ANTHROPIC_API_KEY in the sidebar.]"
    client = anthropic.Anthropic(api_key=key)
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    tool_results: List[Dict[str,Any]] = []
    for _ in range(4):
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", ANTH_MODEL_DEFAULT),
            system=system_prompt,
            max_tokens=1400,
            temperature=0.2,
            tools=anthropic_tools,
            messages=msgs + tool_results
        )
        made_tool_call, out_text = False, []
        for block in resp.content:
            if block.type == "text":
                out_text.append(block.text)
            elif block.type == "tool_use":
                made_tool_call = True
                name, args, tool_id = block.name, (block.input or {}), block.id
                st.session_state["tool_events"].append(f"▶ {name} {args}")
                try:
                    if name == "gdrive_search":
                        res = drv_search(args.get("query",""), page_size=int(args.get("page_size",10)))
                        payload = {"results": res}
                    elif name == "gdrive_get_text":
                        fid = args["file_id"]; mx = int(args.get("max_chars",60000))
                        txt = drv_get_text(fid, max_chars=mx); meta = drv_meta(fid)
                        payload = {"file_id": fid, "meta": meta, "text": txt}
                    else:
                        payload = {"error": f"unknown tool {name}"}
                except Exception as e:
                    payload = {"error": str(e)}
                preview = json.dumps(payload)[:500]
                st.session_state["tool_events"].append(f"◀ result {preview}{' ...' if len(json.dumps(payload))>500 else ''}")
                tool_results.append({
                    "role":"tool","tool_use_id":tool_id,
                    "content":[{"type":"tool_result","content":json.dumps(payload)[:100000]}]
                })
        if not made_tool_call:
            return "".join(out_text).strip() or "[No content]"
    return "[Too many tool rounds without a final answer]"

# sidebar
with st.sidebar:
    st.header("Settings")
    key = st.text_input("ANTHROPIC_API_KEY", type="password", value=os.getenv("ANTHROPIC_API_KEY",""))
    if key: os.environ["ANTHROPIC_API_KEY"] = key
    model = st.text_input("Model", value=os.getenv("ANTHROPIC_MODEL", ANTH_MODEL_DEFAULT))
    if model: os.environ["ANTHROPIC_MODEL"] = model
    st.divider()
    st.subheader("Tool Activity (last 20)")
    st.write("\n".join(st.session_state["tool_events"][-20:]) or "—")

st.title("💬 Claude + Google Drive")
for m in st.session_state["messages"]:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"]=="user" else "🤖"):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask anything. Claude can search & read Drive via tools.")
if user_msg:
    st.session_state["messages"].append({"role":"user","content":user_msg})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_msg)

    system = ("You are Claude inside a custom UI. "
              "Use gdrive_search and gdrive_get_text when helpful. "
              "When quoting, cite the file name.")
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            answer = run_claude_with_tools(st.session_state["messages"], system)
            st.markdown(answer)
    st.session_state["messages"].append({"role":"assistant","content":answer})
