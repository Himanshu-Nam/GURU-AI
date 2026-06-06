import threading

memory_store = []
lock = threading.Lock()

def add_message(role, content):
    with lock:
        memory_store.append({
            "role": role,
            "content": content
        })

def get_history(limit=6):
    with lock:
        recent = memory_store[-limit:]

    return "\n".join(
        f"{m['role']}: {m['content']}" for m in recent
    )