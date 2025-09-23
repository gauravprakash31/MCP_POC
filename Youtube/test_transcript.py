import importlib
import sys

# -----------------------------
# Step 1: Choose which server to test
# -----------------------------
# Default is "server" (server.py). 
# Run in PowerShell as:
#   python testmcp.py server
# or:
#   python testmcp.py server_2

server_module = "server"  # default
if len(sys.argv) > 1:
    server_module = sys.argv[1]

print(f"🔧 Importing tools from: {server_module}.py")

# Dynamically import the chosen server file
server = importlib.import_module(server_module)

# -----------------------------
# Step 2: Run test cases
# -----------------------------
def run_tests():
    print("\n🔍 Testing YouTube MCP tools...\n")

    # Test 1: Search YouTube
    print("📺 Search results for 'Python tutorials':")
    results = server.search_youtube("Python tutorials", max_results=2)
    for r in results:
        print("-", r)

    # Test 2: Channel stats (default or pass channel_id)
    if hasattr(server, "get_channel_stats"):
        print("\n📊 Channel Stats (MrBeast):")
        stats = server.get_channel_stats(getattr(server, "DEFAULT_CHANNEL_ID", "UCX6OQ3DkcsbYNE6H8uQQuVA"))
        print(stats)

    # Test 3: Latest videos
    if hasattr(server, "get_latest_videos_from_channel"):
        print("\n🆕 Latest Videos:")
        latest = server.get_latest_videos_from_channel(getattr(server, "DEFAULT_CHANNEL_ID", "UCX6OQ3DkcsbYNE6H8uQQuVA"), max_results=2)
        for v in latest:
            print("-", v["title"], v["url"])

    # Test 4: Semantic search (if transcripts were indexed)
    if hasattr(server, "semantic_search"):
        print("\n📚 Semantic Search (metadata only):")
        sem = server.semantic_search("gaming challenge", top_k=2, return_full_transcript=False)
        print(sem)

if __name__ == "__main__":
    run_tests()
