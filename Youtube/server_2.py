import os
import sys
import time
import threading
import googleapiclient.discovery
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# New imports for semantic search
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi

# -----------------------------
# Step 1: Load environment vars
# -----------------------------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("⚠️ YOUTUBE_API_KEY is missing! Please add it to your .env file.")

# -----------------------------
# Step 2: Initialize YouTube API
# -----------------------------
youtube = googleapiclient.discovery.build(
    "youtube", "v3", developerKey=YOUTUBE_API_KEY
)

# -----------------------------
# Step 3: Initialize Vector DB
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # embedding size for MiniLM model
index = faiss.IndexFlatL2(dimension)
video_store = []  # store metadata

# -----------------------------
# Step 4: Create MCP Server
# -----------------------------
mcp = FastMCP("YouTube-MCP")

# -----------------------------
# Step 4.a: Default Channel
# -----------------------------

# Default channel lock 
DEFAULT_CHANNEL_ID = "UC4a-Gbdw7vOaccHmFo40b9g"
# -----------------------------
# Step 5: YouTube API Tools
# -----------------------------
@mcp.tool()
def search_youtube(query: str, max_results: int = 5) -> list[dict]:
    """Search YouTube for videos matching the query (keyword search)."""
    request = youtube.search().list(
        part="snippet",
        q=query,
        maxResults=max_results,
        type="video",
        order="relevance"
    )
    response = request.execute()

    results = []
    for item in response.get("items", []):
        title = item["snippet"]["title"]
        video_id = item["id"]["videoId"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        channel = item["snippet"]["channelTitle"]
        published = item["snippet"]["publishedAt"]
        description = item["snippet"]["description"]
        
        results.append({
            "title": title,
            "video_id": video_id,
            "url": url,
            "channel": channel,
            "published": published,
            "description": description
        })

    print(f"Served keyword search: {query}", file=sys.stderr, flush=True)
    return results


@mcp.tool()
def get_channel_stats(channel_id: str = DEFAULT_CHANNEL_ID) -> dict:
    """Get channel subscriber, view, and video count stats."""
    request = youtube.channels().list(
        part="statistics",
        id=channel_id
    )
    response = request.execute()

    if "items" not in response or not response["items"]:
        return {"error": "Channel not found"}

    stats = response["items"][0]["statistics"]
    print(f"Fetched channel stats for {channel_id}", file=sys.stderr, flush=True)
    return {
        "subscribers": stats.get("subscriberCount"),
        "views": stats.get("viewCount"),
        "videos": stats.get("videoCount")
    }


@mcp.tool()
def get_channel_id_by_username(username: str) -> dict:
    """Get channel ID from channel username/handle."""
    try:
        # Try searching for the channel
        request = youtube.search().list(
            part="snippet",
            q=username,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        
        if response.get("items"):
            channel_id = response["items"][0]["snippet"]["channelId"]
            channel_title = response["items"][0]["snippet"]["title"]
            return {"channel_id": channel_id, "channel_title": channel_title}
        else:
            return {"error": f"Channel '{username}' not found"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_latest_videos_from_channel(channel_id: str = DEFAULT_CHANNEL_ID, max_results: int = 5) -> list[dict]:
    """Get the latest videos from a specific channel."""
    try:
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            type="video",
            order="date",
            maxResults=max_results
        )
        response = request.execute()

        results = []
        for item in response.get("items", []):
            video_data = {
                "title": item["snippet"]["title"],
                "video_id": item["id"]["videoId"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "published": item["snippet"]["publishedAt"],
                "description": item["snippet"]["description"]
            }
            results.append(video_data)

        print(f"Fetched latest videos for channel {channel_id}", file=sys.stderr, flush=True)
        return results
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_video_details(video_id: str) -> dict:
    """Get detailed information about a video including title, description, and stats."""
    try:
        # Get video details
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()

        if "items" not in response or not response["items"]:
            return {"error": "Video not found"}

        item = response["items"][0]
        snippet = item["snippet"]
        stats = item["statistics"]

        return {
            "title": snippet["title"],
            "description": snippet["description"],
            "channel": snippet["channelTitle"],
            "published": snippet["publishedAt"],
            "views": stats.get("viewCount", "0"),
            "likes": stats.get("likeCount", "0"),
            "comments": stats.get("commentCount", "0"),
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_video_stats(video_id: str) -> dict:
    """Fetch view count, like count, and comment count for a specific video."""
    try:
        request = youtube.videos().list(
            part="statistics",
            id=video_id
        )
        response = request.execute()

        if "items" not in response or not response["items"]:
            return {"error": "Video not found"}

        stats = response["items"][0]["statistics"]

        return {
            "views": stats.get("viewCount"),
            "likes": stats.get("likeCount"),
            "comments": stats.get("commentCount")
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def compare_video_stats(video_ids: list[str]) -> list[dict]:
    """Compare stats across multiple videos to find which has most views/likes/comments."""
    results = []
    
    for video_id in video_ids:
        try:
            request = youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            )
            response = request.execute()

            if "items" in response and response["items"]:
                item = response["items"][0]
                snippet = item["snippet"]
                stats = item["statistics"]
                
                results.append({
                    "video_id": video_id,
                    "title": snippet["title"],
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "comments": int(stats.get("commentCount", 0)),
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                })
        except Exception as e:
            results.append({
                "video_id": video_id,
                "error": str(e)
            })
    
    return results


@mcp.tool()
def get_video_transcript(video_id: str) -> dict:
    """Fetch transcript of a YouTube video."""
    try:
        transcript_data = YouTubeTranscriptApi.fetch(video_id)
        transcript_text = " ".join([t["text"] for t in transcript_data])
        
        return {
            "video_id": video_id,
            "transcript": transcript_text,
            "length": len(transcript_text)
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def search_and_analyze_videos(query: str, max_results: int = 10) -> dict:
    """Search for videos and return detailed analysis including stats."""
    try:
        # First search for videos
        search_results = search_youtube(query, max_results)
        
        # Get detailed stats for each video
        analyzed_videos = []
        for video in search_results:
            video_details = get_video_details(video["video_id"])
            if "error" not in video_details:
                analyzed_videos.append(video_details)
        
        # Sort by different metrics
        by_views = sorted(analyzed_videos, key=lambda x: int(x.get("views", 0)), reverse=True)
        by_likes = sorted(analyzed_videos, key=lambda x: int(x.get("likes", 0)), reverse=True)
        by_comments = sorted(analyzed_videos, key=lambda x: int(x.get("comments", 0)), reverse=True)
        
        return {
            "query": query,
            "total_videos": len(analyzed_videos),
            "most_viewed": by_views[0] if by_views else None,
            "most_liked": by_likes[0] if by_likes else None,
            "most_commented": by_comments[0] if by_comments else None,
            "all_videos": analyzed_videos
        }
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Step 6: FAISS Tools
# -----------------------------
@mcp.tool()
def index_video(title: str, transcript: str, video_id: str = None):
    """Index a video transcript/description for semantic search."""
    embedding = model.encode([transcript])
    index.add(np.array(embedding, dtype=np.float32))
    video_store.append({
        "title": title, 
        "transcript": transcript,
        "video_id": video_id
    })
    print(f"Indexed video: {title}", file=sys.stderr, flush=True)
    return {"status": "indexed", "title": title}


@mcp.tool()
def semantic_search(query: str, top_k: int = 3, return_full_transcript: bool = False):
    """Search indexed transcripts/comments by meaning."""
    if len(video_store) == 0:
        return {"error": "No videos indexed yet. Use index_video first."}

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), min(top_k, len(video_store)))
    
    results = []
    for i, idx in enumerate(I[0]):
        if idx < len(video_store):  # Valid index
            video_data = {
                "title": video_store[idx]["title"],
                "video_id": video_store[idx]["video_id"],
                "url": video_store[idx].get("url"),
                # Convert FAISS distance → % similarity
                "similarity_score": round((1 / (1 + D[0][i])) * 100, 2)
            }
            if return_full_transcript:
                video_data["transcript"] = video_store[idx]["transcript"]
            results.append(video_data)
    
    print(f"Semantic search performed for: {query} | full_transcript={return_full_transcript}", file=sys.stderr, flush=True)
    return results


@mcp.tool()
def fetch_and_index_transcript(video_id: str):
    """Fetch transcript of a YouTube video (if available) and index it into FAISS."""
    try:
        # Get video details first
        video_details = get_video_details(video_id)
        if "error" in video_details:
            return video_details
        
        # Get transcript
        transcript_data = YouTubeTranscriptApi.fetch(video_id)
        transcript_text = " ".join([t["text"] for t in transcript_data])

        # Create embedding & store
        embedding = model.encode([transcript_text])
        index.add(np.array(embedding, dtype=np.float32))
        video_store.append({
            "title": video_details["title"], 
            "transcript": transcript_text,
            "video_id": video_id,
            "url": video_details["url"]
        })

        print(f"Transcript fetched and indexed for video: {video_id}", file=sys.stderr, flush=True)

        return {
            "status": "indexed", 
            "video_id": video_id, 
            "title": video_details["title"],
            "transcript_length": len(transcript_text)
        }

    except Exception as e:
        error_msg = str(e)
        if "Subtitles are disabled" in error_msg:
            return {"error": f"Transcript not available for {video_id}: Subtitles disabled by uploader"}
        elif "No transcripts found" in error_msg:
            return {"error": f"Transcript not available for {video_id}: No captions found"}
        else:
            return {"error": f"Transcript not available for {video_id}: {error_msg}"}


@mcp.tool()
def bulk_index_channel_videos(channel_id: str = DEFAULT_CHANNEL_ID, max_videos: int = 10):
    """Fetch and index transcripts for multiple videos from a channel."""
    try:
        # Get latest videos
        videos = get_latest_videos_from_channel(channel_id, max_videos)
        if isinstance(videos, dict) and "error" in videos:
            return videos
        
        indexed_count = 0
        failed_count = 0
        
        for video in videos:
            result = fetch_and_index_transcript(video["video_id"])
            if "error" in result:
                failed_count += 1
                print(f"Failed to index {video['title']}: {result['error']}", file=sys.stderr)
            else:
                indexed_count += 1
        
        return {
            "status": "completed",
            "indexed": indexed_count,
            "failed": failed_count,
            "total_attempted": len(videos)
        }
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Step 7: Run MCP Server
# -----------------------------
def heartbeat():
    while True:
        print("Server alive, waiting for requests...", file=sys.stderr, flush=True)
        time.sleep(30)

def main():
    print("YouTube MCP Server started", file=sys.stderr, flush=True)
    print("Available tools:", file=sys.stderr, flush=True)
    print("- search_youtube", file=sys.stderr, flush=True)
    print("- get_channel_stats", file=sys.stderr, flush=True)
    print("- get_channel_id_by_username", file=sys.stderr, flush=True)
    print("- get_latest_videos_from_channel", file=sys.stderr, flush=True)
    print("- get_video_details", file=sys.stderr, flush=True)
    print("- get_video_stats", file=sys.stderr, flush=True)
    print("- compare_video_stats", file=sys.stderr, flush=True)
    print("- get_video_transcript", file=sys.stderr, flush=True)
    print("- search_and_analyze_videos", file=sys.stderr, flush=True)
    print("- semantic_search", file=sys.stderr, flush=True)
    print("- fetch_and_index_transcript", file=sys.stderr, flush=True)
    print("- bulk_index_channel_videos", file=sys.stderr, flush=True)
    print("Press CTRL+C to stop the server.\n", file=sys.stderr, flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    mcp.run()

if __name__ == "__main__":
    main()