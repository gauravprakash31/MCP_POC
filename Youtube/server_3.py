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
from youtube_transcript_api.formatters import TextFormatter

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
DEFAULT_CHANNEL_ID = "UCX6OQ3DkcsbYNE6H8uQQuVA"
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
def check_transcript_availability(video_id: str = None) -> dict:
    """Check what transcripts are available for a video without fetching them."""
    if not video_id:
        return {"error": "You must provide a video_id to check transcript availability"}

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        available_transcripts = []
        for transcript in transcript_list:
            transcript_info = {
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "is_translatable": transcript.is_translatable
            }
            available_transcripts.append(transcript_info)

        return {
            "video_id": video_id,
            "has_transcripts": len(available_transcripts) > 0,
            "available_transcripts": available_transcripts,
            "count": len(available_transcripts)
        }
    except Exception as e:
        return {
            "video_id": video_id,
            "has_transcripts": False,
            "error": str(e)
        }


@mcp.tool()
def get_video_transcript(video_id: str = None, languages: list[str] = None) -> dict:
    """
    Fetch transcript of a YouTube video with better error handling.
    Claude should first call check_transcript_availability(video_id),
    and only call this tool if transcripts exist.
    """
    if not video_id:
        return {"error": "You must provide a video_id (e.g., 'rfscVS0vtbw')"}

    try:
        if languages is None:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
                transcript_data = transcript.fetch()
            except:
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                    transcript_data = transcript.fetch()
                except:
                    available_transcripts = list(transcript_list)
                    if available_transcripts:
                        transcript_data = available_transcripts[0].fetch()
                    else:
                        return {"error": f"No transcripts available for video {video_id}"}
        else:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript_data)

        return {
            "video_id": video_id,
            "transcript": transcript_text,
            "length": len(transcript_text),
            "segments": len(transcript_data)
        }
    except Exception as e:
        error_msg = str(e)
        if "TranscriptsDisabled" in error_msg:
            return {"error": f"Transcripts are disabled for video {video_id}"}
        elif "NoTranscriptFound" in error_msg:
            return {"error": f"No transcript found for video {video_id} in requested languages"}
        elif "VideoUnavailable" in error_msg:
            return {"error": f"Video {video_id} is unavailable"}
        else:
            return {"error": f"Failed to get transcript for {video_id}: {error_msg}"}


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
def fetch_and_index_transcript(video_id: str, languages: list[str] = None):
    """Fetch transcript of a YouTube video (if available) and index it into FAISS."""
    try:
        # Get video details first
        video_details = get_video_details(video_id)
        if "error" in video_details:
            return video_details
        
        # Get transcript with improved handling
        transcript_result = get_video_transcript(video_id, languages)
        if "error" in transcript_result:
            return transcript_result
        
        transcript_text = transcript_result["transcript"]

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
            "transcript_length": len(transcript_text),
            "segments": transcript_result.get("segments", 0)
        }

    except Exception as e:
        return {"error": f"Failed to fetch and index transcript for {video_id}: {str(e)}"}


@mcp.tool()
def bulk_index_channel_videos(channel_id: str = DEFAULT_CHANNEL_ID, max_videos: int = 10):
    """Fetch and index transcripts for multiple videos from a channel with detailed reporting."""
    try:
        # Get latest videos
        videos = get_latest_videos_from_channel(channel_id, max_videos)
        if isinstance(videos, dict) and "error" in videos:
            return videos
        
        indexed_count = 0
        failed_count = 0
        results = []
        
        for video in videos:
            # First check if transcript is available
            availability = check_transcript_availability(video["video_id"])
            
            if availability.get("has_transcripts", False):
                # Try to fetch and index
                result = fetch_and_index_transcript(video["video_id"])
                if "error" in result:
                    failed_count += 1
                    results.append({
                        "video_id": video["video_id"],
                        "title": video["title"],
                        "status": "failed",
                        "error": result["error"]
                    })
                else:
                    indexed_count += 1
                    results.append({
                        "video_id": video["video_id"],
                        "title": video["title"],
                        "status": "indexed",
                        "transcript_length": result.get("transcript_length", 0)
                    })
            else:
                failed_count += 1
                results.append({
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "status": "no_transcript",
                    "error": "No transcripts available"
                })
        
        return {
            "status": "completed",
            "indexed": indexed_count,
            "failed": failed_count,
            "total_attempted": len(videos),
            "details": results
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def find_videos_with_transcripts(query: str = None, channel_id: str = None, max_results: int = 20):
    """Search for videos that have transcripts available."""
    try:
        # Get videos to check
        if query:
            videos = search_youtube(query, max_results)
        elif channel_id:
            videos = get_latest_videos_from_channel(channel_id, max_results)
        else:
            videos = get_latest_videos_from_channel(DEFAULT_CHANNEL_ID, max_results)
        
        if isinstance(videos, dict) and "error" in videos:
            return videos
        
        videos_with_transcripts = []
        
        for video in videos:
            availability = check_transcript_availability(video["video_id"])
            if availability.get("has_transcripts", False):
                video_info = video.copy()
                video_info["transcript_info"] = availability["available_transcripts"]
                videos_with_transcripts.append(video_info)
        
        return {
            "total_checked": len(videos),
            "videos_with_transcripts": len(videos_with_transcripts),
            "videos": videos_with_transcripts
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_video_comments(video_id: str, max_results: int = 50) -> list[dict]:
    """Fetch top-level comments for a video."""
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()
    results = []
    for item in response.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        results.append({
            "author": snippet["authorDisplayName"],
            "text": snippet["textDisplay"],
            "likes": snippet["likeCount"],
            "published": snippet["publishedAt"]
        })
    return results
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
    print("- check_transcript_availability", file=sys.stderr, flush=True)
    print("- search_and_analyze_videos", file=sys.stderr, flush=True)
    print("- semantic_search", file=sys.stderr, flush=True)
    print("- fetch_and_index_transcript", file=sys.stderr, flush=True)
    print("- bulk_index_channel_videos", file=sys.stderr, flush=True)
    print("- find_videos_with_transcripts", file=sys.stderr, flush=True)
    print("Press CTRL+C to stop the server.\n", file=sys.stderr, flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    mcp.run()

if __name__ == "__main__":
    main()