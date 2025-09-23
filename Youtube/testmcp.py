from server_2 import (
    search_youtube,
    get_channel_stats,
    get_channel_id_by_username,
    get_latest_videos_from_channel,
    get_video_details,
    get_video_stats,
    compare_video_stats,
    get_video_transcript,
    search_and_analyze_videos,
    semantic_search,
    fetch_and_index_transcript,
    bulk_index_channel_videos,
)

def run_tests():
    print("\n🔎 YouTube Search:")
    results = search_youtube("Python tutorials", max_results=3)
    for r in results:
        print(f"- {r['title']} ({r['url']})")

    print("\n📊 Channel Stats:")
    channel = get_channel_id_by_username("freeCodeCamp.org")
    if "channel_id" in channel:
        stats = get_channel_stats(channel["channel_id"])
        print(f"{channel['channel_title']} → {stats}")
    else:
        print(channel)

    print("\n🎥 Latest Videos from Channel:")
    if "channel_id" in channel:
        latest = get_latest_videos_from_channel(channel["channel_id"], max_results=2)
        for v in latest:
            print(f"- {v['title']} ({v['url']})")

    print("\n📊 Video Stats:")
    video_stats = get_video_stats("LXb3EKWsInQ")  # sample video ID
    print(video_stats)

    print("\n📝 Transcript Fetch + Index:")
    transcript = fetch_and_index_transcript("LXb3EKWsInQ")  # sample video ID
    print(transcript)

    print("\n📚 Semantic Search (metadata only):")
    sem_results = semantic_search("learn python basics", top_k=2)
    print(sem_results)

    print("\n📊 Compare Video Stats:")
    compare_results = compare_video_stats(["LXb3EKWsInQ", "_uQrJ0TkZlc"])
    for r in compare_results:
        print(r)

    print("\n🔎 Search + Analyze:")
    analyze_results = search_and_analyze_videos("Python tutorial", max_results=2)
    print(f"Most viewed: {analyze_results.get('most_viewed')}")
    print(f"Most liked: {analyze_results.get('most_liked')}")
    print(f"Most commented: {analyze_results.get('most_commented')}")

if __name__ == "__main__":
    run_tests()
