"""Web status page module for displaying bot statistics."""

from flask import Flask, Response
from threading import Thread
from datetime import datetime
import psutil

app = Flask('web_status')

# References to bot objects (set by main module).
_bot = None
_message_history = None
_tfidf_vectorizer = None
_tfidf_matrix_getter = None
_start_time = None
_max_messages = None
_last_rebuild_getter = None
_token_stats_getter = None


def configure(
    bot,
    message_history,
    tfidf_vectorizer,
    tfidf_matrix_getter,
    start_time,
    max_messages,
    last_rebuild_getter=None,
    token_stats_getter=None
):
    """Configure references to bot objects for status display."""
    global _bot, _message_history, _tfidf_vectorizer, _tfidf_matrix_getter
    global _start_time, _max_messages, _last_rebuild_getter, _token_stats_getter

    _bot = bot
    _message_history = message_history
    _tfidf_vectorizer = tfidf_vectorizer
    _tfidf_matrix_getter = tfidf_matrix_getter
    _start_time = start_time
    _max_messages = max_messages
    _last_rebuild_getter = last_rebuild_getter
    _token_stats_getter = token_stats_getter


def _generate_status_text():
    """Generate status text for the web page."""
    lines = []
    lines.append("=" * 50)
    lines.append("SIMON'S BIONIX DISCORD CHATBOT STATUS")
    lines.append("=" * 50)
    lines.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    try:
        if _bot and _bot.user:
            # Bot information.
            lines.append("-" * 50)
            lines.append("BOT INFO")
            lines.append("-" * 50)
            status = "Online" if not _bot.is_closed() else "Offline"
            lines.append(f"Status:          {status}")
            lines.append(f"Bot User:        {_bot.user.name}#{_bot.user.discriminator}")
            lines.append(f"Bot ID:          {_bot.user.id}")

            if _start_time:
                uptime = datetime.now() - _start_time
                uptime_str = str(uptime).split('.')[0]
                lines.append(f"Uptime:          {uptime_str}")

            lines.append(f"Latency:         {_bot.latency * 1000:.1f}ms")
            lines.append("")

            # Memory and storage.
            lines.append("-" * 50)
            lines.append("MEMORY & STORAGE")
            lines.append("-" * 50)

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            lines.append(f"RAM Usage:       {memory_mb:.1f} MB")

            if _message_history is not None:
                total_messages = len(_message_history)
                max_msgs = _max_messages or 1000000
                bot_msg_count = sum(1 for msg in _message_history if msg.get('is_bot', False))
                user_msg_count = total_messages - bot_msg_count
                storage_pct = (total_messages / max_msgs * 100) if max_msgs > 0 else 0

                lines.append(f"Messages Stored: {total_messages:,} / {max_msgs:,}")
                lines.append(f"  - User msgs:   {user_msg_count:,}")
                lines.append(f"  - Bot msgs:    {bot_msg_count:,}")
                lines.append(f"Storage Used:    {storage_pct:.2f}%")
            lines.append("")

            # TF-IDF configuration.
            lines.append("-" * 50)
            lines.append("TF-IDF CONFIGURATION")
            lines.append("-" * 50)

            tfidf_matrix = _tfidf_matrix_getter() if callable(_tfidf_matrix_getter) else _tfidf_matrix_getter
            if tfidf_matrix is not None:
                lines.append("Status:          Active")
                lines.append(f"Matrix Shape:    {tfidf_matrix.shape}")
                if _tfidf_vectorizer and hasattr(_tfidf_vectorizer, 'vocabulary_'):
                    lines.append(f"Vocabulary Size: {len(_tfidf_vectorizer.vocabulary_):,}")
            else:
                lines.append("Status:          Not Built")

            lines.append("Max Features:    5,000")
            lines.append("N-gram Range:    (1, 3)")
            lines.append("Max Doc Freq:    0.95")
            lines.append("Rebuild Interval: 1 hour")

            if _last_rebuild_getter:
                last_rebuild = _last_rebuild_getter()
                if last_rebuild:
                    time_since = datetime.now() - last_rebuild
                    lines.append(f"Last Rebuild:    {str(time_since).split('.')[0]} ago")

            lines.append("")

            # LLM configuration.
            lines.append("-" * 50)
            lines.append("LLM CONFIGURATION")
            lines.append("-" * 50)
            lines.append("Provider:        Groq")
            lines.append("Model:           openai/gpt-oss-20b")
            lines.append("Temperature:     1.0")
            lines.append("Max Tokens:      8192")
            lines.append("Reasoning Effort: medium")
            lines.append("")

            # Token usage statistics.
            if _token_stats_getter:
                lines.append("-" * 50)
                lines.append("TOKEN USAGE")
                lines.append("-" * 50)

                token_stats = _token_stats_getter()
                total_api_calls = token_stats.get('total_api_calls', 0)
                total_prompt_tokens = token_stats.get('total_prompt_tokens', 0)
                total_completion_tokens = token_stats.get('total_completion_tokens', 0)
                total_tokens_used = token_stats.get('total_tokens_used', 0)

                avg_tokens_per_call = total_tokens_used / total_api_calls if total_api_calls > 0 else 0

                lines.append(f"Total API Calls:     {total_api_calls:,}")
                lines.append(f"Prompt Tokens:       {total_prompt_tokens:,}")
                lines.append(f"Completion Tokens:   {total_completion_tokens:,}")
                lines.append(f"Total Tokens Used:   {total_tokens_used:,}")
                lines.append(f"Avg Tokens/Call:     {avg_tokens_per_call:.1f}")
                lines.append("")

            # Guild information.
            lines.append("-" * 50)
            lines.append("SERVER INFO")
            lines.append("-" * 50)
            lines.append(f"Connected Servers: {len(_bot.guilds)}")

            for guild in _bot.guilds:
                lines.append(f"  - {guild.name}: {guild.member_count} members")
            lines.append("")

            # Channel breakdown.
            if _message_history is not None and len(_message_history) > 0:
                lines.append("-" * 50)
                lines.append("MESSAGES BY CHANNEL")
                lines.append("-" * 50)

                channel_counts = {}
                for msg in _message_history:
                    channel_name = msg['channel']
                    channel_counts[channel_name] = channel_counts.get(channel_name, 0) + 1

                if channel_counts:
                    max_count = max(channel_counts.values())
                    sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)

                    for name, count in sorted_channels[:15]:
                        bar_length = int(count / max_count * 20)
                        bar = "#" * bar_length + "." * (20 - bar_length)
                        lines.append(f"#{name[:15]:<15} [{bar}] {count:,}")

                    if len(sorted_channels) > 15:
                        lines.append(f"... and {len(sorted_channels) - 15} more channels")

        else:
            lines.append("Bot is starting up or not connected...")

    except Exception as e:
        lines.append(f"Error generating status: {e}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


@app.route('/')
def home():
    """Render real-time status as plain text."""
    return Response(_generate_status_text(), mimetype='text/plain')


@app.route('/health')
def health():
    """Simple health check endpoint for keep-alive pings."""
    return Response("OK", mimetype='text/plain')


def _run_server():
    """Run the Flask server."""
    app.run(host='0.0.0.0', port=8080)


def start():
    """Start the web status server in a background thread."""
    thread = Thread(target=_run_server, daemon=True)
    thread.start()
    return thread
