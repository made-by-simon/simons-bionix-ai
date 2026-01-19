from flask import Flask, Response
from threading import Thread
import psutil
from datetime import datetime

app = Flask('')

# These will be set by the main bot script.
bot_reference = None
message_history_reference = None
tfidf_vectorizer_reference = None
tfidf_matrix_reference = None
bot_start_time_reference = None
max_messages_reference = None
last_rebuild_getter_reference = None
token_stats_getter_reference = None


def set_references(bot, message_history, tfidf_vectorizer, tfidf_matrix_getter, start_time, max_messages, last_rebuild_getter=None, token_stats_getter=None):
    """Set references to bot objects for status display."""
    global bot_reference, message_history_reference, tfidf_vectorizer_reference
    global tfidf_matrix_reference, bot_start_time_reference, max_messages_reference
    global last_rebuild_getter_reference, token_stats_getter_reference
    bot_reference = bot
    message_history_reference = message_history
    tfidf_vectorizer_reference = tfidf_vectorizer
    tfidf_matrix_reference = tfidf_matrix_getter
    bot_start_time_reference = start_time
    max_messages_reference = max_messages
    last_rebuild_getter_reference = last_rebuild_getter
    token_stats_getter_reference = token_stats_getter


def get_connection_status():
    """Determine the actual connection status based on latency and state."""
    if not bot_reference or bot_reference.is_closed():
        return "Offline", "N/A"

    latency = bot_reference.latency
    if latency == float('inf'):
        return "Disconnected (no heartbeat)", "∞"
    elif latency > 5.0:
        return "Unhealthy (high latency)", f"{latency * 1000:.0f}ms"
    elif latency > 1.0:
        return "Degraded", f"{latency * 1000:.0f}ms"
    else:
        return "Online", f"{latency * 1000:.1f}ms"


@app.route('/health')
def health():
    """Health check endpoint for monitoring."""
    if not bot_reference or bot_reference.is_closed():
        return Response("UNHEALTHY: Bot offline", status=503, mimetype='text/plain')

    latency = bot_reference.latency
    if latency == float('inf') or latency > 5.0:
        return Response(f"UNHEALTHY: Latency {latency}", status=503, mimetype='text/plain')

    return Response(f"OK: Latency {latency*1000:.1f}ms", status=200, mimetype='text/plain')


@app.route('/')
def home():
    """Render real-time status as plain text."""
    lines = []
    lines.append("=" * 50)
    lines.append("SIMON'S BIONIX DISCORD CHATBOT STATUS")
    lines.append("=" * 50)
    lines.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    try:
        if bot_reference and bot_reference.user:
            # Bot information.
            lines.append("-" * 50)
            lines.append("BOT INFO")
            lines.append("-" * 50)
            status, latency_str = get_connection_status()
            lines.append(f"Status:          {status}")
            lines.append(f"Bot User:        {bot_reference.user.name}#{bot_reference.user.discriminator}")
            lines.append(f"Bot ID:          {bot_reference.user.id}")

            if bot_start_time_reference:
                uptime = datetime.now() - bot_start_time_reference
                uptime_str = str(uptime).split('.')[0]
                lines.append(f"Uptime:          {uptime_str}")

            lines.append(f"Latency:         {latency_str}")
            lines.append("")

            # Memory and storage.
            lines.append("-" * 50)
            lines.append("MEMORY & STORAGE")
            lines.append("-" * 50)

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            lines.append(f"RAM Usage:       {memory_mb:.1f} MB")

            if message_history_reference is not None:
                total_messages = len(message_history_reference)
                max_msgs = max_messages_reference or 1000000
                bot_msg_count = sum(1 for msg in message_history_reference if msg.get('is_bot', False))
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

            tfidf_matrix = tfidf_matrix_reference() if callable(tfidf_matrix_reference) else tfidf_matrix_reference
            if tfidf_matrix is not None:
                lines.append("Status:          Active")
                lines.append(f"Matrix Shape:    {tfidf_matrix.shape}")
                if tfidf_vectorizer_reference and hasattr(tfidf_vectorizer_reference, 'vocabulary_'):
                    lines.append(f"Vocabulary Size: {len(tfidf_vectorizer_reference.vocabulary_):,}")
            else:
                lines.append("Status:          Not Built")

            lines.append("Max Features:    5,000")
            lines.append("N-gram Range:    (1, 3)")
            lines.append("Max Doc Freq:    0.95")
            lines.append("Rebuild Interval: 1 hour")

            if last_rebuild_getter_reference:
                last_rebuild = last_rebuild_getter_reference()
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
            if token_stats_getter_reference:
                lines.append("-" * 50)
                lines.append("TOKEN USAGE")
                lines.append("-" * 50)

                token_stats = token_stats_getter_reference()
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
            lines.append(f"Connected Servers: {len(bot_reference.guilds)}")

            for guild in bot_reference.guilds:
                lines.append(f"  - {guild.name}: {guild.member_count} members")
            lines.append("")

            # Channel breakdown.
            if message_history_reference is not None and len(message_history_reference) > 0:
                lines.append("-" * 50)
                lines.append("MESSAGES BY CHANNEL")
                lines.append("-" * 50)

                channel_counts = {}
                for msg in message_history_reference:
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

    return Response("\n".join(lines), mimetype='text/plain')


def run():
    """Run the Flask server."""
    app.run(host='0.0.0.0', port=8080)


def keep_alive():
    """Start the keep-alive server in a separate thread."""
    t = Thread(target=run)
    t.daemon = True
    t.start()