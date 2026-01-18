"""Web status page module for displaying bot statistics."""

from collections import Counter
from datetime import datetime
from threading import Thread

import psutil
from flask import Flask, Response

app = Flask('web_status')

# Configuration state (set by main module).
_config = {}


def configure(**kwargs):
    """Configure references to bot objects for status display."""
    _config.update(kwargs)


def _section(title):
    """Format a section header."""
    return f"{'-' * 50}\n{title}\n{'-' * 50}"


def _generate_status_text():
    """Generate status text for the web page."""
    lines = [
        "=" * 50,
        "SIMON'S BIONIX DISCORD CHATBOT STATUS",
        "=" * 50,
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]

    bot = _config.get('bot')
    try:
        if bot and bot.user:
            lines.append(_section("BOT INFO"))
            lines.append(f"Status:          {'Online' if not bot.is_closed() else 'Offline'}")
            lines.append(f"Bot User:        {bot.user.name}#{bot.user.discriminator}")
            lines.append(f"Bot ID:          {bot.user.id}")
            if start_time := _config.get('start_time'):
                lines.append(f"Uptime:          {str(datetime.now() - start_time).split('.')[0]}")
            lines.append(f"Latency:         {bot.latency * 1000:.1f}ms\n")

            lines.append(_section("MEMORY & STORAGE"))
            lines.append(f"RAM Usage:       {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
            if (history := _config.get('message_history')) is not None:
                bot_count = sum(1 for msg in history if msg.get('is_bot'))
                lines.append(f"Messages Stored: {len(history):,}")
                lines.append(f"  - User msgs:   {len(history) - bot_count:,}")
                lines.append(f"  - Bot msgs:    {bot_count:,}")
            lines.append("")

            lines.append(_section("TF-IDF CONFIGURATION"))
            getter = _config.get('tfidf_matrix_getter')
            tfidf_matrix = getter() if callable(getter) else getter
            vectorizer = _config.get('tfidf_vectorizer')
            if tfidf_matrix is not None:
                lines.append("Status:          Built")
                lines.append(f"Matrix Shape:    {tfidf_matrix.shape}")
                if vectorizer and hasattr(vectorizer, 'vocabulary_'):
                    lines.append(f"Vocabulary Size: {len(vectorizer.vocabulary_):,}")
            else:
                lines.append("Status:          Not Built")
            lines.extend(["Max Features:    5,000", "N-gram Range:    (1, 3)", "Max Doc Freq:    0.95", "Rebuild Interval: 1 hour"])

            if (getter := _config.get('last_rebuild_getter')) and (last := getter()):
                lines.append(f"Last Rebuild:    {str(datetime.now() - last).split('.')[0]} ago")
            if (getter := _config.get('next_rebuild_getter')) and (nxt := getter()):
                if (secs := (nxt - datetime.now()).total_seconds()) > 0:
                    lines.append(f"Next Rebuild:    {int(secs // 60)} minute{'s' if int(secs // 60) != 1 else ''}")
            lines.append("")

            lines.append(_section("LLM CONFIGURATION"))
            lines.extend(["Provider:        Groq", "Model:           openai/gpt-oss-20b", "Temperature:     1.0", "Max Tokens:      8192", "Reasoning Effort: medium", ""])

            if getter := _config.get('token_stats_getter'):
                stats = getter()
                total_calls = stats.get('total_api_calls', 0)
                lines.append(_section("TOKEN USAGE"))
                lines.append(f"Total API Calls:     {total_calls:,}")
                lines.append(f"Prompt Tokens:       {stats.get('total_prompt_tokens', 0):,}")
                lines.append(f"Completion Tokens:   {stats.get('total_completion_tokens', 0):,}")
                lines.append(f"Total Tokens Used:   {stats.get('total_tokens_used', 0):,}")
                lines.append(f"Avg Tokens/Call:     {stats.get('total_tokens_used', 0) / total_calls if total_calls else 0:.1f}\n")

            if limits := _config.get('context_limits'):
                lines.append(_section("CONTEXT LIMITS"))
                for key, label in [('bot_channel_limit', 'Bot Channel Msgs'), ('other_channel_limit', 'Other Channel Msgs'),
                                   ('max_other_channels', 'Max Other Channels'), ('semantic_top_k', 'Semantic Top K'),
                                   ('bot_msg_max_len', 'Bot Msg Max Len'), ('other_msg_max_len', 'Other Msg Max Len'),
                                   ('semantic_msg_max_len', 'Semantic Msg Max Len')]:
                    lines.append(f"{label + ':':<21}{limits.get(key, 'N/A')}")
                lines.append("")

            lines.append(_section("SERVER INFO"))
            lines.append(f"Connected Servers: {len(bot.guilds)}")
            lines.extend(f"  - {g.name}: {g.member_count} members" for g in bot.guilds)
            lines.append("")

            if (history := _config.get('message_history')) and history:
                lines.append(_section("MESSAGES BY CHANNEL"))
                channel_counts = Counter(msg['channel'] for msg in history)
                max_count = max(channel_counts.values())
                for name, count in channel_counts.most_common(15):
                    bar_len = int(count / max_count * 20)
                    lines.append(f"#{name[:15]:<15} [{'#' * bar_len}{'.' * (20 - bar_len)}] {count:,}")
                if len(channel_counts) > 15:
                    lines.append(f"... and {len(channel_counts) - 15} more channels")
        else:
            lines.append("Bot is starting up or not connected...")
    except Exception as e:
        lines.append(f"Error generating status: {e}")

    lines.extend(["", "=" * 50])
    return "\n".join(lines)


@app.route('/')
def home():
    """Render real-time status as plain text."""
    return Response(_generate_status_text(), mimetype='text/plain')


@app.route('/health')
def health():
    """Simple health check endpoint for keep-alive pings."""
    return Response("OK", mimetype='text/plain')


def start():
    """Start the web status server in a background thread."""
    thread = Thread(target=lambda: app.run(host='0.0.0.0', port=8080), daemon=True)
    thread.start()
    return thread
