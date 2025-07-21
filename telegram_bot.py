import os, asyncio, logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from bot import get_balance, get_open, tick, close_all_trades, place_dummy_trade
import nest_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
last = datetime.utcnow() - timedelta(minutes=10)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bal = await get_balance()
    ops = await get_open()
    open_count = len(ops)
    unrealized_total = sum(float(t['unrealizedPL']) for t in ops) if ops else 0
    roi_percent = (unrealized_total / bal * 100) if bal else 0
    time_since_trade = (datetime.utcnow() - last).seconds // 60
    msg = (
        f"✅ *Bot Status*\n"
        f"🤖 Bot is online and operational.\n"
        f"💰 Balance: `£{bal:,.2f}`\n"
        f"📈 Open Trades: {open_count}\n"
        f"💹 Unrealized P&L: `£{unrealized_total:.2f}` (~{roi_percent:.2f}%)\n"
        f"⏱️ Last trade: {time_since_trade} minutes ago\n"
        f"⚙️ Running scheduled trade ticks every 5 minutes.\n\n"
        f"📊 *Performance Snapshot:*\n"
        f"- Risk per trade: `{float(os.getenv('TRADE_RISK_PERCENT', '0.02'))*100:.0f}%`\n"
        f"- Max leverage: `{os.getenv('MAX_LEVERAGE', '20')}x`\n"
        f"- Allowed pairs: `{os.getenv('ALLOWED_PAIRS', 'EUR_USD,USD_JPY')}`"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ops = await get_open()
    if not ops:
        await update.message.reply_text("📭 No open trades for today, no P&L to report.")
        return
    bal = await get_balance()
    total_unrealized = sum(float(t['unrealizedPL']) for t in ops)
    roi = (total_unrealized / bal * 100) if bal else 0
    msg = (
        f"📅 *Daily P&L Report*\n"
        f"💰 Current Balance: `£{bal:,.2f}`\n"
        f"📈 Unrealized P&L from open trades: `£{total_unrealized:.2f}` (~{roi:.2f}%)\n"
        f"⚠️ Note: Realized P&L and closed trades tracking pending implementation."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def weekly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ops = await get_open()
    bal = await get_balance()
    total_unrealized = sum(float(t['unrealizedPL']) for t in ops) if ops else 0
    roi = (total_unrealized / bal * 100) if bal else 0
    msg = (
        f"📈 *Weekly P&L Report*\n"
        f"💰 Current Balance: `£{bal:,.2f}`\n"
        f"📊 Unrealized P&L on open trades: `£{total_unrealized:.2f}` (~{roi:.2f}%)\n"
        f"⚠️ Realized weekly P&L tracking coming soon."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ops = await get_open()
    if not ops:
        await update.message.reply_text("📭 No open trades.")
        return
    bal = await get_balance()
    msgs = []
    for t in ops:
        unrealized = float(t['unrealizedPL'])
        roi_pct = (100 * unrealized / bal) if bal else 0
        emoji = "📈" if unrealized >= 0 else "📉"
        msgs.append(f"{emoji} {t['instrument']}: `£{unrealized:.2f}` (~{roi_pct:.2f}%)")
    total_roi = sum(100 * float(t['unrealizedPL']) / bal for t in ops)
    msgs.append(f"📊 *Total ROI:* `{total_roi:.2f}%`")
    await update.message.reply_text("\n".join(msgs), parse_mode="Markdown")

async def maketrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last
    now = datetime.utcnow()
    last = now
    await update.message.reply_text("⚙️ Trade request acknowledged.\nAuto-trading operates on a 5-min cycle.")
    try:
        await tick()
        await update.message.reply_text("✅ Trade tick executed.")
    except Exception as e:
        logger.error(f"Error during trade tick: {e}")
        await update.message.reply_text(f"❌ Error executing trade tick: {e}")

async def dummytrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await place_dummy_trade()
        await update.message.reply_text("✅ Dummy trade placed successfully.")
    except Exception as e:
        logger.error(f"Error placing dummy trade: {e}")
        await update.message.reply_text(f"❌ Failed to place dummy trade: {e}")

async def closeall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    closed_trades = await close_all_trades()
    if not closed_trades:
        await update.message.reply_text("🛑 No trades to close.")
        return
    msg_lines = ["🚪 *Closed All Trades*"]
    total_pl = 0
    for trade in closed_trades:
        pl = trade.get('pl', 0)
        total_pl += pl
        status_emoji = "✅" if pl >= 0 else "❌"
        msg_lines.append(f"{status_emoji} {trade['instrument']} P&L: `£{pl:.2f}`")
    msg_lines.append(f"💵 *Total P&L:* `£{total_pl:.2f}`")
    await update.message.reply_text("\n".join(msg_lines), parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🧾 *Command Menu*\n"
        "/status - Check bot and balance status\n"
        "/open - Show current open trades\n"
        "/maketrade - Trigger manual trade (no cooldown)\n"
        "/dummytrade - Place a random small dummy trade instantly\n"
        "/daily - View today's P&L\n"
        "/weekly - View weekly P&L\n"
        "/closeall - Close all open trades\n"
        "/help - Show this menu\n"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

def build():
    app = ApplicationBuilder().token(os.getenv("TELEGRAM_TOKEN")).build()
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("daily", daily))
    app.add_handler(CommandHandler("weekly", weekly))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("maketrade", maketrade))
    app.add_handler(CommandHandler("dummytrade", dummytrade))
    app.add_handler(CommandHandler("closeall", closeall))
    app.add_handler(CommandHandler("help", help_cmd))
    return app

async def main():
    nest_asyncio.apply()
    app = build()
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
