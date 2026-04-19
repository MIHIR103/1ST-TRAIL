import os
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# -------- LOGGING --------
logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("TOKEN")

# -------- DETECT STOCK --------
def detect(name):
    mapping = {
        "reliance":"RELIANCE.NS",
        "tcs":"TCS.NS",
        "infosys":"INFY.NS",
        "apple":"AAPL",
        "tesla":"TSLA"
    }
    return mapping.get(name.lower(), name.upper())

# -------- FETCH DATA --------
def get_data(symbol, period="6mo"):
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        return df.dropna()
    except Exception as e:
        logging.error(e)
        return None

# -------- RSI --------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

# -------- ANALYSIS --------
def analyze(symbol):
    df = get_data(symbol)
    if df is None:
        return "❌ Invalid stock", None

    df["RSI"] = rsi(df["Close"])
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df.dropna(inplace=True)

    last = df.iloc[-1]

    price = float(last["Close"])
    rsi_v = float(last["RSI"])
    ma20 = float(last["MA20"])
    ma50 = float(last["MA50"])

    decision = "BUY" if rsi_v < 30 and ma20 > ma50 else "SELL" if rsi_v > 70 else "HOLD"

    text = f"""📊 {symbol}
Price: {round(price,2)}
RSI: {round(rsi_v,2)}
MA20: {round(ma20,2)}
MA50: {round(ma50,2)}

Decision: {decision}"""

    return text, df

# -------- ML --------
def ml(df):
    try:
        df = df.copy()
        df["RSI"] = rsi(df["Close"])
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df.dropna(inplace=True)

        if len(df) < 50:
            return "\n🤖 ML: Not enough data"

        X = df[["RSI","MA20","MA50"]][:-1]
        y = df["Close"].shift(-1)[:-1]

        model = LinearRegression().fit(X,y)
        pred = model.predict([X.iloc[-1]])[0]
        curr = df["Close"].iloc[-1]

        signal = "BUY" if pred > curr else "SELL"
        return f"\n🤖 ML Prediction\nNext Price: {round(pred,2)}\nSignal: {signal}"
    except Exception as e:
        logging.error(e)
        return "\n🤖 ML Error"

# -------- BACKTEST --------
def backtest(symbol):
    df = get_data(symbol, "1y")
    if df is None:
        return "❌ No data", None

    df["Returns"] = df["Close"].pct_change()
    df["Strategy"] = df["Returns"].shift()
    df.dropna(inplace=True)

    eq = (df["Strategy"]+1).cumprod()
    return f"📈 Return: {round((eq.iloc[-1]-1)*100,2)}%", eq

# -------- PORTFOLIO --------
def portfolio(symbols):
    if not symbols:
        return "❌ Use: /portfolio AAPL TSLA"

    valid = []
    for s in symbols:
        _, df = analyze(s)
        if df is not None:
            valid.append(s)

    if not valid:
        return "❌ No valid stocks"

    return f"📊 Portfolio: {', '.join(valid)}"

# -------- ALERT SYSTEM --------
alerts = {}

async def set_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Use: /alert AAPL 200")
        return

    sym = detect(context.args[0])
    price = float(context.args[1])

    alerts[update.effective_chat.id] = (sym, price)
    await update.message.reply_text(f"🔔 Alert set for {sym} at {price}")

async def check_alerts(context: ContextTypes.DEFAULT_TYPE):
    for chat_id, (sym, target) in alerts.items():
        df = get_data(sym, "1d")
        if df is None:
            continue

        curr = float(df["Close"].iloc[-1])
        if curr >= target:
            await context.bot.send_message(chat_id, f"🚨 {sym} reached {curr}")

# -------- CHART --------
def chart(df):
    file = "chart.png"
    if os.path.exists(file):
        os.remove(file)

    mpf.plot(df, type='candle', mav=(20,50), volume=True,
             savefig=dict(fname=file, dpi=100))
    return file

# -------- COMMANDS --------
async def stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Use: /stock AAPL")
        return

    sym = detect(context.args[0])
    text, df = analyze(sym)

    if df is None:
        await update.message.reply_text(text)
        return

    await update.message.reply_text(text + ml(df))

    img = chart(df)
    with open(img, "rb") as f:
        await update.message.reply_photo(photo=f)

async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Use: /backtest AAPL")
        return

    sym = detect(context.args[0])
    res, eq = backtest(sym)

    if eq is not None:
        plt.figure()
        plt.plot(eq)
        plt.savefig("bt.png")
        plt.close()

        await update.message.reply_text(res)
        await update.message.reply_photo(photo=open("bt.png","rb"))
    else:
        await update.message.reply_text(res)

async def portfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Use: /portfolio AAPL TSLA")
        return

    syms = [detect(s) for s in context.args]
    res = portfolio(syms)
    await update.message.reply_text(res)

# -------- ERROR HANDLER --------
async def error_handler(update, context):
    logging.error(f"Update {update} caused error {context.error}")

# -------- RUN --------
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("stock", stock))
app.add_handler(CommandHandler("backtest", backtest_cmd))
app.add_handler(CommandHandler("portfolio", portfolio_cmd))
app.add_handler(CommandHandler("alert", set_alert))

app.add_error_handler(error_handler)

# FIX: ensure job queue exists
if app.job_queue:
    app.job_queue.run_repeating(check_alerts, interval=60)

print("🚀 BOT RUNNING...")
app.run_polling()
