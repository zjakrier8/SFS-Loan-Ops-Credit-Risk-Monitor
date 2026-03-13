"""
SFS Loan Ops Credit Risk Escalation & Loan Performance Dashboard

Monitors:
1. "Auto Evaluation recommends reserve" messages from #cap-credit-risk-esc
2. Bi-directional escalations between Credit Risk and SFS Loan Ops
3. Loan performance metrics from Snowflake (APP_CAPITAL, APP_RISK)
"""

import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import Counter

import anthropic
import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SLACK_CHANNEL_ID = "C685Q2H3P"
CHANNEL_NAME = "#cap-credit-risk-esc"
CREDENTIALS_PATH = Path.home() / ".config" / "slack-skill" / "credentials.json"


# ---------------------------------------------------------------------------
# Helpers — Slack
# ---------------------------------------------------------------------------
def _load_slack_token() -> str:
    env_token = os.environ.get("SLACK_TOKEN", "")
    if env_token:
        return env_token
    with open(CREDENTIALS_PATH) as f:
        return json.load(f)["token"]


def _slack_get(method: str, params: dict, _retries: int = 3) -> dict:
    token = _load_slack_token()
    with httpx.Client(
        base_url="https://slack.com/api",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    ) as client:
        for attempt in range(_retries):
            resp = client.get(f"/{method}", params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                return data
            if data.get("error") == "ratelimited" and attempt < _retries - 1:
                wait = int(resp.headers.get("Retry-After", 3))
                time.sleep(wait)
                continue
            raise RuntimeError(f"Slack API error: {data.get('error')}")
        return data


def _fetch_channel_history(oldest_ts: str, cursor: str | None = None, limit: int = 1000) -> dict:
    params = {"channel": SLACK_CHANNEL_ID, "oldest": oldest_ts, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    return _slack_get("conversations.history", params)


@st.cache_data(ttl=900)
def fetch_slack_messages(days_back: int) -> list[dict]:
    """Fetch all messages from the channel within the last `days_back` days."""
    oldest = datetime.now(timezone.utc) - timedelta(days=days_back)
    oldest_ts = str(oldest.timestamp())
    all_msgs: list[dict] = []
    cursor = None
    while True:
        data = _fetch_channel_history(oldest_ts, cursor=cursor)
        all_msgs.extend(data.get("messages", []))
        cursor = data.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return all_msgs


def classify_message(msg: dict) -> str | None:
    """Classify a message into a category."""
    text = msg.get("text", "")
    if "Auto Evaluation recommends reserve" in text or "Auto Eval recommends reserve" in text:
        return "auto_eval_reserve"
    if "Credit Risk Submission" in text:
        return "bpo_cr_escalation"
    if "Capital Submission" in text:
        return "capital_escalation"
    return None


def extract_token_from_message(text: str) -> str | None:
    """Pull the merchant token from the structured message."""
    match = re.search(r"\*Token\*\s*\n([A-Z0-9]+)", text)
    return match.group(1) if match else None


def regulator_url(token: str) -> str:
    return f"https://regulator.sqprod.co/n/merchants/{token}"



def _col(df: pd.DataFrame, name: str):
    """Case-insensitive column access helper."""
    col_map = {c.upper(): c for c in df.columns}
    return col_map.get(name.upper(), name)


# Regex fallback if LLM classification is unavailable
NO_EXTEND_PATTERNS = re.compile(
    r"do not extend|don.t extend|not recommend extend|would not recommend|do not recommend"
    r"|remove eligib|pause eligib|recommend reserve|recommends reserve|no further capital",
    re.IGNORECASE,
)

_thread_fetch_semaphore = threading.Semaphore(10)


def _fetch_one_thread_raw(ts: str) -> list[dict]:
    """Fetch replies for a single thread. Returns empty list on failure."""
    with _thread_fetch_semaphore:
        for attempt in range(2):
            try:
                data = _slack_get("conversations.replies", {
                    "channel": SLACK_CHANNEL_ID, "ts": ts, "limit": 50,
                })
                msgs = data.get("messages", [])
                return msgs[1:] if len(msgs) > 1 else []
            except Exception:
                if attempt < 1:
                    time.sleep(2)
        return []


@st.cache_data(ttl=900)
def fetch_thread_replies(thread_ts_list: tuple[str, ...]) -> dict[str, list[dict]]:
    """Fetch thread replies in parallel. Shared across all users via st.cache_data."""
    results: dict[str, list[dict]] = {}
    if not thread_ts_list:
        return results
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_one_thread_raw, ts): ts for ts in thread_ts_list}
        for future in as_completed(futures):
            ts = futures[future]
            results[ts] = future.result()
    return results


def _get_anthropic_client() -> anthropic.Anthropic | None:
    """Return an Anthropic client if an API key is available, else None."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


@st.cache_data(ttl=900)
def classify_threads_with_llm(thread_data: tuple[tuple[str, str], ...]) -> set[str]:
    """Use Claude to classify which threads have CR recommending not to extend.

    Args:
        thread_data: tuple of (thread_ts, combined_reply_text) pairs.

    Returns:
        Set of thread_ts values where CR recommends not extending.
    """
    client = _get_anthropic_client()
    if client is None:
        return set()

    # Build the prompt with all threads
    thread_block = ""
    for i, (ts, text) in enumerate(thread_data, 1):
        # Truncate each thread's text to keep prompt manageable
        snippet = text[:1000] if len(text) > 1000 else text
        thread_block += f"\n--- Thread {i} (ID: {ts}) ---\n{snippet}\n"

    prompt = f"""You are analyzing Slack thread replies from a credit risk escalation channel.
For each thread below, determine if any reply indicates that Credit Risk (CR) recommends NOT extending capital/loans to the seller. This includes:
- Explicit: "do not extend", "don't extend", "do not recommend", "would not recommend"
- Actions: "remove eligibility", "pause eligibility", "recommend reserve"
- Implicit: "hold off", "not comfortable extending", "too risky to extend", "should not proceed", "decline", "we should pass"

Return ONLY a valid JSON array of thread IDs where CR recommends not extending. If none, return [].
Example: ["{thread_data[0][0] if thread_data else '1234.5678'}"]

Threads:
{thread_block}"""

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Parse the JSON array from the response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            ts_list = json.loads(match.group())
            return set(ts_list)
    except Exception:
        pass
    return set()


def classify_no_extend_threads(
    thread_ts_list: tuple[str, ...],
    all_replies: dict[str, list[dict]],
) -> set[str]:
    """Classify threads using regex + LLM (union of both).

    Returns set of thread timestamps where CR recommends not extending.
    """
    # Collect threads that have replies
    thread_data = []
    for ts in thread_ts_list:
        replies = all_replies.get(ts, [])
        if replies:
            combined = "\n".join(r.get("text", "") for r in replies)
            thread_data.append((ts, combined))

    if not thread_data:
        return set()

    # Regex pass (always runs, instant)
    flagged_ts: set[str] = set()
    for ts, combined in thread_data:
        if NO_EXTEND_PATTERNS.search(combined):
            flagged_ts.add(ts)

    # LLM pass (union with regex results)
    client = _get_anthropic_client()
    if client is not None:
        for i in range(0, len(thread_data), 50):
            batch = tuple(thread_data[i : i + 50])
            flagged_ts |= classify_threads_with_llm(batch)

    return flagged_ts


# ---------------------------------------------------------------------------
# Helpers — Snowflake
# ---------------------------------------------------------------------------
_USE_PYSNOWFLAKE = os.environ.get("USE_PYSNOWFLAKE", "").lower() in ("1", "true")

import snowflake.connector


@st.cache_resource
def _get_snowflake_conn():
    """Persistent Snowflake connection — created once, reused across reruns."""
    if _USE_PYSNOWFLAKE:
        return None  # sq-pysnowflake manages its own connection
    return snowflake.connector.connect(
        account="SQUAREINC-SQUARE",
user="zachk@squareup.com",
authenticator="externalbrowser",
        warehouse="ADHOC__SMALL",
        role="ZACHK",
    )


@st.cache_data(ttl=1800, persist="disk")
def run_snowflake_query(sql: str) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""
    if _USE_PYSNOWFLAKE:
        try:
            from pysnowflake import Session
            with Session() as ss:
                return ss.download(query=sql)
        except Exception as e:
            st.error(f"Snowflake query failed: {e}")
            return pd.DataFrame()
    else:
        try:
            conn = _get_snowflake_conn()
            cur = conn.cursor()
            cur.execute(sql)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()
        except Exception as e:
            # Connection may have expired — reconnect once
            try:
                _get_snowflake_conn.clear()
                conn = _get_snowflake_conn()
                cur = conn.cursor()
                cur.execute(sql)
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()
            except Exception as e2:
                st.error(f"Snowflake query failed: {e2}")
                return pd.DataFrame()



@st.cache_data(ttl=1800, persist="disk")
def fetch_all_snowflake_data(tokens: tuple[str, ...], days_back: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Fetch loan performance, denied apps, and MCC in a single Snowflake call.

    Returns (df_loan_perf, df_denied, mcc_map).
    """
    if not tokens:
        return pd.DataFrame(), pd.DataFrame(), {}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    all_perf: list[pd.DataFrame] = []
    all_denied: list[pd.DataFrame] = []
    mcc_map: dict[str, str] = {}

    for i in range(0, len(tokens), 500):
        batch = tokens[i : i + 500]
        in_list = ", ".join(f"'{t}'" for t in batch)

        # Single query with 3 result sets via UNION ALL markers
        sql = f"""
        SELECT 'PERF' AS _src,
            pg.plan_group_id,
            pg.primary_user_token AS unit_token,
            pg.status,
            pg.created_at,
            pgdps.past_due_days AS days_past_due,
            pgdps.past_due_cents / 100 AS past_due_dollars,
            pg.product_name,
            pg.financed_cents / 100 AS financed_dollars,
            pg.receivables_cents / 100 AS receivables_dollars,
            pgdps.outstanding_cents / 100 AS outstanding_dollars,
            pg.risk_grade_name,
            NULL AS denied_dollars,
            NULL AS submitted_at,
            NULL AS mcc
        FROM APP_CAPITAL.APP_CAPITAL.PLAN_GROUP_DAILY_PERFORMANCE_SUMMARY pgdps
        JOIN APP_CAPITAL.APP_CAPITAL.PLAN_GROUPS pg
            ON pg.plan_group_id = pgdps.plan_group_id
        WHERE pgdps.the_date = (
            SELECT MAX(the_date)
            FROM APP_CAPITAL.APP_CAPITAL.PLAN_GROUP_DAILY_PERFORMANCE_SUMMARY
        )
        AND pg.primary_user_token IN ({in_list})
        AND pg.currency_code = 'USD'

        UNION ALL

        SELECT 'DENIED' AS _src,
            NULL, af.UNIT_TOKEN, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            af.MAX_OFFER_AMOUNT_USD AS denied_dollars,
            af.SUBMITTED_AT AS submitted_at,
            NULL
        FROM APP_CAPITAL.APP_CAPITAL.APPLICATION_FUNNEL af
        WHERE af.UNIT_TOKEN IN ({in_list})
        AND af.APPLICATION_FLOW_STATUS = 'declined'
        AND af.SUBMITTED_AT >= '{cutoff}'

        UNION ALL

        SELECT 'MCC' AS _src,
            NULL, m.UNIT_TOKEN, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, m.MCC
        FROM (
            SELECT UNIT_TOKEN, MCC,
                   ROW_NUMBER() OVER (PARTITION BY UNIT_TOKEN ORDER BY SUBMITTED_AT DESC) AS rn
            FROM APP_CAPITAL.APP_CAPITAL.APPLICATION_FUNNEL
            WHERE UNIT_TOKEN IN ({in_list}) AND MCC IS NOT NULL
        ) m
        WHERE m.rn = 1
        """
        df = run_snowflake_query(sql)
        if not df.empty:
            src_col = _col(df, "_SRC")
            perf = df[df[src_col] == "PERF"].drop(columns=[src_col])
            denied = df[df[src_col] == "DENIED"].drop(columns=[src_col])
            mcc_rows = df[df[src_col] == "MCC"].drop(columns=[src_col])
            if not perf.empty:
                all_perf.append(perf)
            if not denied.empty:
                all_denied.append(denied)
            if not mcc_rows.empty:
                tok_c = _col(mcc_rows, "UNIT_TOKEN")
                mcc_c = _col(mcc_rows, "MCC")
                for _, row in mcc_rows.iterrows():
                    mcc_map[str(row[tok_c])] = str(row[mcc_c])

    df_perf = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()
    df_denied = pd.concat(all_denied, ignore_index=True) if all_denied else pd.DataFrame()
    return df_perf, df_denied, mcc_map


# ---------------------------------------------------------------------------
# Dashboard Layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SFS Loan Ops Credit Risk Monitor",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("SFS Loan Ops | Credit Risk Monitor")
st.caption(
    f"Monitoring {CHANNEL_NAME} | Quantifying collaboration between SFS Loan Ops and Credit Risk "
    "to mitigate shared loss on escalated accounts"
)

# Sidebar controls
st.sidebar.header("Controls")
days_back = st.sidebar.slider("Days of Slack history", 7, 90, 30)
refresh = st.sidebar.button("Refresh data")

if refresh:
    st.cache_data.clear()
    st.session_state.pop("thread_replies", None)
    st.session_state.pop("flagged_thread_ts", None)

# Clear thread session state when slider changes
if st.session_state.get("_last_days_back") != days_back:
    st.session_state["_last_days_back"] = days_back
    st.session_state.pop("thread_replies", None)
    st.session_state.pop("flagged_thread_ts", None)

# ===== Tab layout =====
tab_esc, tab_miti = st.tabs([
    "Escalation Monitor",
    "Loan Loss Mitigation",
])

# ---------------------------------------------------------------------------
# Fetch & classify Slack data (shared across tabs)
# ---------------------------------------------------------------------------
with st.spinner("Fetching Slack messages..."):
    all_messages_180d = fetch_slack_messages(30)

# Filter to slider window in Python (no extra API call)
cutoff_ts = str((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
messages = [m for m in all_messages_180d if m.get("ts", "0") >= cutoff_ts]

classified: dict[str, list[dict]] = {
    "auto_eval_reserve": [],
    "bpo_cr_escalation": [],
    "capital_escalation": [],
}
for m in messages:
    cat = classify_message(m)
    if cat:
        classified[cat].append(m)

# ---------------------------------------------------------------------------
# Collect ALL tokens from 180-day window (stable cache key for Snowflake)
# ---------------------------------------------------------------------------
all_unique_tokens: set[str] = set()
for m in all_messages_180d:
    cat = classify_message(m)
    if cat:
        token = extract_token_from_message(m.get("text", ""))
        if token:
            all_unique_tokens.add(token)

# Identify SFS→CR escalation threads for thread reply scanning (slider window only)
cap_msgs_with_threads = [
    m for m in messages
    if classify_message(m) == "capital_escalation" and int(m.get("reply_count", 0)) > 0
]
thread_ts_list = tuple(m["ts"] for m in cap_msgs_with_threads) if cap_msgs_with_threads else ()

# Extract watch tokens from full 180-day messages
watch_tokens: set[str] = set()
watch_tokens_recent: set[str] = set()
for m in all_messages_180d:
    reactions = m.get("reactions", [])
    if any(r.get("name") == "watch" for r in reactions):
        token = extract_token_from_message(m.get("text", ""))
        if token:
            watch_tokens.add(token)
            if m.get("ts", "0") >= cutoff_ts:
                watch_tokens_recent.add(token)

# Include watch tokens so Snowflake returns their loan data
all_unique_tokens |= watch_tokens

df_loan_perf_all = pd.DataFrame()
df_denied_apps_all = pd.DataFrame()
mcc_map: dict[str, str] = {}

# Kick off thread replies in background NOW so they fetch in parallel with Snowflake
_thread_future = None
if thread_ts_list and "thread_replies" not in st.session_state:
    _bg_pool = ThreadPoolExecutor(max_workers=1)
    _thread_future = _bg_pool.submit(fetch_thread_replies, thread_ts_list)

# Fetch Snowflake data (runs in parallel with thread replies above)
if all_unique_tokens:
    with st.spinner("Fetching loan data..."):
        sorted_tokens = tuple(sorted(all_unique_tokens))
        df_loan_perf_all, df_denied_apps_all, mcc_map = fetch_all_snowflake_data(sorted_tokens, 180)

# Filter denied apps to slider window in Python
df_denied_apps = pd.DataFrame()
if not df_denied_apps_all.empty:
    sub_col = _col(df_denied_apps_all, "SUBMITTED_AT")
    if sub_col in df_denied_apps_all.columns:
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        df_denied_apps = df_denied_apps_all[df_denied_apps_all[sub_col].astype(str) >= cutoff_date]
    else:
        df_denied_apps = df_denied_apps_all


# ---------------------------------------------------------------------------
# TAB 1 — Escalation Monitor
# ---------------------------------------------------------------------------
with tab_esc:
    st.header("Escalation Monitor")

    auto_msgs = classified["auto_eval_reserve"]
    cap_msgs = classified["capital_escalation"]

    bpo_msgs = classified["bpo_cr_escalation"]
    st.metric("Total Escalations", len(auto_msgs) + len(cap_msgs) + len(bpo_msgs))

    st.divider()

    # Build combined daily series
    def daily_series(msgs, label):
        dates = [
            datetime.fromtimestamp(float(m["ts"]), tz=timezone.utc).date()
            for m in msgs
        ]
        counts = Counter(dates)
        return pd.DataFrame([
            {"Date": d, "Count": c, "Type": label}
            for d, c in sorted(counts.items())
        ])

    df_esc = pd.concat([
        daily_series(auto_msgs, "CR Escalations to Pause Loan Eligibility"),
        daily_series(cap_msgs, "SFS Loan Ops → CR"),
        daily_series(bpo_msgs, "CR Requests to Place on Watch"),
    ], ignore_index=True)

    if not df_esc.empty:
        fig = px.bar(
            df_esc, x="Date", y="Count", color="Type", barmode="group",
            title="Daily Escalation Volume",
            color_discrete_map={
                "CR Escalations to Pause Loan Eligibility": "#f58518",
                "SFS Loan Ops → CR": "#4c78a8",
                "CR Requests to Place on Watch": "#e45756",
            },
        )
        st.plotly_chart(fig, width="stretch")

    # ---- MCC Distribution ----
    all_esc_tokens = set()
    for m in auto_msgs + cap_msgs + classified["bpo_cr_escalation"]:
        t = extract_token_from_message(m.get("text", ""))
        if t:
            all_esc_tokens.add(t)

    mcc_data = [mcc_map[t] for t in all_esc_tokens if t in mcc_map]
    if mcc_data:
        mcc_counts = pd.Series(mcc_data).value_counts().reset_index()
        mcc_counts.columns = ["MCC", "Units"]
        fig_mcc = px.pie(
            mcc_counts, names="MCC", values="Units",
            title=f"Escalated Sellers by MCC ({len(mcc_data)} units)",
            hole=0.35,
        )
        fig_mcc.update_traces(textposition="none", textinfo="none")
        total = mcc_counts["Units"].sum()
        mcc_counts["Pct"] = (mcc_counts["Units"] / total * 100).round(1)
        mcc_counts["Legend"] = mcc_counts["MCC"] + " (" + mcc_counts["Pct"].astype(str) + "%)"
        fig_mcc.for_each_trace(lambda t: t.update(
            labels=mcc_counts["Legend"].tolist(),
        ))
        fig_mcc.update_traces(hovertemplate="%{label}<extra></extra>")
        fig_mcc.update_layout(legend=dict(orientation="v", x=1.05, y=0.5))
        st.plotly_chart(fig_mcc, width="stretch")

    # Helper: build escalation table
    def build_escalation_table(msgs, df_perf):
        # Compute total outstanding from ALL messages (not capped)
        total_outstanding = 0.0
        if msgs and not df_perf.empty:
            all_msg_tokens = set()
            for m in msgs:
                t = extract_token_from_message(m.get("text", ""))
                if t:
                    all_msg_tokens.add(t)
            if all_msg_tokens:
                tok_col = _col(df_perf, "UNIT_TOKEN")
                out_col = _col(df_perf, "OUTSTANDING_DOLLARS")
                if tok_col in df_perf.columns:
                    matched = df_perf[df_perf[tok_col].astype(str).isin(all_msg_tokens)]
                    if not matched.empty:
                        total_outstanding = pd.to_numeric(matched[out_col], errors="coerce").sum()

        # Build table rows (capped at 50 for display)
        rows = []
        for m in sorted(msgs, key=lambda x: x["ts"], reverse=True)[:50]:
            ts = datetime.fromtimestamp(float(m["ts"]), tz=timezone.utc)
            token = extract_token_from_message(m.get("text", ""))
            row = {
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M UTC"),
                "Unit Token": f"[{token}]({regulator_url(token)})" if token else "—",
                "MCC": mcc_map.get(token, "—") if token else "—",
            }
            if token and not df_perf.empty:
                tok_col = _col(df_perf, "UNIT_TOKEN")
                unit_plans = df_perf[df_perf[tok_col].astype(str) == token] if tok_col in df_perf.columns else pd.DataFrame()
                if not unit_plans.empty:
                    p = unit_plans.iloc[0]
                    row["Loan Status"] = p.get(_col(unit_plans, "STATUS"), "—")
                    row["Product"] = p.get(_col(unit_plans, "PRODUCT_NAME"), "—")
                    row["Risk Grade"] = p.get(_col(unit_plans, "RISK_GRADE_NAME"), "—")
                    out_val = pd.to_numeric(unit_plans[_col(unit_plans, "OUTSTANDING_DOLLARS")], errors="coerce").sum()
                    row["Outstanding $"] = f"${out_val:,.0f}"
                else:
                    row.update({"Loan Status": "—", "Product": "—", "Risk Grade": "—", "Outstanding $": "—"})
            else:
                row.update({"Loan Status": "—", "Product": "—", "Risk Grade": "—", "Outstanding $": "—"})
            rows.append(row)
        return pd.DataFrame(rows), total_outstanding

    # Recent escalations — SFS Loan Ops → CR
    st.subheader("Recent SFS Loan Ops → CR Escalations")
    if cap_msgs:
        df_cap, cap_outstanding = build_escalation_table(cap_msgs, df_loan_perf_all)
        st.metric("Total Outstanding $", f"${cap_outstanding:,.0f}")
        st.dataframe(
            df_cap, width="stretch", hide_index=True,
            column_config={"Unit Token": st.column_config.LinkColumn(display_text=r"\[(.*?)\]")},
        )
    else:
        st.info("No SFS Loan Ops → CR escalations in the selected window.")

    # CR Escalations to Pause Loan Eligibility
    st.subheader("CR Escalations to Pause Loan Eligibility")
    if auto_msgs:
        df_auto, auto_outstanding = build_escalation_table(auto_msgs, df_loan_perf_all)
        st.metric("Total Outstanding $", f"${auto_outstanding:,.0f}")
        st.dataframe(
            df_auto, width="stretch", hide_index=True,
            column_config={"Unit Token": st.column_config.LinkColumn(display_text=r"\[(.*?)\]")},
        )
    else:
        st.info("No 'Auto Evaluation recommends reserve' messages found in the selected window.")

    # CR → SFS Loan Ops (Credit Risk Submission)
    st.subheader("CR Requests to Place on Watch")
    if bpo_msgs:
        df_bpo, bpo_outstanding = build_escalation_table(bpo_msgs, df_loan_perf_all)
        st.metric("Total Outstanding $", f"${bpo_outstanding:,.0f}")
        st.dataframe(
            df_bpo, width="stretch", hide_index=True,
            column_config={"Unit Token": st.column_config.LinkColumn(display_text=r"\[(.*?)\]")},
        )
    else:
        st.info("No CR requests to place on watch in the selected window.")


# ---------------------------------------------------------------------------
# TAB 2 — Loan Loss Mitigation
# ---------------------------------------------------------------------------
with tab_miti:
    st.header("Loan Loss Mitigation")

    # Collect thread replies (started in background earlier, parallel with Snowflake)
    if "thread_replies" not in st.session_state or refresh:
        if _thread_future is not None:
            with st.spinner("Analyzing thread replies..."):
                st.session_state["thread_replies"] = _thread_future.result()
        elif thread_ts_list:
            with st.spinner("Analyzing thread replies..."):
                st.session_state["thread_replies"] = fetch_thread_replies(thread_ts_list)
        else:
            st.session_state["thread_replies"] = {}

    all_replies = st.session_state["thread_replies"]

    if "flagged_thread_ts" not in st.session_state or refresh:
        flagged_thread_ts: set[str] = set()
        if all_replies:
            flagged_thread_ts = classify_no_extend_threads(thread_ts_list, all_replies)
        st.session_state["flagged_thread_ts"] = flagged_thread_ts
    flagged_thread_ts = st.session_state["flagged_thread_ts"]

    # Map flagged threads back to tokens
    no_extend_tokens = set()
    for m in cap_msgs_with_threads:
        if m["ts"] in flagged_thread_ts:
            token = extract_token_from_message(m.get("text", ""))
            if token:
                no_extend_tokens.add(token)

    using_llm = _get_anthropic_client() is not None
    method_note = "LLM-classified" if using_llm else "pattern-matched"
    st.caption(f"Escalated sellers at high risk of CR controls being placed based on thread reply language ({method_note}).")
    st.metric("Loan Escalations Flagged as High Risk by CR", len(no_extend_tokens))

    if no_extend_tokens:
        # Declined application amounts for sellers CR flagged (date-filtered in SQL, deduped here)
        total_denied = 0.0
        if not df_denied_apps.empty:
            dn_tok = _col(df_denied_apps, "UNIT_TOKEN")
            dn_col = _col(df_denied_apps, "DENIED_DOLLARS")
            sub_col = _col(df_denied_apps, "SUBMITTED_AT")
            if dn_tok in df_denied_apps.columns and dn_col in df_denied_apps.columns:
                df_denied = df_denied_apps[df_denied_apps[dn_tok].astype(str).isin(no_extend_tokens)].copy()
                # Deduplicate: one decline per seller (most recent)
                if not df_denied.empty and sub_col in df_denied.columns:
                    df_denied = df_denied.sort_values(sub_col, ascending=False).drop_duplicates(subset=[dn_tok], keep="first")
                total_denied = pd.to_numeric(df_denied[dn_col], errors="coerce").sum()

        st.metric("Declined Loan $ for Flagged Sellers", f"${total_denied:,.0f}")
        st.caption("Declined applications submitted by flagged sellers — may include declines unrelated to CR escalation.")

    else:
        st.info("No escalations with CR 'do not extend' recommendations found in the selected window.")

    # ---- Sellers Placed on Watch ----
    st.divider()
    st.caption("Escalated sellers placed on watch. Active until loan is fully repaid.")
    st.metric("Sellers Placed on Watch", len(watch_tokens_recent), f"{len(watch_tokens)} total monitored")

    if watch_tokens and not df_loan_perf_all.empty:
        tok_col = _col(df_loan_perf_all, "UNIT_TOKEN")
        out_col = _col(df_loan_perf_all, "OUTSTANDING_DOLLARS")
        fin_col = _col(df_loan_perf_all, "FINANCED_DOLLARS")
        if tok_col in df_loan_perf_all.columns:
            watch_perf = df_loan_perf_all[df_loan_perf_all[tok_col].astype(str).isin(watch_tokens)]
            if not watch_perf.empty:
                watch_agg = watch_perf.groupby(tok_col).agg(
                    financed=(fin_col, lambda x: pd.to_numeric(x, errors="coerce").sum()),
                    outstanding=(out_col, lambda x: pd.to_numeric(x, errors="coerce").sum()),
                ).reset_index()
                watch_agg.columns = ["TOKEN", "FINANCED", "OUTSTANDING"]

                active = watch_agg[watch_agg["OUTSTANDING"] > 0]
                repaid = watch_agg[watch_agg["OUTSTANDING"] <= 0]

                w1, w2 = st.columns(2)
                w1.metric("Active Watch — Outstanding ($)", f"${float(active['OUTSTANDING'].sum()):,.0f}", f"{len(active)} sellers")
                w2.metric("Watch Loans Repaid ($)", f"${float(repaid['FINANCED'].sum()):,.0f}", f"{len(repaid)} sellers")
            else:
                st.info("No loan data found for watched sellers.")
    elif watch_tokens:
        st.info("No loan performance data available.")
    else:
        st.info("No sellers placed on watch in the selected window.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Data refreshes every 15 min (Slack) / 30 min (Snowflake). "
    f"Last loaded: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)
