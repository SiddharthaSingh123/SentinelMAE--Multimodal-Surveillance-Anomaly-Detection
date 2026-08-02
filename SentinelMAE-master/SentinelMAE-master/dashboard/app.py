"""
dashboard/app.py
Flask web dashboard for SentinelMAE incident visualization.

Usage:
    pip install flask
    python dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, render_template_string, jsonify, request
from database.db_handler import init_db, get_incidents, get_stats
from chatbot.query_parser import parse_query

app = Flask(__name__)

# ── HTML Template ──────────────────────────────────────────────────────────

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SentinelMAE Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; }

    header {
      background: #1a1d2e;
      padding: 18px 32px;
      display: flex;
      align-items: center;
      gap: 12px;
      border-bottom: 1px solid #2a2d3e;
    }
    header h1 { font-size: 1.4rem; color: #fff; }
    header span { font-size: 0.85rem; color: #888; margin-left: auto; }

    .grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      padding: 24px 32px 0;
    }
    .stat-card {
      background: #1a1d2e;
      border-radius: 10px;
      padding: 20px;
      border: 1px solid #2a2d3e;
    }
    .stat-card .label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .stat-card .value { font-size: 2rem; font-weight: 700; margin-top: 6px; color: #fff; }
    .stat-card.red .value   { color: #ff4d4d; }
    .stat-card.orange .value{ color: #ff9f43; }
    .stat-card.green .value { color: #2ecc71; }

    .charts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      padding: 16px 32px;
    }
    .chart-card {
      background: #1a1d2e;
      border-radius: 10px;
      padding: 20px;
      border: 1px solid #2a2d3e;
    }
    .chart-card h3 { font-size: 0.9rem; color: #aaa; margin-bottom: 14px; }

    .bottom {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
      padding: 0 32px 32px;
    }

    .incidents-table {
      background: #1a1d2e;
      border-radius: 10px;
      padding: 20px;
      border: 1px solid #2a2d3e;
      overflow-x: auto;
    }
    .incidents-table h3 { font-size: 0.9rem; color: #aaa; margin-bottom: 14px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th { color: #666; font-weight: 600; padding: 8px; text-align: left; border-bottom: 1px solid #2a2d3e; }
    td { padding: 8px; border-bottom: 1px solid #1e2130; }
    tr:hover td { background: #1e2234; }
    .badge {
      padding: 2px 8px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 600;
    }
    .CRITICAL { background: #ff4d4d22; color: #ff4d4d; }
    .HIGH     { background: #ff9f4322; color: #ff9f43; }
    .MEDIUM   { background: #ffd70022; color: #ffd700; }
    .LOW      { background: #2ecc7122; color: #2ecc71; }
    .NORMAL   { background: #88888822; color: #888; }

    .chatbot {
      background: #1a1d2e;
      border-radius: 10px;
      padding: 20px;
      border: 1px solid #2a2d3e;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .chatbot h3 { font-size: 0.9rem; color: #aaa; }
    .chat-log {
      flex: 1;
      min-height: 200px;
      max-height: 300px;
      overflow-y: auto;
      background: #0f1117;
      border-radius: 8px;
      padding: 10px;
      font-size: 0.8rem;
      line-height: 1.6;
    }
    .chat-log .user { color: #7eb8f7; }
    .chat-log .bot  { color: #2ecc71; white-space: pre-wrap; }
    .chat-input { display: flex; gap: 8px; }
    .chat-input input {
      flex: 1;
      background: #0f1117;
      border: 1px solid #2a2d3e;
      border-radius: 6px;
      padding: 8px 12px;
      color: #fff;
      font-size: 0.83rem;
      outline: none;
    }
    .chat-input button {
      background: #5c6ef8;
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 8px 16px;
      cursor: pointer;
      font-size: 0.83rem;
    }
    .chat-input button:hover { background: #4a5ce6; }
  </style>
</head>
<body>

<header>
  <div style="width:32px;height:32px;background:#5c6ef8;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;">🛡</div>
  <h1>SentinelMAE</h1>
  <span>Real-time Anomaly Detection Dashboard</span>
</header>

<div class="grid" id="stats-grid">
  <div class="stat-card red"><div class="label">Total Incidents</div><div class="value" id="total">—</div></div>
  <div class="stat-card orange"><div class="label">Total Alerts</div><div class="value" id="alerts">—</div></div>
  <div class="stat-card red"><div class="label">Critical</div><div class="value" id="critical">—</div></div>
  <div class="stat-card green"><div class="label">Normal</div><div class="value" id="normal">—</div></div>
</div>

<div class="charts">
  <div class="chart-card"><h3>Incidents by Class</h3><canvas id="classChart"></canvas></div>
  <div class="chart-card"><h3>Risk Level Distribution</h3><canvas id="riskChart"></canvas></div>
</div>

<div class="bottom">
  <div class="incidents-table">
    <h3>Recent Incidents</h3>
    <table>
      <thead><tr><th>Time</th><th>Class</th><th>Risk</th><th>Confidence</th><th>Alert</th></tr></thead>
      <tbody id="incident-rows"></tbody>
    </table>
  </div>

  <div class="chatbot">
    <h3>🤖 Ask SentinelMAE</h3>
    <div class="chat-log" id="chat-log">
      <div class="bot">Hi! Ask me about incidents. Try:<br>• Show critical alerts<br>• How many robberies?<br>• Show fighting incidents</div>
    </div>
    <div class="chat-input">
      <input id="chat-input" type="text" placeholder="Ask a question..." onkeydown="if(event.key==='Enter')sendChat()"/>
      <button onclick="sendChat()">Send</button>
    </div>
  </div>
</div>

<script>
async function loadStats() {
  const res = await fetch('/api/stats');
  const d   = await res.json();
  document.getElementById('total').textContent    = d.total_incidents;
  document.getElementById('alerts').textContent   = d.total_alerts;
  document.getElementById('critical').textContent = d.by_risk_level?.CRITICAL || 0;
  document.getElementById('normal').textContent   = d.by_risk_level?.NORMAL   || 0;

  // Class chart
  const classes = Object.keys(d.top_classes || {});
  const counts  = Object.values(d.top_classes || {});
  new Chart(document.getElementById('classChart'), {
    type: 'bar',
    data: {
      labels: classes,
      datasets: [{ data: counts, backgroundColor: '#5c6ef8', borderRadius: 4 }]
    },
    options: { plugins: { legend: { display: false } }, scales: {
      x: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#1e2130' } },
      y: { ticks: { color: '#888' }, grid: { color: '#1e2130' } }
    }}
  });

  // Risk chart
  const risks  = Object.keys(d.by_risk_level || {});
  const rcounts = Object.values(d.by_risk_level || {});
  const colors  = { CRITICAL:'#ff4d4d', HIGH:'#ff9f43', MEDIUM:'#ffd700', LOW:'#2ecc71', NORMAL:'#888' };
  new Chart(document.getElementById('riskChart'), {
    type: 'doughnut',
    data: {
      labels: risks,
      datasets: [{ data: rcounts, backgroundColor: risks.map(r => colors[r] || '#5c6ef8') }]
    },
    options: { plugins: { legend: { labels: { color: '#aaa', font: { size: 11 } } } } }
  });
}

async function loadIncidents() {
  const res  = await fetch('/api/incidents?limit=20');
  const data = await res.json();
  const tbody = document.getElementById('incident-rows');
  tbody.innerHTML = data.map(i => `
    <tr>
      <td>${i.detected_at?.slice(0,19) || '—'}</td>
      <td>${i.predicted_class}</td>
      <td><span class="badge ${i.risk_level}">${i.risk_level}</span></td>
      <td>${(i.confidence*100).toFixed(1)}%</td>
      <td>${i.alert ? '⚠️' : '—'}</td>
    </tr>`).join('');
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const log   = document.getElementById('chat-log');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  log.innerHTML += `<div class="user">You: ${query}</div>`;
  const res  = await fetch('/api/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query}) });
  const data = await res.json();
  log.innerHTML += `<div class="bot">Bot: ${data.response}</div>`;
  log.scrollTop = log.scrollHeight;
}

loadStats();
loadIncidents();
setInterval(() => { loadStats(); loadIncidents(); }, 30000);
</script>
</body>
</html>
"""

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(TEMPLATE)


@app.route("/api/stats")
def api_stats():
    stats = get_stats()
    return jsonify(stats)


@app.route("/api/incidents")
def api_incidents():
    limit      = int(request.args.get("limit", 20))
    risk_level = request.args.get("risk_level")
    alert_only = request.args.get("alert_only", "false").lower() == "true"
    incidents  = get_incidents(risk_level=risk_level, alert_only=alert_only, limit=limit)
    return jsonify(incidents)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    from chatbot.retrieval import answer_query
    query    = request.json.get("query", "")
    response = answer_query(query)
    return jsonify({"response": response})


if __name__ == "__main__":
    init_db()
    print("\n── SentinelMAE Dashboard ──────────────────")
    print("  Open http://localhost:5000 in your browser")
    print("───────────────────────────────────────────\n")
    app.run(debug=True, port=5000)