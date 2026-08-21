"""Modern glassmorphism dashboard for compression stats."""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legroom</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-primary: #0a0a0f;
            --bg-card: rgba(255, 255, 255, 0.03);
            --bg-card-hover: rgba(255, 255, 255, 0.06);
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.15);
            --text-primary: #f0f0f5;
            --text-secondary: rgba(240, 240, 245, 0.6);
            --text-tertiary: rgba(240, 240, 245, 0.35);
            --accent-blue: #6c8aff;
            --accent-blue-glow: rgba(108, 138, 255, 0.25);
            --accent-green: #34d399;
            --accent-green-glow: rgba(52, 211, 153, 0.25);
            --accent-purple: #a78bfa;
            --accent-purple-glow: rgba(167, 139, 250, 0.25);
            --accent-amber: #fbbf24;
            --accent-amber-glow: rgba(251, 191, 36, 0.25);
            --accent-red: #f87171;
            --accent-red-glow: rgba(248, 113, 113, 0.25);
            --glass-blur: 20px;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated gradient background */
        .bg-gradient {
            position: fixed;
            inset: 0;
            z-index: 0;
            background:
                radial-gradient(ellipse 80% 50% at 20% 10%, rgba(108, 138, 255, 0.08) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 80%, rgba(167, 139, 250, 0.06) 0%, transparent 60%),
                radial-gradient(ellipse 50% 30% at 50% 50%, rgba(52, 211, 153, 0.04) 0%, transparent 60%);
            animation: bgShift 20s ease-in-out infinite alternate;
        }

        @keyframes bgShift {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.05); }
            100% { opacity: 1; transform: scale(1); }
        }

        /* Floating orbs */
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.15;
            pointer-events: none;
            z-index: 0;
        }
        .orb-1 { width: 400px; height: 400px; background: var(--accent-blue); top: -100px; right: -100px; animation: orbFloat1 25s ease-in-out infinite; }
        .orb-2 { width: 300px; height: 300px; background: var(--accent-purple); bottom: -50px; left: -50px; animation: orbFloat2 30s ease-in-out infinite; }
        .orb-3 { width: 250px; height: 250px; background: var(--accent-green); top: 50%; left: 50%; animation: orbFloat3 20s ease-in-out infinite; }

        @keyframes orbFloat1 { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(-60px, 80px); } }
        @keyframes orbFloat2 { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(80px, -60px); } }
        @keyframes orbFloat3 { 0%, 100% { transform: translate(-50%, -50%); } 50% { transform: translate(-30%, -70%); } }

        /* Layout */
        .app {
            position: relative;
            z-index: 1;
            max-width: 1320px;
            margin: 0 auto;
            padding: 0 24px;
        }

        /* Header */
        .header {
            position: sticky;
            top: 0;
            z-index: 100;
            margin: 0 -24px;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            gap: 16px;
            background: rgba(10, 10, 15, 0.7);
            backdrop-filter: blur(var(--glass-blur));
            -webkit-backdrop-filter: blur(var(--glass-blur));
            border-bottom: 1px solid var(--border);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logo-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            font-weight: 700;
            box-shadow: 0 4px 12px var(--accent-blue-glow);
        }

        .logo-text {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .header-spacer { flex: 1; }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            background: var(--bg-card);
            border: 1px solid var(--border);
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 8px var(--accent-green-glow);
            animation: pulse 2s ease-in-out infinite;
        }

        .status-dot.disconnected {
            background: var(--accent-red);
            box-shadow: 0 0 8px var(--accent-red-glow);
            animation: none;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .header-time {
            font-size: 13px;
            color: var(--text-secondary);
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin: 28px 0;
        }

        .stat-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.02));
            pointer-events: none;
        }

        .stat-card:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-hover);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }

        .stat-card .icon {
            font-size: 18px;
            margin-bottom: 10px;
        }

        .stat-card .label {
            font-size: 11px;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }

        .stat-card .value {
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1;
        }

        .stat-card .value.blue { color: var(--accent-blue); text-shadow: 0 0 20px var(--accent-blue-glow); }
        .stat-card .value.green { color: var(--accent-green); text-shadow: 0 0 20px var(--accent-green-glow); }
        .stat-card .value.purple { color: var(--accent-purple); text-shadow: 0 0 20px var(--accent-purple-glow); }
        .stat-card .value.amber { color: var(--accent-amber); text-shadow: 0 0 20px var(--accent-amber-glow); }
        .stat-card .value.red { color: var(--accent-red); text-shadow: 0 0 20px var(--accent-red-glow); }

        .stat-card .sub {
            font-size: 12px;
            color: var(--text-tertiary);
            margin-top: 8px;
        }

        .stat-card .mini-bar {
            height: 4px;
            background: rgba(255,255,255,0.05);
            border-radius: 2px;
            margin-top: 12px;
            overflow: hidden;
        }

        .stat-card .mini-bar-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .stat-card .mini-bar-fill.blue { background: linear-gradient(90deg, var(--accent-blue), rgba(108, 138, 255, 0.4)); }
        .stat-card .mini-bar-fill.green { background: linear-gradient(90deg, var(--accent-green), rgba(52, 211, 153, 0.4)); }
        .stat-card .mini-bar-fill.purple { background: linear-gradient(90deg, var(--accent-purple), rgba(167, 139, 250, 0.4)); }

        /* Section */
        .section {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 14px;
            margin-bottom: 20px;
            overflow: hidden;
        }

        .section-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-header h2 {
            font-size: 14px;
            font-weight: 600;
        }

        .section-header .count {
            font-size: 11px;
            font-weight: 500;
            color: var(--text-secondary);
            background: rgba(255,255,255,0.05);
            padding: 2px 8px;
            border-radius: 10px;
        }

        .section-header .controls {
            margin-left: auto;
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .search-input {
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 13px;
            color: var(--text-primary);
            font-family: inherit;
            width: 180px;
            outline: none;
            transition: all 0.2s ease;
        }

        .search-input::placeholder { color: var(--text-tertiary); }
        .search-input:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px var(--accent-blue-glow);
        }

        /* Chart */
        .chart-container {
            padding: 20px;
            height: 220px;
            position: relative;
        }

        .chart-empty {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-tertiary);
            font-size: 14px;
        }

        canvas { display: block; }

        /* Table */
        .table-wrap { overflow-x: auto; }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        thead th {
            padding: 10px 16px;
            font-size: 11px;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            text-align: left;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
            position: sticky;
            top: 0;
            background: rgba(10, 10, 15, 0.8);
            backdrop-filter: blur(10px);
        }

        tbody td {
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            font-size: 13px;
            white-space: nowrap;
        }

        tbody tr {
            cursor: pointer;
            transition: background 0.15s ease;
        }

        tbody tr:hover {
            background: rgba(255,255,255,0.03);
        }

        tbody tr:last-child td { border-bottom: none; }

        .tag {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 500;
            margin-right: 4px;
            border: 1px solid;
        }

        .tag.compress { color: var(--accent-blue); background: rgba(108,138,255,0.1); border-color: rgba(108,138,255,0.2); }
        .tag.dedup { color: var(--text-secondary); background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.08); }
        .tag.lifecycle { color: var(--accent-red); background: rgba(248,113,113,0.1); border-color: rgba(248,113,113,0.2); }
        .tag.ansi { color: var(--accent-amber); background: rgba(251,191,36,0.1); border-color: rgba(251,191,36,0.2); }
        .tag.json { color: var(--accent-green); background: rgba(52,211,153,0.1); border-color: rgba(52,211,153,0.2); }
        .tag.read { color: var(--accent-purple); background: rgba(167,139,250,0.1); border-color: rgba(167,139,250,0.2); }

        .saved-positive { color: var(--accent-green); font-weight: 600; }
        .saved-negative { color: var(--accent-red); font-weight: 600; }
        .saved-zero { color: var(--text-secondary); }

        .model-badge {
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            background: rgba(167,139,250,0.1);
            color: var(--accent-purple);
            border: 1px solid rgba(167,139,250,0.15);
        }

        /* Breakdown */
        .breakdown-bar {
            height: 6px;
            border-radius: 3px;
            background: rgba(255,255,255,0.05);
            overflow: hidden;
            margin-top: 6px;
        }

        .breakdown-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }

        /* Lifecycle */
        .lifecycle-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            padding: 20px;
        }

        .lifecycle-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .lifecycle-card:hover {
            background: rgba(255,255,255,0.04);
        }

        .lifecycle-card .state {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        .lifecycle-card .count-val {
            font-size: 22px;
            font-weight: 700;
        }

        .lifecycle-card .pct {
            font-size: 12px;
            color: var(--text-tertiary);
            margin-top: 4px;
        }

        .lifecycle-card.stale { border-color: rgba(248,113,113,0.2); }
        .lifecycle-card.stale .count-val { color: var(--accent-red); }
        .lifecycle-card.superseded { border-color: rgba(251,191,36,0.2); }
        .lifecycle-card.superseded .count-val { color: var(--accent-amber); }
        .lifecycle-card.fresh { border-color: rgba(52,211,153,0.2); }
        .lifecycle-card.fresh .count-val { color: var(--accent-green); }

        /* Empty state */
        .empty-state {
            padding: 48px 20px;
            text-align: center;
            color: var(--text-tertiary);
            font-size: 13px;
        }

        .empty-state .emoji {
            font-size: 28px;
            margin-bottom: 8px;
            display: block;
        }

        /* Modal */
        .modal-overlay {
            position: fixed;
            inset: 0;
            z-index: 1000;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            display: none;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.2s ease;
        }

        .modal-overlay.active { display: flex; }

        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

        .modal {
            background: rgba(20, 20, 30, 0.95);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            max-width: 640px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            animation: slideUp 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 24px 64px rgba(0,0,0,0.4);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px) scale(0.97); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        .modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        .modal-header h3 {
            font-size: 15px;
            font-weight: 600;
        }

        .modal-close {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--bg-card);
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: all 0.15s ease;
        }

        .modal-close:hover {
            background: var(--bg-card-hover);
            color: var(--text-primary);
        }

        .modal pre {
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            font-size: 12px;
            font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
            line-height: 1.6;
            overflow-x: auto;
            color: var(--text-secondary);
        }

        /* Refresh FAB */
        .fab {
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 50;
            width: 48px;
            height: 48px;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: rgba(20, 20, 30, 0.8);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }

        .fab:hover {
            background: rgba(108, 138, 255, 0.15);
            border-color: var(--accent-blue);
            color: var(--accent-blue);
            box-shadow: 0 4px 20px var(--accent-blue-glow);
        }

        .fab:active { transform: scale(0.92); }

        .fab.spinning { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

        /* Responsive */
        @media (max-width: 768px) {
            .app { padding: 0 12px; }
            .header { margin: 0 -12px; padding: 12px; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
            .stat-card { padding: 16px; }
            .stat-card .value { font-size: 20px; }
            .lifecycle-grid { grid-template-columns: 1fr; }
            .section-header .controls { flex-wrap: wrap; }
            .search-input { width: 100%; }
        }

        @media (max-width: 480px) {
            .stats-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <!-- Background effects -->
    <div class="bg-gradient"></div>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>

    <div class="app">
        <!-- Header -->
        <header class="header">
            <div class="logo">
                <div class="logo-icon">L</div>
                <span class="logo-text">Legroom</span>
            </div>
            <div class="header-spacer"></div>
            <div class="status-badge">
                <span class="status-dot" id="status-dot"></span>
                <span id="status-text">Live</span>
            </div>
            <span class="header-time" id="header-time">—</span>
        </header>

        <!-- Stats -->
        <div class="stats-grid" id="stats-grid">
            <div class="stat-card">
                <div class="icon">📊</div>
                <div class="label">Requests</div>
                <div class="value blue" id="s-requests">0</div>
            </div>
            <div class="stat-card">
                <div class="icon">💾</div>
                <div class="label">Tokens Saved</div>
                <div class="value green" id="s-saved">0</div>
            </div>
            <div class="stat-card">
                <div class="icon">📉</div>
                <div class="label">Compression</div>
                <div class="value amber" id="s-ratio">0%</div>
                <div class="mini-bar"><div class="mini-bar-fill amber" id="s-ratio-bar" style="width:0%"></div></div>
            </div>
            <div class="stat-card">
                <div class="icon">🧮</div>
                <div class="label">Avg Saved</div>
                <div class="value purple" id="s-avg">0</div>
            </div>
            <div class="stat-card">
                <div class="icon">🔄</div>
                <div class="label">Reads Compressed</div>
                <div class="value blue" id="s-reads">0</div>
            </div>
            <div class="stat-card">
                <div class="icon">📦</div>
                <div class="label">CCR Stored</div>
                <div class="value purple" id="s-ccr">0</div>
            </div>
        </div>

        <!-- Chart -->
        <div class="section">
            <div class="section-header">
                <h2>Token Savings</h2>
                <span class="count" id="chart-count">0</span>
            </div>
            <div class="chart-container">
                <canvas id="chart"></canvas>
                <div class="chart-empty" id="chart-empty">No data yet</div>
            </div>
        </div>

        <!-- History -->
        <div class="section">
            <div class="section-header">
                <h2>Request History</h2>
                <span class="count" id="history-count">0</span>
                <div class="controls">
                    <input class="search-input" type="text" id="search-input" placeholder="Search models...">
                </div>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Model</th>
                            <th>Before</th>
                            <th>After</th>
                            <th>Saved</th>
                            <th>Transforms</th>
                        </tr>
                    </thead>
                    <tbody id="history-body">
                        <tr><td colspan="6" class="empty-state"><span class="emoji">📭</span>No requests yet</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Breakdown -->
        <div class="section">
            <div class="section-header">
                <h2>Strategy Breakdown</h2>
            </div>
            <div id="breakdown-body" style="padding: 20px;"></div>
        </div>

        <!-- Lifecycle -->
        <div class="section">
            <div class="section-header">
                <h2>Read Lifecycle</h2>
            </div>
            <div class="lifecycle-grid" id="lifecycle-body">
                <div class="empty-state" style="grid-column: 1/-1;"><span class="emoji">🔄</span>No lifecycle data</div>
            </div>
        </div>
    </div>

    <!-- Modal -->
    <div class="modal-overlay" id="modal">
        <div class="modal">
            <div class="modal-header">
                <h3 id="modal-title">Details</h3>
                <button class="modal-close" onclick="closeModal()">✕</button>
            </div>
            <pre id="modal-content"></pre>
        </div>
    </div>

    <!-- FAB -->
    <button class="fab" onclick="refreshData()" title="Refresh" id="fab">↻</button>

    <script>
        const CHART_POINTS = 30;
        const chartData = { labels: [], before: [], after: [], saved: [] };
        let ws = null, evtSource = null, connected = false;
        let refreshTimer = null;

        // ── Chart ──────────────────────────────────────────────
        const canvas = document.getElementById('chart');
        const ctx = canvas.getContext('2d');
        let dpr = window.devicePixelRatio || 1;

        function resizeCanvas() {
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = 220 * dpr;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = '220px';
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        function drawChart() {
            resizeCanvas();
            const w = canvas.width / dpr;
            const h = canvas.height / dpr;
            const pad = { top: 20, right: 20, bottom: 30, left: 55 };
            const cw = w - pad.left - pad.right;
            const ch = h - pad.top - pad.bottom;

            ctx.clearRect(0, 0, w, h);

            const empty = document.getElementById('chart-empty');

            if (chartData.saved.length === 0) {
                empty.style.display = 'flex';
                return;
            }
            empty.style.display = 'none';

            const maxVal = Math.max(...chartData.before, 1);
            const n = chartData.labels.length;

            // Grid lines
            ctx.strokeStyle = 'rgba(255,255,255,0.04)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = pad.top + (ch / 4) * i;
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(w - pad.right, y);
                ctx.stroke();

                // Y labels
                const val = Math.round(maxVal * (1 - i / 4));
                ctx.fillStyle = 'rgba(240,240,245,0.3)';
                ctx.font = '11px Inter, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(val.toLocaleString(), pad.left - 8, y + 4);
            }

            const barW = Math.max(3, (cw / n) - 4);
            const gap = 2;

            // Bars with gradient
            for (let i = 0; i < n; i++) {
                const x = pad.left + (i / n) * cw + gap;
                const beforeH = (chartData.before[i] / maxVal) * ch;
                const afterH = (chartData.after[i] / maxVal) * ch;

                // Before (full) gradient
                const grad1 = ctx.createLinearGradient(x, pad.top + ch - beforeH, x, pad.top + ch);
                grad1.addColorStop(0, 'rgba(108, 138, 255, 0.35)');
                grad1.addColorStop(1, 'rgba(108, 138, 255, 0.08)');
                ctx.fillStyle = grad1;
                ctx.beginPath();
                ctx.roundRect(x, pad.top + ch - beforeH, barW, beforeH, [3, 3, 0, 0]);
                ctx.fill();

                // After (compressed) gradient
                const grad2 = ctx.createLinearGradient(x, pad.top + ch - afterH, x, pad.top + ch);
                grad2.addColorStop(0, 'rgba(52, 211, 153, 0.35)');
                grad2.addColorStop(1, 'rgba(52, 211, 153, 0.08)');
                ctx.fillStyle = grad2;
                ctx.beginPath();
                ctx.roundRect(x, pad.top + ch - afterH, barW, afterH, [3, 3, 0, 0]);
                ctx.fill();
            }

            // Legend
            ctx.font = '11px Inter, sans-serif';
            const lx = w - pad.right - 160;
            ctx.fillStyle = 'rgba(108, 138, 255, 0.5)';
            ctx.fillRect(lx, 6, 10, 10);
            ctx.fillStyle = 'rgba(240,240,245,0.5)';
            ctx.textAlign = 'left';
            ctx.fillText('Before', lx + 14, 15);

            ctx.fillStyle = 'rgba(52, 211, 153, 0.5)';
            ctx.fillRect(lx + 90, 6, 10, 10);
            ctx.fillStyle = 'rgba(240,240,245,0.5)';
            ctx.fillText('After', lx + 104, 15);
        }

        // ── Data ───────────────────────────────────────────────
        async function refreshData() {
            const fab = document.getElementById('fab');
            fab.classList.add('spinning');
            try {
                const [stats, history, lifecycle, ccr] = await Promise.all([
                    fetch('/api/stats').then(r => r.json()),
                    fetch('/api/history?limit=100').then(r => r.json()),
                    fetch('/api/read-lifecycle').then(r => r.json()),
                    fetch('/api/ccr').then(r => r.json()),
                ]);
                updateStats(stats, ccr);
                updateHistory(history);
                updateLifecycle(lifecycle);
                document.getElementById('header-time').textContent = new Date(stats.last_updated * 1000).toLocaleTimeString();
            } catch (e) {
                console.error('Refresh failed:', e);
            } finally {
                fab.classList.remove('spinning');
            }
        }

        function updateStats(stats, ccr) {
            animate('s-requests', stats.total_requests);
            animate('s-saved', stats.total_tokens_saved);
            document.getElementById('s-ratio').textContent = stats.compression_ratio + '%';
            document.getElementById('s-ratio-bar').style.width = stats.compression_ratio + '%';
            animate('s-avg', stats.avg_tokens_saved);
            animate('s-reads', stats.total_reads_compressed);
            animate('s-ccr', stats.total_ccr_stored);

            // Breakdown
            const bd = document.getElementById('breakdown-body');
            bd.innerHTML = '';
            const total = Object.values(stats.strategy_counts).reduce((a, b) => a + b, 0);
            const colors = ['var(--accent-blue)', 'var(--accent-green)', 'var(--accent-purple)', 'var(--accent-amber)', 'var(--accent-red)'];
            let ci = 0;

            for (const [strategy, count] of Object.entries(stats.strategy_counts)) {
                if (!count) continue;
                const pct = total > 0 ? (count / total * 100) : 0;
                const color = colors[ci++ % colors.length];
                const tr = document.createElement('div');
                tr.style.cssText = 'display:flex;align-items:center;gap:12px;padding:8px 0;';
                tr.innerHTML = `
                    <div style="min-width:140px;font-size:13px;font-weight:500;">${strategy}</div>
                    <div style="flex:1;height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;">
                        <div style="height:100%;width:${pct}%;background:${color};border-radius:3px;transition:width 0.5s ease;"></div>
                    </div>
                    <div style="min-width:60px;text-align:right;font-size:13px;font-weight:600;">${count.toLocaleString()}</div>
                    <div style="min-width:40px;text-align:right;font-size:11px;color:var(--text-tertiary);">${pct.toFixed(0)}%</div>
                `;
                bd.appendChild(tr);
            }
        }

        function updateHistory(history) {
            const tbody = document.getElementById('history-body');
            document.getElementById('history-count').textContent = history.total;

            if (!history.history?.length) {
                tbody.innerHTML = '<tr><td colspan="6" class="empty-state"><span class="emoji">📭</span>No requests yet</td></tr>';
                document.getElementById('chart-count').textContent = '0';
                chartData.labels = [];
                chartData.before = [];
                chartData.after = [];
                chartData.saved = [];
                drawChart();
                return;
            }

            // Chart data (oldest first)
            const recent = history.history.slice().reverse().slice(-CHART_POINTS);
            chartData.labels = recent.map(r => r.model);
            chartData.before = recent.map(r => r.tokens_before);
            chartData.after = recent.map(r => r.tokens_after);
            chartData.saved = recent.map(r => r.tokens_saved);
            document.getElementById('chart-count').textContent = chartData.labels.length;
            drawChart();

            tbody.innerHTML = '';
            const query = document.getElementById('search-input').value.toLowerCase();

            history.history.forEach(req => {
                const modelMatch = !query || req.model.toLowerCase().includes(query);
                if (query && !modelMatch) return;

                const tr = document.createElement('tr');
                const t = new Date(req.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const saved = req.tokens_saved;
                const savedClass = saved > 0 ? 'saved-positive' : (saved < 0 ? 'saved-negative' : 'saved-zero');
                const savedText = saved > 0 ? `+${saved.toLocaleString()}` : saved.toLocaleString();

                const tags = (req.transforms_applied || []).map(t2 => {
                    let cls = 'compress';
                    if (t2.includes('dedup')) cls = 'dedup';
                    if (t2.includes('lifecycle')) cls = 'lifecycle';
                    if (t2.includes('ansi') || t2.includes('lossless')) cls = 'ansi';
                    if (t2.includes('json')) cls = 'json';
                    if (t2.includes('read')) cls = 'read';
                    return `<span class="tag ${cls}">${t2}</span>`;
                }).join('');

                tr.innerHTML = `
                    <td style="color:var(--text-secondary)">${t}</td>
                    <td><span class="model-badge">${req.model}</span></td>
                    <td>${req.tokens_before.toLocaleString()}</td>
                    <td>${req.tokens_after.toLocaleString()}</td>
                    <td class="${savedClass}">${savedText}</td>
                    <td>${tags}</td>
                `;
                tr.onclick = () => showDetail(req);
                tbody.appendChild(tr);
            });
        }

        function updateLifecycle(lifecycle) {
            const lb = document.getElementById('lifecycle-body');
            const total = Math.max((lifecycle.total_reads_stale || 0) +
                                   (lifecycle.total_reads_superseded || 0) +
                                   (lifecycle.total_reads_fresh || 0), 1);

            const states = [
                { label: 'Stale', key: 'stale', cls: 'stale', count: lifecycle.total_reads_stale || 0 },
                { label: 'Superseded', key: 'superseded', cls: 'superseded', count: lifecycle.total_reads_superseded || 0 },
                { label: 'Fresh', key: 'fresh', cls: 'fresh', count: lifecycle.total_reads_fresh || 0 },
            ];

            lb.innerHTML = '';
            let any = false;
            states.forEach(s => {
                if (s.count > 0) any = true;
                const pct = ((s.count / total) * 100).toFixed(1);
                const div = document.createElement('div');
                div.className = `lifecycle-card ${s.cls}`;
                div.innerHTML = `
                    <div class="state">${s.label}</div>
                    <div class="count-val">${s.count.toLocaleString()}</div>
                    <div class="pct">${pct}%</div>
                `;
                lb.appendChild(div);
            });

            if (!any) {
                lb.innerHTML = '<div class="empty-state" style="grid-column:1/-1;"><span class="emoji">🔄</span>No lifecycle data</div>';
            }
        }

        // ── Helpers ────────────────────────────────────────────
        function animate(id, target) {
            const el = document.getElementById(id);
            const current = parseInt(el.textContent.replace(/,/g, '')) || 0;
            if (current === target) { el.textContent = target.toLocaleString(); return; }
            const diff = target - current;
            const steps = 20;
            const stepVal = diff / steps;
            let i = 0;
            const timer = setInterval(() => {
                i++;
                el.textContent = Math.round(current + stepVal * i).toLocaleString();
                if (i >= steps) { el.textContent = target.toLocaleString(); clearInterval(timer); }
            }, 20);
        }

        function showDetail(req) {
            document.getElementById('modal-title').textContent = `${req.model} — ${req.request_id}`;
            document.getElementById('modal-content').textContent = JSON.stringify(req, null, 2);
            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        document.getElementById('modal').addEventListener('click', (e) => {
            if (e.target.id === 'modal') closeModal();
        });

        // Search
        document.getElementById('search-input').addEventListener('input', () => {
            refreshData();
        });

        // ── Live Connection ────────────────────────────────────
        function setStatus(online) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');
            if (online) {
                dot.classList.remove('disconnected');
                text.textContent = 'Live';
            } else {
                dot.classList.add('disconnected');
                text.textContent = 'Disconnected';
            }
        }

        function connectWebSocket() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws/events`);
            ws.onopen = () => { connected = true; setStatus(true); };
            ws.onmessage = () => refreshData();
            ws.onclose = () => { connected = false; setStatus(false); connectSSE(); };
        }

        function connectSSE() {
            try {
                evtSource = new EventSource('/api/events');
                evtSource.onmessage = (e) => {
                    if (e.data === ': keepalive') return;
                    refreshData();
                };
                evtSource.onerror = () => { evtSource.close(); evtSource = null; setStatus(false); };
            } catch {}
        }

        // ── Init ───────────────────────────────────────────────
        refreshData();
        connectWebSocket();
        setInterval(refreshData, 30000); // Fallback poll every 30s
        window.addEventListener('resize', drawChart);
    </script>
</body>
</html>"""


def get_dashboard_html() -> str:
    """Return the dashboard HTML content."""
    return DASHBOARD_HTML
