import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WEB_DIR = os.path.join(BASE_DIR, "web")
ARTIFACT_DIR = r"C:\Users\akats\.gemini\antigravity\brain\e3933ebd-d891-42c9-86c4-f190bddb31a2"

with open(os.path.join(OUTPUT_DIR, "csv_b64_payloads.json"), "r", encoding="utf-8") as f:
    payloads = json.load(f)

js_download_helper = f"""
    const CSV_PAYLOADS = {json.dumps(payloads)};

    async function downloadUnifiedCSV(key, btnEl) {{
      const item = CSV_PAYLOADS[key];
      if (!item) return;
      const origText = btnEl.innerHTML;
      try {{
        btnEl.innerHTML = "⏳ 展開・準備中...";
        btnEl.disabled = true;
        const binStr = atob(item.b64);
        const len = binStr.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {{
          bytes[i] = binStr.charCodeAt(i);
        }}
        const ds = new DecompressionStream("gzip");
        const writer = ds.writable.getWriter();
        writer.write(bytes);
        writer.close();
        const decompressedBlob = await new Response(ds.readable).blob();
        const finalBlob = new Blob([decompressedBlob], {{ type: "text/csv;charset=utf-8;" }});
        const url = URL.createObjectURL(finalBlob);
        const a = document.createElement("a");
        a.href = url;
        a.download = item.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        btnEl.innerHTML = "✅ ダウンロード完了 (" + item.filename + ")";
        setTimeout(() => {{
          btnEl.innerHTML = origText;
          btnEl.disabled = false;
        }}, 2500);
      }} catch (err) {{
        console.error(err);
        btnEl.innerHTML = "❌ エラー: " + err.message;
        btnEl.disabled = false;
      }}
    }}
"""

# 1. Create compact inline widget `csv_download_widget.html` (< 360px height)
widget_html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-3">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-xl p-4 shadow-sm space-y-3">
    <div class="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] pb-2.5">
      <div>
        <h2 class="text-[var(--foreground)] font-bold text-base">📥 プロ野球データFreak × NPB公式ベストナイン 統合マスターCSVダウンロード (2015–2026年)</h2>
        <p class="text-[var(--muted-foreground)] text-xs">バラバラの成績表・日別スタメン表・選手名鑑・順位表・ベストナイン得票数を1つに統合したExcel対応（UTF-8 BOM付）CSVです</p>
      </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
      <!-- Card 1: Hitters -->
      <div class="p-3.5 rounded-xl border border-[var(--border)] bg-[var(--background)] flex flex-col justify-between space-y-2.5">
        <div class="space-y-1">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold px-2 py-0.5 rounded bg-[var(--primary)] text-[var(--primary-foreground)]">打者統合マスター</span>
            <span class="text-xs font-mono text-[var(--muted-foreground)]">4,140行 × 86列 (1.3MB)</span>
          </div>
          <h3 class="text-sm font-bold pt-1">master_hitters_2015_2026.csv</h3>
          <p class="text-xs text-[var(--muted-foreground)] leading-relaxed">
            打者4表（基本・RC・OPS・IsoP・BABIP等）＋11守備位置別スタメン数＋1〜9番打順別スタメン数＋年俸・年齢＋チーム順位＋ベストナイン得票・受賞フラグを完全統合。
          </p>
        </div>
        <button onclick="downloadUnifiedCSV('hitters', this)" class="w-full py-2 px-3 rounded-lg text-xs font-bold bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90 transition cursor-pointer">
          📥 打者統合CSVをダウンロード
        </button>
      </div>

      <!-- Card 2: Pitchers -->
      <div class="p-3.5 rounded-xl border border-[var(--border)] bg-[var(--background)] flex flex-col justify-between space-y-2.5">
        <div class="space-y-1">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold px-2 py-0.5 rounded bg-[var(--primary)] text-[var(--primary-foreground)]">投手統合マスター</span>
            <span class="text-xs font-mono text-[var(--muted-foreground)]">4,158行 × 54列 (871KB)</span>
          </div>
          <h3 class="text-sm font-bold pt-1">master_pitchers_2015_2026.csv</h3>
          <p class="text-xs text-[var(--muted-foreground)] leading-relaxed">
            投手3表（防御率・勝利・WHIP・DIPS・K/BB・完投・完封等）＋先発スタメン登板数＋投球回3進法換算値＋年俸・年齢＋チーム順位＋ベストナイン得票を完全統合。
          </p>
        </div>
        <button onclick="downloadUnifiedCSV('pitchers', this)" class="w-full py-2 px-3 rounded-lg text-xs font-bold bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90 transition cursor-pointer">
          📥 投手統合CSVをダウンロード
        </button>
      </div>

      <!-- Card 3: All Unified -->
      <div class="p-3.5 rounded-xl border border-[var(--border)] bg-[var(--background)] flex flex-col justify-between space-y-2.5">
        <div class="space-y-1">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold px-2 py-0.5 rounded bg-[var(--secondary)] text-[var(--secondary-foreground)]">打者＋投手 全1ファイル版</span>
            <span class="text-xs font-mono text-[var(--muted-foreground)]">8,285行 × 126列 (3.3MB)</span>
          </div>
          <h3 class="text-sm font-bold pt-1">master_all_players_unified_2015_2026.csv</h3>
          <p class="text-xs text-[var(--muted-foreground)] leading-relaxed">
            全12年分の打者成績・投手成績・守備スタメン・打順・プロフィール・順位・ベストナイン得票を1つのCSVファイルにすべてまとめた完全統合版（二刀流選手も1行に統合）。
          </p>
        </div>
        <button onclick="downloadUnifiedCSV('all_unified', this)" class="w-full py-2 px-3 rounded-lg text-xs font-bold bg-[var(--secondary)] text-[var(--secondary-foreground)] border border-[var(--border)] hover:opacity-90 transition cursor-pointer">
          📥 全選手1ファイル統合CSVをダウンロード
        </button>
      </div>
    </div>
  </div>

  <script>
{js_download_helper}
  </script>
</body>
</html>
"""

with open(os.path.join(ARTIFACT_DIR, "csv_download_widget.html"), "w", encoding="utf-8") as f:
    f.write(widget_html)

# 2. Inject download bar into bestnine_guide.html and web/index.html
download_banner_html = """
      <!-- Unified CSV Download Banner -->
      <div class="mt-4 p-4 rounded-xl border border-[var(--border)] bg-[var(--background)] flex flex-wrap items-center justify-between gap-3">
        <div class="space-y-0.5">
          <div class="text-xs font-bold text-[var(--primary)]">📥 統合済みマスターCSVダウンロード（2015〜2026年・Excel対応 UTF-8 BOM付）</div>
          <div class="text-xs text-[var(--muted-foreground)]">バラバラの成績表・日別スタメン・選手名鑑・順位表・ベストナイン得票を1つに結合したCSVをワンクリックで保存できます</div>
        </div>
        <div class="flex flex-wrap gap-2">
          <button onclick="downloadUnifiedCSV('hitters', this)" class="px-3 py-1.5 rounded-lg text-xs font-bold bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90 transition cursor-pointer">
            📥 打者統合CSV (4,140行×86列)
          </button>
          <button onclick="downloadUnifiedCSV('pitchers', this)" class="px-3 py-1.5 rounded-lg text-xs font-bold bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90 transition cursor-pointer">
            📥 投手統合CSV (4,158行×54列)
          </button>
          <button onclick="downloadUnifiedCSV('all_unified', this)" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-[var(--secondary)] text-[var(--secondary-foreground)] border border-[var(--border)] hover:opacity-90 transition cursor-pointer">
            📥 全選手1ファイル統合CSV (8,285行×126列)
          </button>
        </div>
      </div>
"""

for target_path in [
    os.path.join(ARTIFACT_DIR, "bestnine_guide.html"),
    os.path.join(WEB_DIR, "index.html"),
]:
    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "downloadUnifiedCSV" not in content:
        content = content.replace("      <!-- Navigation Tabs -->", download_banner_html + "\n      <!-- Navigation Tabs -->", 1)
        content = content.replace("    function switchTab(tabId, btnEl) {", js_download_helper + "\n    function switchTab(tabId, btnEl) {", 1)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

print("Successfully created csv_download_widget.html and updated bestnine_guide.html & web/index.html!")
