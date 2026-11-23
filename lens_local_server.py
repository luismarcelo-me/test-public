#!/usr/bin/env python3
"""
Servidor web local para a investigação de lifecycle com S3 Storage Lens.

Abre uma página em http://localhost:8000 onde você:
  1. informa o bucket, o prefixo da regra e (opcional) a data da regra;
  2. seleciona os CSVs do Storage Lens (.csv ou .csv.gz, de qualquer tamanho).
     O navegador lê cada arquivo e envia ao servidor SOMENTE as linhas do
     bucket informado;
  3. executa o s3_lens_lifecycle_investigation.py sobre os arquivos recebidos
     e vê o relatório na própria página, com links para os CSVs gerados.

Tudo roda na sua máquina: o servidor escuta apenas em 127.0.0.1 por padrão.

Uso:
    python lens_local_server.py                # http://localhost:8000
    python lens_local_server.py --port 9000 --open
    python lens_local_server.py --data-dir ./meus_dados

Requisitos: Python 3.8+. A análise (passo 3) também precisa de pandas.
O script s3_lens_lifecycle_investigation.py deve estar na mesma pasta.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
ANALYZER = HERE / "s3_lens_lifecycle_investigation.py"
BUCKET_RE = re.compile(r"^(?!.*\.\.)[A-Za-z0-9][A-Za-z0-9._\-]{1,254}$")
MAX_UPLOAD = 200 * 1024 * 1024  # por requisição (o navegador envia partes de até 15 MB)

DATA_DIR: Path = HERE / "lens_local_data"


def safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "arquivo"
    return name[:150]


def uploads_dir(bucket: str) -> Path:
    return DATA_DIR / "uploads" / bucket


def output_dir(bucket: str) -> Path:
    return DATA_DIR / "output" / bucket


class Handler(BaseHTTPRequestHandler):
    server_version = "LensLocal/1.0"

    # ------------------------------------------------------------ helpers
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, status: int, msg: str) -> None:
        self._json(status, {"ok": False, "error": msg})

    def log_message(self, fmt, *args):  # log mais enxuto
        sys.stderr.write("  %s %s\n" % (self.command, fmt % args))

    # ------------------------------------------------------------ GET
    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(HTTPStatus.OK, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if url.path == "/api/uploads":
            return self._json(HTTPStatus.OK, {"ok": True, "uploads": list_uploads()})
        if url.path.startswith("/output/"):
            return self._serve_output(unquote(url.path[len("/output/"):]))
        self._error(HTTPStatus.NOT_FOUND, "não encontrado")

    def _serve_output(self, rel: str):
        parts = rel.split("/")
        if len(parts) != 2 or not BUCKET_RE.match(parts[0]):
            return self._error(HTTPStatus.BAD_REQUEST, "caminho inválido")
        path = output_dir(parts[0]) / safe_name(parts[1])
        if not path.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "arquivo não encontrado")
        ctype = {".txt": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8",
                 ".gz": "application/gzip"}.get(path.suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, path.read_bytes(), ctype,
                   {"Content-Disposition": f'inline; filename="{path.name}"'})

    # ------------------------------------------------------------ POST
    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/upload":
            return self._upload(parse_qs(url.query))
        if url.path == "/api/analyze":
            return self._analyze()
        if url.path == "/api/clear":
            return self._clear()
        self._error(HTTPStatus.NOT_FOUND, "não encontrado")

    def _read_body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length else HTTPStatus.BAD_REQUEST,
                        "corpo vazio ou grande demais")
            return None
        return self.rfile.read(length)

    def _upload(self, q):
        bucket = (q.get("bucket") or [""])[0]
        name = (q.get("name") or [""])[0]
        part = (q.get("part") or ["1"])[0]
        if not BUCKET_RE.match(bucket):
            return self._error(HTTPStatus.BAD_REQUEST, "nome de bucket inválido")
        if not part.isdigit():
            return self._error(HTTPStatus.BAD_REQUEST, "parte inválida")
        body = self._read_body()
        if body is None:
            return
        base = re.sub(r"\.csv(\.gz)?$", "", safe_name(name), flags=re.I)
        dest = uploads_dir(bucket)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{base}.part{int(part):03d}.csv"
        path.write_bytes(body)
        self._json(HTTPStatus.OK, {"ok": True, "file": path.name, "bytes": len(body)})

    def _clear(self):
        body = self._read_body()
        if body is None:
            return
        try:
            bucket = json.loads(body).get("bucket", "")
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "JSON inválido")
        if not BUCKET_RE.match(bucket):
            return self._error(HTTPStatus.BAD_REQUEST, "nome de bucket inválido")
        removed = 0
        for f in uploads_dir(bucket).glob("*.csv"):
            f.unlink()
            removed += 1
        self._json(HTTPStatus.OK, {"ok": True, "removed": removed})

    def _analyze(self):
        body = self._read_body()
        if body is None:
            return
        try:
            req = json.loads(body)
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "JSON inválido")
        bucket = str(req.get("bucket", ""))
        prefix = str(req.get("prefix", "exp/")) or "exp/"
        rule_date = str(req.get("rule_date", ""))
        config_id = str(req.get("config_id", ""))
        if not BUCKET_RE.match(bucket):
            return self._error(HTTPStatus.BAD_REQUEST, "nome de bucket inválido")
        if rule_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", rule_date):
            return self._error(HTTPStatus.BAD_REQUEST, "data inválida (use AAAA-MM-DD)")
        src = uploads_dir(bucket)
        if not any(src.glob("*.csv")):
            return self._error(HTTPStatus.BAD_REQUEST, "nenhum arquivo enviado para este bucket")
        if not ANALYZER.is_file():
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR,
                               f"script de análise não encontrado em {ANALYZER}")
        out = output_dir(bucket)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ANALYZER), "--input", str(src), "--bucket", bucket,
               "--prefix", prefix, "--output", str(out)]
        if rule_date:
            cmd += ["--rule-date", rule_date]
        if config_id:
            cmd += ["--config-id", config_id]
        started = datetime.now()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            return self._error(HTTPStatus.GATEWAY_TIMEOUT, "a análise passou de 1 hora")
        report_path = out / "report.txt"
        report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
        if proc.returncode != 0 and not report:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
            hint = ""
            if "No module named 'pandas'" in (proc.stderr or ""):
                hint = " Instale com: pip install pandas"
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR,
                               "a análise falhou." + hint + "\n" + "\n".join(err))
        files = sorted(p.name for p in out.iterdir() if p.is_file())
        self._json(HTTPStatus.OK, {
            "ok": True, "returncode": proc.returncode, "report": report, "files": files,
            "bucket": bucket, "seconds": round((datetime.now() - started).total_seconds(), 1),
            "output_dir": str(out),
        })


def list_uploads():
    base = DATA_DIR / "uploads"
    rows = []
    if base.is_dir():
        for bdir in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(bdir.glob("*.csv")):
                st = f.stat()
                rows.append({"bucket": bdir.name, "file": f.name, "bytes": st.st_size,
                             "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
    return rows


def main() -> int:
    global DATA_DIR
    ap = argparse.ArgumentParser(description="Servidor web local para análise do S3 Storage Lens.")
    ap.add_argument("--host", default="127.0.0.1", help="Endereço (padrão: 127.0.0.1, só sua máquina)")
    ap.add_argument("--port", type=int, default=8000, help="Porta (padrão: 8000)")
    ap.add_argument("--data-dir", default=str(HERE / "lens_local_data"),
                    help="Onde guardar os arquivos recebidos e os resultados")
    ap.add_argument("--open", action="store_true", help="Abrir o navegador automaticamente")
    args = ap.parse_args()

    DATA_DIR = Path(args.data_dir).expanduser().resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"Servidor no ar: {url}")
    print(f"Dados em: {DATA_DIR}")
    print("Ctrl+C para encerrar.")
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        httpd.server_close()
    return 0


PAGE = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Investigação Storage Lens</title>
<style>
:root {
  --bg: #f5f7fa; --surface: #ffffff; --fg: #17202e; --muted: #5a6577; --line: #d9dfe8;
  --accent: #2552c4; --accent-fg: #ffffff; --accent-soft: #e6edfb;
  --ok: #1f7a4d; --warn: #a15c00; --bad: #b3261e;
  --font-body: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #10141b; --surface: #171c25; --fg: #e4e9f1; --muted: #98a3b5; --line: #2a3240;
    --accent: #7ea0ff; --accent-fg: #0d1320; --accent-soft: #1c2740;
    --ok: #5cc48e; --warn: #e2a64a; --bad: #f08a83; color-scheme: dark; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font-body);
  font-size: 15px; line-height: 1.55; padding: 32px max(16px, 4vw) 56px; }
main { max-width: 900px; margin: 0 auto; display: grid; gap: 28px; }
h1, h2 { font-weight: 600; margin: 0; text-wrap: balance; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.05rem; }
p { margin: 0; }
.lead { color: var(--muted); max-width: 64ch; margin-top: 6px; }
.eyebrow { font-family: var(--font-mono); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
section { display: grid; gap: 14px; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 20px; display: grid; gap: 16px; min-width: 0; }
.fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
label { display: grid; gap: 4px; font-size: .85rem; font-weight: 500; align-content: start; }
label small { font-weight: 400; color: var(--muted); }
input[type=text], input[type=date] { font: inherit; font-family: var(--font-mono); font-size: .9rem; color: var(--fg);
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; width: 100%; min-width: 0; }
input:focus-visible, button:focus-visible, .drop:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.check { display: flex; gap: 8px; align-items: flex-start; font-weight: 400; font-size: .88rem; }
.check input { margin-top: 4px; }
.drop { border: 1.5px dashed var(--line); border-radius: 10px; padding: 28px 16px; text-align: center;
  display: grid; gap: 6px; justify-items: center; cursor: pointer; background: var(--bg); }
.drop.over { border-color: var(--accent); background: var(--accent-soft); }
.drop span { color: var(--muted); font-size: .85rem; }
button { font: inherit; font-weight: 500; border-radius: 6px; padding: 9px 16px; cursor: pointer;
  border: 1px solid var(--line); background: var(--surface); color: var(--fg); }
button.primary { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
button:disabled { opacity: .5; cursor: not-allowed; }
.actions { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.status { font-size: .88rem; color: var(--muted); white-space: pre-wrap; }
.status.ok { color: var(--ok); } .status.bad { color: var(--bad); } .status.warn { color: var(--warn); }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { font-weight: 500; color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; background: var(--bg); }
tr:last-child td { border-bottom: 0; }
td.num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; text-align: right; }
td.mono { font-family: var(--font-mono); }
.pill { display: inline-block; font-size: .75rem; padding: 1px 8px; border-radius: 999px; border: 1px solid currentColor; }
.pill.ok { color: var(--ok); } .pill.bad { color: var(--bad); } .pill.warn { color: var(--warn); } .pill.wait { color: var(--muted); }
.empty { color: var(--muted); font-size: .9rem; padding: 18px; text-align: center; border: 1px dashed var(--line); border-radius: 8px; }
.note { font-size: .85rem; color: var(--muted); }
pre.report { margin: 0; font-family: var(--font-mono); font-size: .78rem; line-height: 1.5; background: var(--bg);
  border: 1px solid var(--line); border-radius: 8px; padding: 14px; overflow: auto; max-height: 75vh; }
.files { display: flex; flex-wrap: wrap; gap: 8px; }
.files a { font-family: var(--font-mono); font-size: .8rem; color: var(--accent); border: 1px solid var(--line);
  border-radius: 6px; padding: 3px 8px; text-decoration: none; background: var(--bg); }
.files a:hover { border-color: var(--accent); }
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">S3 Storage Lens · investigação de lifecycle · local</div>
    <h1>Investigação Storage Lens</h1>
    <p class="lead">Selecione os CSVs do Lens. O navegador lê cada arquivo e envia ao servidor local só as linhas do bucket informado. Depois, rode a análise e veja o relatório aqui mesmo. Nada sai da sua máquina.</p>
  </header>

  <section>
    <h2>1. Bucket e regra</h2>
    <div class="panel">
      <div class="fields">
        <label for="bucket">Nome do bucket
          <input type="text" id="bucket" placeholder="meu-bucket" autocomplete="off" spellcheck="false"></label>
        <label for="prefix">Prefixo da regra
          <input type="text" id="prefix" value="exp/" autocomplete="off" spellcheck="false"></label>
        <label for="ruledate"><span>Data da regra <small>(opcional)</small></span>
          <input type="date" id="ruledate"></label>
        <label for="configid"><span>Configuração do Lens <small>(opcional)</small></span>
          <input type="text" id="configid" placeholder="configuration_id" autocomplete="off" spellcheck="false"></label>
      </div>
      <label class="check" for="mask"><input type="checkbox" id="mask" checked>
        <span>Mascarar o número da conta AWS nos arquivos salvos (o script não precisa dele)</span></label>
    </div>
  </section>

  <section>
    <h2>2. Arquivos do Storage Lens</h2>
    <div class="panel">
      <div class="drop" id="drop" tabindex="0" role="button" aria-label="Selecionar arquivos CSV do Storage Lens">
        <strong>Arraste os arquivos aqui ou clique para escolher</strong>
        <span>Aceita .csv e .csv.gz, vários de uma vez, de qualquer tamanho</span>
      </div>
      <input type="file" id="file" multiple accept=".csv,.gz,text/csv" hidden>
      <div id="files-empty" class="empty">Nenhum arquivo processado nesta sessão. Informe o bucket e selecione os exports.</div>
      <div class="table-wrap" id="files-wrap" hidden>
        <table><thead><tr><th>Arquivo</th><th>Situação</th><th>Linhas lidas</th><th>Do bucket</th><th>Período</th></tr></thead>
        <tbody id="files-body"></tbody></table>
      </div>
      <span class="status" id="status" role="status"></span>
    </div>
  </section>

  <section>
    <h2>3. Arquivos recebidos pelo servidor</h2>
    <div id="up-empty" class="empty">Nenhum arquivo recebido ainda.</div>
    <div class="table-wrap" id="up-wrap" hidden>
      <table><thead><tr><th>Bucket</th><th>Arquivo</th><th>Tamanho</th><th>Recebido em</th></tr></thead>
      <tbody id="up-body"></tbody></table>
    </div>
    <div class="actions">
      <button class="primary" id="analyze" disabled>Executar análise</button>
      <button id="clear" disabled>Apagar arquivos deste bucket</button>
      <span class="status" id="an-status" role="status"></span>
    </div>
  </section>

  <section id="result" hidden>
    <h2>4. Relatório</h2>
    <div class="panel">
      <div class="files" id="out-files"></div>
      <pre class="report" id="report"></pre>
    </div>
  </section>
</main>

<script>
(() => {
  const EXPECTED = ["version_number","configuration_id","report_date","aws_account_number","aws_region","storage_class","record_type","record_value","bucket_name","metric_name","metric_value"];
  const PART_LIMIT = 15 * 1024 * 1024;
  const $ = (id) => document.getElementById(id);
  const fmtN = (n) => n.toLocaleString("pt-BR");
  const fmtB = (b) => b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : (b / 1024).toFixed(0) + " KB";
  const processed = [];
  let busy = false, uploads = [];

  function splitCsv(line) {
    if (line.indexOf('"') === -1) return line.split(",");
    const out = []; let cur = "", q = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (q) { if (c === '"') { if (line[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += c; }
      else if (c === '"') q = true;
      else if (c === ",") { out.push(cur); cur = ""; }
      else cur += c;
    }
    out.push(cur); return out;
  }
  const joinCsv = (cells) => cells.map((v) => /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v).join(",");

  function setStatus(el, text, kind) { const s = $(el); s.textContent = text; s.className = "status" + (kind ? " " + kind : ""); }

  async function sendPart(entry, text) {
    entry.partNo++;
    const q = new URLSearchParams({ bucket: entry.bucket, name: entry.name, part: String(entry.partNo) });
    const r = await fetch("/api/upload?" + q, { method: "POST", headers: { "Content-Type": "text/csv" }, body: text });
    const j = await r.json().catch(() => ({ ok: false, error: "resposta inválida do servidor" }));
    if (!j.ok) throw new Error(j.error || "falha no envio");
    entry.sentBytes += j.bytes;
  }

  async function processFile(file, entry) {
    let stream = file.stream();
    if (/\.gz$/i.test(file.name)) {
      if (typeof DecompressionStream === "undefined") throw new Error("este navegador não descompacta .gz; descompacte antes");
      stream = stream.pipeThrough(new DecompressionStream("gzip"));
    }
    const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
    let rest = "", header = null, idx = null, headerLine = EXPECTED.join(",");
    let part = [], partSize = headerLine.length + 1;
    const pending = [];
    const flush = () => {
      if (!part.length) return;
      pending.push(headerLine + "\n" + part.join("\n") + "\n");
      part = []; partSize = headerLine.length + 1;
    };
    const handle = (raw) => {
      const line = raw.replace(/\r$/, "");
      if (!line.trim()) return;
      if (header === null) {
        const cells = splitCsv(line).map((c) => c.trim().toLowerCase());
        if (cells.includes("bucket_name")) {
          header = cells; idx = Object.fromEntries(cells.map((c, i) => [c, i]));
          headerLine = cells.join(","); partSize = headerLine.length + 1; return;
        }
        header = EXPECTED; idx = Object.fromEntries(EXPECTED.map((c, i) => [c, i]));
      }
      entry.read++;
      if (line.indexOf(entry.bucket) === -1) return;
      const cells = splitCsv(line).map((c) => c.trim());
      if (cells[idx.bucket_name] !== entry.bucket) return;
      if (entry.mask && idx.aws_account_number !== undefined) cells[idx.aws_account_number] = "000000000000";
      const d = cells[idx.report_date] || "";
      if (d) { if (!entry.minDate || d < entry.minDate) entry.minDate = d; if (!entry.maxDate || d > entry.maxDate) entry.maxDate = d; }
      const out = joinCsv(cells);
      if (partSize + out.length + 1 > PART_LIMIT) flush();
      part.push(out); partSize += out.length + 1; entry.kept++;
    };
    let ticks = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const lines = (rest + value).split("\n");
      rest = lines.pop();
      for (const l of lines) handle(l);
      while (pending.length) await sendPart(entry, pending.shift());
      if (++ticks % 50 === 0) renderFiles();
    }
    if (rest) handle(rest);
    flush();
    while (pending.length) await sendPart(entry, pending.shift());
  }

  function renderFiles() {
    const body = $("files-body"); body.textContent = "";
    const labels = { lendo: ["wait", "lendo e enviando…"], enviado: ["ok", "enviado"], vazio: ["warn", "sem linhas do bucket"], erro: ["bad", "erro"] };
    for (const e of processed) {
      const tr = document.createElement("tr");
      const p = labels[e.status];
      const vals = [[e.name, "mono"], [null, ""], [fmtN(e.read), "num"],
        [fmtN(e.kept) + (e.partNo > 1 ? ` (${e.partNo} partes)` : ""), "num"],
        [e.minDate ? `${e.minDate} → ${e.maxDate}` : (e.error || "—"), "mono"]];
      vals.forEach(([t, c], i) => {
        const td = document.createElement("td"); if (c) td.className = c;
        if (i === 1) { const s = document.createElement("span"); s.className = "pill " + p[0]; s.textContent = p[1]; td.appendChild(s); }
        else td.textContent = t;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    }
    $("files-wrap").hidden = !processed.length; $("files-empty").hidden = !!processed.length;
    renderButtons();
  }

  function renderButtons() {
    const b = $("bucket").value.trim();
    const has = uploads.some((u) => u.bucket === b);
    $("analyze").disabled = busy || !has;
    $("clear").disabled = busy || !has;
  }

  async function loadUploads() {
    try {
      const j = await (await fetch("/api/uploads")).json();
      uploads = j.uploads || [];
    } catch { uploads = []; }
    const body = $("up-body"); body.textContent = "";
    for (const u of uploads) {
      const tr = document.createElement("tr");
      for (const [t, c] of [[u.bucket, "mono"], [u.file, "mono"], [fmtB(u.bytes), "num"], [new Date(u.modified).toLocaleString("pt-BR"), ""]]) {
        const td = document.createElement("td"); if (c) td.className = c; td.textContent = t; tr.appendChild(td);
      }
      body.appendChild(tr);
    }
    $("up-wrap").hidden = !uploads.length; $("up-empty").hidden = !!uploads.length;
    renderButtons();
  }

  async function addFiles(list) {
    const bucket = $("bucket").value.trim();
    if (!/^(?!.*\.\.)[A-Za-z0-9][A-Za-z0-9._\-]{1,254}$/.test(bucket)) { setStatus("status", "Informe um nome de bucket válido antes de escolher os arquivos.", "bad"); $("bucket").focus(); return; }
    const files = Array.from(list).filter((f) => /\.csv(\.gz)?$/i.test(f.name));
    if (!files.length) { setStatus("status", "Nenhum .csv ou .csv.gz entre os arquivos escolhidos.", "bad"); return; }
    busy = true; setStatus("status", `Lendo ${files.length} arquivo(s)…`);
    for (const f of files) {
      const e = { name: f.name, bucket, mask: $("mask").checked, status: "lendo", read: 0, kept: 0, minDate: "", maxDate: "", partNo: 0, sentBytes: 0, error: "" };
      processed.push(e); renderFiles();
      try { await processFile(f, e); e.status = e.kept ? "enviado" : "vazio"; }
      catch (err) { e.status = "erro"; e.error = String((err && err.message) || err); }
      renderFiles();
    }
    busy = false;
    const kept = processed.filter((e) => e.bucket === bucket).reduce((a, e) => a + e.kept, 0);
    setStatus("status", kept ? `${fmtN(kept)} linhas do bucket enviadas ao servidor.` : `Nenhuma linha com bucket_name = "${bucket}". Confira o nome.`, kept ? "ok" : "warn");
    await loadUploads();
  }

  async function analyze() {
    busy = true; renderButtons();
    setStatus("an-status", "Executando a análise… (pode levar alguns minutos)");
    try {
      const r = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bucket: $("bucket").value.trim(), prefix: $("prefix").value.trim(),
          rule_date: $("ruledate").value || "", config_id: $("configid").value.trim() }) });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error);
      $("report").textContent = j.report;
      const files = $("out-files"); files.textContent = "";
      for (const f of j.files) {
        const a = document.createElement("a");
        a.href = "/output/" + encodeURIComponent(j.bucket) + "/" + encodeURIComponent(f);
        a.target = "_blank"; a.rel = "noopener"; a.textContent = f; files.appendChild(a);
      }
      $("result").hidden = false;
      setStatus("an-status", `Concluída em ${j.seconds}s. Resultados em ${j.output_dir}`, "ok");
      $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setStatus("an-status", "Falha: " + ((err && err.message) || err), "bad");
    } finally { busy = false; renderButtons(); }
  }

  let confirmClear = false;
  async function clearUploads() {
    if (!confirmClear) { confirmClear = true; $("clear").textContent = "Clique de novo para confirmar"; setTimeout(() => { confirmClear = false; $("clear").textContent = "Apagar arquivos deste bucket"; }, 4000); return; }
    confirmClear = false; $("clear").textContent = "Apagar arquivos deste bucket";
    const r = await fetch("/api/clear", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ bucket: $("bucket").value.trim() }) });
    const j = await r.json().catch(() => ({ ok: false }));
    setStatus("an-status", j.ok ? `${j.removed} arquivo(s) apagado(s).` : "Não foi possível apagar.", j.ok ? "ok" : "bad");
    await loadUploads();
  }

  const drop = $("drop"), input = $("file");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (!busy) addFiles(e.dataTransfer.files); });
  input.addEventListener("change", () => { if (!busy) addFiles(input.files); input.value = ""; });
  $("bucket").addEventListener("input", renderButtons);
  $("analyze").addEventListener("click", analyze);
  $("clear").addEventListener("click", clearUploads);
  renderFiles(); loadUploads();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
