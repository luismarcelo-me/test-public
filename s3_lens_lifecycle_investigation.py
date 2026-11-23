#!/usr/bin/env python3
"""
Investigação de lifecycle S3 a partir de exports CSV do S3 Storage Lens.

Contexto: uma lifecycle rule de expiração (1 dia) foi aplicada no prefixo
``exp/`` de um bucket, mas parte dos dados (ex.: 31 TB de 581 TB) não foi
removida. Este script lê os CSVs do Storage Lens (que contêm TODOS os buckets
da conta/organização), filtra apenas o bucket alvo e procura evidências das
causas mais comuns:

  1. Versionamento habilitado no passado (e depois suspenso)
     -> A ação ``Expiration`` só atua sobre a versão *corrente*. Em bucket
        versionado/suspenso ela cria um delete marker e a versão antiga vira
        *noncurrent*. Versões criadas enquanto o versionamento estava
        "Enabled" continuam ocupando espaço até existir uma regra
        ``NoncurrentVersionExpiration``.
  2. Uploads multipart incompletos
     -> Não são removidos por ``Expiration``; precisam de
        ``AbortIncompleteMultipartUpload``.
  3. Delete markers órfãos (expired object delete markers).
  4. Object Lock (retention / legal hold) impedindo a deleção.
  5. Versões correntes ainda presentes -> filtro da regra (tag, tamanho,
     prefixo digitado diferente), regra desabilitada ou processamento
     assíncrono ainda em andamento.

A seção 3 do relatório faz uma análise aprofundada do prefixo: cobertura dos
dados, participação no bucket, balanço antes/depois, progresso diário,
inferência do estado de versionamento (Enabled/Suspended/não versionado),
decomposição do residual, residual por storage class e por subprefixo.
Sem --rule-date, a data da regra é inferida pela primeira queda > 1%.

Uso:
    python s3_lens_lifecycle_investigation.py \\
        --input ./lens_csv \\
        --bucket meu-bucket \\
        --prefix exp/ \\
        --rule-date 2026-10-01 \\
        --output ./resultado

Requisitos: Python 3.8+ e pandas (``pip install pandas``).
Aceita arquivos ``.csv`` e ``.csv.gz`` (busca recursiva no diretório).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import unquote

import pandas as pd

EXPECTED_COLUMNS = [
    "version_number",
    "configuration_id",
    "report_date",
    "aws_account_number",
    "aws_region",
    "storage_class",
    "record_type",
    "record_value",
    "bucket_name",
    "metric_name",
    "metric_value",
]

# Métricas relevantes para a investigação. A chave é um nome lógico e o valor
# é a lista de nomes possíveis no export (comparação case-insensitive).
METRICS: Dict[str, List[str]] = {
    "total_bytes": ["StorageBytes"],
    "total_objects": ["ObjectCount"],
    "current_bytes": ["CurrentVersionStorageBytes"],
    "current_objects": ["CurrentVersionObjectCount"],
    "noncurrent_bytes": ["NonCurrentVersionStorageBytes", "NoncurrentVersionStorageBytes"],
    "noncurrent_objects": ["NonCurrentVersionObjectCount", "NoncurrentVersionObjectCount"],
    "delete_marker_objects": ["DeleteMarkerObjectCount"],
    "delete_marker_bytes": ["DeleteMarkerStorageBytes"],
    "mpu_bytes": ["IncompleteMultipartUploadStorageBytes", "IncompleteMPUStorageBytes"],
    "mpu_objects": ["IncompleteMultipartUploadObjectCount", "IncompleteMPUObjectCount"],
    "mpu_bytes_gt7d": [
        "IncompleteMPUStorageBytesOlderThan7Days",
        "IncompleteMultipartUploadStorageBytesOlderThan7Days",
    ],
    "object_lock_bytes": ["ObjectLockEnabledStorageBytes"],
    "object_lock_objects": ["ObjectLockEnabledObjectCount", "ObjectLockEnabledStorageCount"],
    "replicated_bytes": ["ReplicatedStorageBytes"],
}

BYTE_METRICS = [k for k in METRICS if k.endswith("_bytes") or k.endswith("_gt7d")]


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def fmt_bytes(value: float) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    tib = value / 1024**4
    tb = value / 1000**4
    if tib >= 0.01:
        return f"{tib:,.2f} TiB ({tb:,.2f} TB)"
    gib = value / 1024**3
    return f"{gib:,.2f} GiB"


def fmt_count(value: float) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{int(value):,}"


def normalize_prefix(p: str) -> str:
    # O Lens grava prefixos aninhados com "/" codificado: "exp%2Fcanais/".
    p = unquote((p or "").strip()).lstrip("/")
    return p if p.endswith("/") or p == "" else p + "/"


def find_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    files = sorted(
        f for f in root.rglob("*")
        if f.is_file() and (f.name.endswith(".csv") or f.name.endswith(".csv.gz"))
    )
    return files


def build_metric_lookup(available: Iterable[str]) -> Dict[str, str]:
    """Mapeia nome lógico -> nome real presente nos dados."""
    lower = {m.lower(): m for m in available}
    lookup = {}
    for key, candidates in METRICS.items():
        for c in candidates:
            if c.lower() in lower:
                lookup[key] = lower[c.lower()]
                break
    return lookup


# --------------------------------------------------------------------------- #
# Leitura
# --------------------------------------------------------------------------- #
def read_bucket_rows(files: List[Path], bucket: str, chunksize: int) -> pd.DataFrame:
    frames = []
    total_rows = 0
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] lendo {f}", file=sys.stderr)
        try:
            # Detecta se o arquivo tem header.
            head = pd.read_csv(f, nrows=0, skipinitialspace=True)
            cols = [c.strip().lower() for c in head.columns]
            has_header = "bucket_name" in cols
            reader = pd.read_csv(
                f,
                dtype=str,
                skipinitialspace=True,
                chunksize=chunksize,
                header=0 if has_header else None,
                names=None if has_header else EXPECTED_COLUMNS,
                keep_default_na=False,
            )
            for chunk in reader:
                chunk.columns = [c.strip().lower() for c in chunk.columns]
                total_rows += len(chunk)
                sel = chunk[chunk["bucket_name"].str.strip() == bucket]
                if not sel.empty:
                    frames.append(sel)
        except Exception as exc:  # arquivo corrompido não deve parar tudo
            print(f"    ! erro ao ler {f}: {exc}", file=sys.stderr)

    print(f"  linhas lidas no total: {total_rows:,}", file=sys.stderr)
    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce").fillna(0)
    df["record_type"] = df["record_type"].str.upper()
    df["storage_class"] = df["storage_class"].replace("", "N/A")

    # Exports podem se sobrepor (mesmo dia exportado mais de uma vez).
    key = [
        "report_date", "aws_account_number", "aws_region", "storage_class",
        "record_type", "record_value", "bucket_name", "metric_name",
    ]
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="last")
    if len(df) != before:
        print(f"  {before - len(df):,} linhas duplicadas removidas", file=sys.stderr)
    return df


# --------------------------------------------------------------------------- #
# Agregações
# --------------------------------------------------------------------------- #
def timeseries(df: pd.DataFrame, lookup: Dict[str, str]) -> pd.DataFrame:
    """Série diária (somando storage classes) das métricas de interesse."""
    inv = {v: k for k, v in lookup.items()}
    sub = df[df["metric_name"].isin(inv)]
    if sub.empty:
        return pd.DataFrame()
    ts = (
        sub.groupby(["report_date", "metric_name"])["metric_value"].sum()
        .unstack("metric_name")
        .rename(columns=inv)
        .sort_index()
    )
    for k in METRICS:
        if k not in ts.columns:
            ts[k] = float("nan")
    return ts[list(METRICS)].astype(float)


def by_storage_class(df: pd.DataFrame, lookup: Dict[str, str], day) -> pd.DataFrame:
    inv = {v: k for k, v in lookup.items()}
    sub = df[(df["report_date"] == day) & (df["metric_name"].isin(inv))]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["storage_class", "metric_name"])["metric_value"].sum()
        .unstack("metric_name")
        .rename(columns=inv)
        .fillna(0)
    )


def first_date_where(ts: pd.DataFrame, col: str, cond) -> Optional[object]:
    if col not in ts or ts[col].isna().all():
        return None
    s = ts[col].dropna()
    hit = s[cond(s)]
    return hit.index[0] if not hit.empty else None


def value_at(ts: pd.DataFrame, day, col: str) -> float:
    if ts.empty or col not in ts or day not in ts.index:
        return float("nan")
    v = ts.at[day, col]
    return float("nan") if pd.isna(v) else float(v)


def nearest_on_or_before(index, day):
    cands = [d for d in index if d <= day]
    return cands[-1] if cands else None


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def h(self, title: str) -> None:
        self.add("")
        self.add("=" * 78)
        self.add(title)
        self.add("=" * 78)

    def sub(self, title: str) -> None:
        self.add("")
        self.add(f"--- {title} " + "-" * max(0, 74 - len(title)))

    def add(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def describe_snapshot(r: Report, label: str, ts: pd.DataFrame, day) -> None:
    r.add(f"\n{label} ({day}):")
    rows = [
        ("Total armazenado", "total_bytes", "total_objects"),
        ("Versões correntes", "current_bytes", "current_objects"),
        ("Versões NÃO correntes", "noncurrent_bytes", "noncurrent_objects"),
        ("Delete markers", "delete_marker_bytes", "delete_marker_objects"),
        ("Multipart incompleto", "mpu_bytes", "mpu_objects"),
        ("Object Lock habilitado", "object_lock_bytes", "object_lock_objects"),
    ]
    for name, b, c in rows:
        r.add(f"  {name:<26} {fmt_bytes(value_at(ts, day, b)):>32}   "
              f"objetos: {fmt_count(value_at(ts, day, c))}")


def diagnose(r: Report, ts: pd.DataFrame, scope: str, rule_date) -> None:
    if ts.empty:
        r.add(f"Sem dados de métricas para o escopo {scope}.")
        return

    last = ts.index[-1]
    total = value_at(ts, last, "total_bytes")
    cur = value_at(ts, last, "current_bytes")
    nonc = value_at(ts, last, "noncurrent_bytes")
    mpu = value_at(ts, last, "mpu_bytes")
    lock = value_at(ts, last, "object_lock_bytes")
    dm = value_at(ts, last, "delete_marker_objects")

    def pct(v):
        return f"{(v / total * 100):.1f}%" if total and not pd.isna(v) and total > 0 else "n/d"

    r.add(f"\nComposição do que restou em {last} (escopo: {scope}):")
    r.add(f"  Total ......................... {fmt_bytes(total)}")
    r.add(f"  Versões correntes ............. {fmt_bytes(cur)}  [{pct(cur)}]")
    r.add(f"  Versões não correntes ......... {fmt_bytes(nonc)}  [{pct(nonc)}]")
    r.add(f"  Multipart incompleto .......... {fmt_bytes(mpu)}  [{pct(mpu)}]")
    r.add(f"  Object Lock habilitado ........ {fmt_bytes(lock)}  [{pct(lock)}]")
    r.add(f"  Delete markers (objetos) ...... {fmt_count(dm)}")

    findings: List[str] = []

    # --- Versionamento -------------------------------------------------------
    first_nc = first_date_where(ts, "noncurrent_objects", lambda s: s > 0)
    first_dm = first_date_where(ts, "delete_marker_objects", lambda s: s > 0)
    if first_nc is not None:
        r.add(f"\n  Primeira data com versões não correntes nos dados: {first_nc}")
    if first_dm is not None:
        r.add(f"  Primeira data com delete markers nos dados:        {first_dm}")

    if rule_date is not None:
        before = nearest_on_or_before(ts.index, rule_date)
        if before is not None:
            describe_snapshot(r, "Snapshot antes/no dia da regra", ts, before)
            dm_before = value_at(ts, before, "delete_marker_objects")
            nc_before = value_at(ts, before, "noncurrent_bytes")
            if not pd.isna(dm) and not pd.isna(dm_before) and dm > dm_before * 1.1 + 1000:
                findings.append(
                    f"Delete markers cresceram de {fmt_count(dm_before)} para "
                    f"{fmt_count(dm)} após a regra. Em bucket NUNCA versionado a "
                    "expiração apaga o objeto definitivamente e não gera delete "
                    "marker. Isso prova que o bucket está com versionamento "
                    "Enabled ou Suspended no momento da execução da regra."
                )
            if not pd.isna(nc_before) and nc_before > 0:
                findings.append(
                    f"Já existiam {fmt_bytes(nc_before)} em versões não correntes "
                    "ANTES da regra: são versões geradas quando o versionamento "
                    "esteve habilitado. A ação Expiration não as remove."
                )
        else:
            r.add(f"\n  (Não há dados anteriores a {rule_date} para comparar.)")

    if not pd.isna(nonc) and total and nonc / total >= 0.5:
        findings.append(
            f"CAUSA PRINCIPAL PROVÁVEL: {pct(nonc)} do que restou ({fmt_bytes(nonc)}) "
            "são versões NÃO correntes. Isso confirma a hipótese de versionamento "
            "habilitado no passado. A regra de Expiration atua apenas na versão "
            "corrente; é necessária uma regra NoncurrentVersionExpiration "
            "(NoncurrentDays=1) no mesmo filtro de prefixo."
        )
    elif not pd.isna(nonc) and nonc > 0:
        findings.append(
            f"Há {fmt_bytes(nonc)} em versões não correntes ({pct(nonc)}). "
            "Adicione NoncurrentVersionExpiration à regra."
        )

    # --- Multipart -----------------------------------------------------------
    if not pd.isna(mpu) and mpu > 0:
        findings.append(
            f"{fmt_bytes(mpu)} em uploads multipart incompletos ({pct(mpu)}). "
            "Expiration não remove partes de MPU; adicione "
            "AbortIncompleteMultipartUpload (DaysAfterInitiation=1)."
        )

    # --- Object Lock ---------------------------------------------------------
    if not pd.isna(lock) and lock > 0:
        findings.append(
            f"{fmt_bytes(lock)} com Object Lock habilitado ({pct(lock)}). "
            "Versões com retention (Governance/Compliance) ou legal hold ativos "
            "não são removidas pelo lifecycle até o fim da retenção."
        )

    # --- Delete markers ------------------------------------------------------
    if not pd.isna(dm) and dm > 0 and not pd.isna(cur) and cur == 0:
        findings.append(
            f"{fmt_count(dm)} delete markers sem versão corrente. Use "
            "ExpiredObjectDeleteMarker=true (após remover as versões não "
            "correntes) para limpar a listagem."
        )

    # --- Versões correntes remanescentes -------------------------------------
    if not pd.isna(cur) and total and cur / total >= 0.05:
        findings.append(
            f"Ainda há {fmt_bytes(cur)} em versões CORRENTES ({pct(cur)}). Se "
            "estão dentro do prefixo da regra, verifique: filtro com tags ou "
            "ObjectSizeGreaterThan/LessThan, prefixo da regra (case-sensitive, "
            "'exp/' vs 'exp'), status Enabled da regra, Object Lock, ou se o "
            "lifecycle ainda está processando (execução assíncrona; volumes "
            "grandes podem levar dias). Lembre também que o Storage Lens tem "
            "atraso de ~24-48h."
        )

    # --- Tendência -----------------------------------------------------------
    if len(ts) >= 3 and not ts["total_bytes"].isna().all():
        tail = ts["total_bytes"].dropna().tail(4)
        if len(tail) >= 2 and tail.iloc[-1] < tail.iloc[0] * 0.98:
            findings.append(
                "O volume total ainda está caindo nos últimos dias: o lifecycle "
                "pode ainda estar em execução. Aguarde antes de concluir."
            )
        elif len(tail) >= 2:
            findings.append(
                "O volume total estabilizou nos últimos dias: o que restou "
                "não será removido pela regra atual."
            )

    r.add("\nAchados:")
    if findings:
        for i, f in enumerate(findings, 1):
            r.add(f"  {i}. {f}")
    else:
        r.add("  Nenhum achado automático. Revise as séries temporais exportadas.")


def versioning_timeline(r: Report, ts: pd.DataFrame) -> None:
    """Mostra variações relevantes que sugerem mudanças de estado de versionamento."""
    if ts.empty or ts["noncurrent_objects"].isna().all():
        r.add("Métricas de versão não corrente não encontradas no export.")
        return
    nc = ts["noncurrent_objects"].fillna(0)
    dm = ts["delete_marker_objects"].fillna(0)
    d_nc = nc.diff()
    d_dm = dm.diff()

    r.add("Datas com variação relevante (>1% ou >10k objetos) em versões não "
          "correntes / delete markers:")
    shown = 0
    for day in ts.index[1:]:
        notes = []
        for label, series, delta in (("noncurrent", nc, d_nc), ("delete markers", dm, d_dm)):
            dv = delta.get(day, 0)
            prev = series.shift(1).get(day, 0)
            if pd.isna(dv) or dv == 0:
                continue
            if abs(dv) > 10_000 or (prev and abs(dv) / prev > 0.01):
                notes.append(f"{label} {dv:+,.0f}")
        if notes:
            r.add(f"  {day}: " + "; ".join(notes))
            shown += 1
    if not shown:
        r.add("  (nenhuma variação relevante)")

    # Período em que as não correntes pararam de crescer.
    growing = d_nc[d_nc > 0]
    if not growing.empty:
        r.add(f"\n  Último dia em que versões não correntes CRESCERAM: {growing.index[-1]}")
        r.add("  Se depois dessa data não há mais crescimento mesmo com "
              "sobrescritas/deleções, é um indício de que o versionamento foi "
              "suspenso por volta dessa época.")
    r.add("\n  Obs.: o histórico do Storage Lens é limitado (14 dias no free tier, "
          "15 meses no advanced / ou desde quando o export foi ativado). Para "
          "confirmar quando o versionamento mudou de estado, consulte o "
          "CloudTrail (evento PutBucketVersioning) e "
          "'aws s3api get-bucket-versioning'.")


# --------------------------------------------------------------------------- #
# Análise aprofundada do prefixo
# --------------------------------------------------------------------------- #
COLD_CLASSES = {
    "STANDARD_IA": 30, "ONEZONE_IA": 30, "GLACIER_IR": 90,
    "GLACIER": 90, "DEEP_ARCHIVE": 180, "INTELLIGENT_TIERING": 0,
}


def has(ts: pd.DataFrame, key: str) -> bool:
    return not ts.empty and key in ts and not ts[key].isna().all()


def tib(v) -> str:
    return "n/d" if v is None or pd.isna(v) else f"{v / 1024**4:,.2f}"


def pct_of(v, base) -> str:
    if v is None or base is None or pd.isna(v) or pd.isna(base) or not base:
        return "n/d"
    return f"{v / base * 100:.1f}%"


def nz(v) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


def dominant_cause(cur, nonc, mpu) -> str:
    parts = {"correntes": nz(cur), "não correntes": nz(nonc), "multipart": nz(mpu)}
    tot = sum(parts.values())
    if tot <= 0:
        return "-"
    k = max(parts, key=parts.get)
    return f"{k} ({parts[k] / tot * 100:.0f}%)"


def infer_rule_date(ts: pd.DataFrame):
    """Dia anterior à primeira queda > 1% (versões correntes, ou total)."""
    col = "current_bytes" if has(ts, "current_bytes") else "total_bytes"
    if not has(ts, col):
        return None
    s = ts[col].dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return None
    d = s.diff()
    hits = d[d < -0.01 * s.iloc[0]]
    if hits.empty:
        return None
    pos = list(s.index).index(hits.index[0])
    return s.index[pos - 1]


def prefix_pivot(rows: pd.DataFrame, lookup: Dict[str, str], day) -> pd.DataFrame:
    inv = {v: k for k, v in lookup.items()}
    sub = rows[(rows["report_date"] == day) & rows["metric_name"].isin(inv)]
    if sub.empty:
        return pd.DataFrame()
    return (sub.groupby(["norm_prefix", "metric_name"])["metric_value"].sum()
            .unstack("metric_name").rename(columns=inv))


def deep_prefix_analysis(
    r: "Report",
    prefix_rows: pd.DataFrame,
    exact_rows: pd.DataFrame,
    ts_p: pd.DataFrame,
    ts_b: pd.DataFrame,
    prefix: str,
    lookup: Dict[str, str],
    rule_date,
    rule_date_src: str,
    out_dir: Path,
) -> None:
    last = ts_p.index[-1]
    ref = nearest_on_or_before(ts_p.index, rule_date) if rule_date else None
    if ref == last:
        ref = None
    conclusions: List[str] = []

    # ------------------------------------------------------------------ 3.1
    r.sub("3.1 Cobertura dos dados do prefixo")
    p_dates = sorted(ts_p.index)
    b_dates = sorted(ts_b.index) if not ts_b.empty else []
    r.add(f"Dias com o prefixo no export: {len(p_dates)} | dias com o bucket: {len(b_dates)}")
    missing = sorted(set(b_dates) - set(p_dates))
    if missing:
        r.add(f"Dias SEM registro do prefixo ({len(missing)}): "
              f"{', '.join(map(str, missing[:10]))}{' ...' if len(missing) > 10 else ''}")
        r.add("  O Lens só publica um prefixo quando ele passa do threshold (padrão "
              "1% do bucket) e está dentro da profundidade configurada.")
    if b_dates and b_dates[-1] > last:
        r.add(f"ATENÇÃO: o prefixo deixou de aparecer após {last}, mas o bucket tem "
              f"dados até {b_dates[-1]}. Os números do 'último dia' do prefixo podem "
              "estar desatualizados (prefixo caiu abaixo do threshold?).")
        conclusions.append(f"O prefixo sumiu do export após {last}; confirme o residual "
                           "atual pelo nível do bucket ou via S3 Inventory.")
    gaps = [(a, b) for a, b in zip(p_dates, p_dates[1:]) if (b - a).days > 1]
    if gaps:
        r.add("Lacunas no calendário do prefixo: "
              + ", ".join(f"{a}->{b}" for a, b in gaps[:10]))
    r.add("Métricas disponíveis no nível do prefixo:")
    for key, label in (("total_bytes", "StorageBytes"), ("current_bytes", "CurrentVersionStorageBytes"),
                       ("noncurrent_bytes", "NonCurrentVersionStorageBytes"),
                       ("delete_marker_objects", "DeleteMarkerObjectCount"),
                       ("mpu_bytes", "IncompleteMultipartUploadStorageBytes"),
                       ("object_lock_bytes", "ObjectLockEnabledStorageBytes")):
        r.add(f"  {'OK ' if has(ts_p, key) else '-- '} {label}")
    if rule_date:
        r.add(f"Data da regra: {rule_date} ({rule_date_src}) | dia de referência: "
              f"{ref if ref else 'sem dado anterior disponível'} | último dia: {last}")
    else:
        r.add("Data da regra não informada e não foi possível inferir (nenhuma queda >1%).")

    # ------------------------------------------------------------------ 3.2
    r.sub("3.2 Participação do prefixo no bucket")
    outside_last = float("nan")
    for label, day in (("Referência", ref), ("Último dia", last)):
        if day is None:
            continue
        pt = value_at(ts_p, day, "total_bytes")
        bt = value_at(ts_b, day, "total_bytes")
        out = bt - pt if not (pd.isna(bt) or pd.isna(pt)) else float("nan")
        if day == last:
            outside_last = out
        r.add(f"  {label} ({day}): prefixo {fmt_bytes(pt)} = {pct_of(pt, bt)} do bucket "
              f"| fora do prefixo {fmt_bytes(out)}")
    bt_last = value_at(ts_b, last, "total_bytes")
    if not pd.isna(outside_last) and bt_last and outside_last / bt_last >= 0.05:
        conclusions.append(
            f"{fmt_bytes(outside_last)} ({pct_of(outside_last, bt_last)} do bucket) "
            f"estão FORA de {prefix} e não são afetados pela regra. Se o número de "
            "'TB restantes' veio do bucket inteiro, essa parte não é falha do lifecycle.")

    # ------------------------------------------------------------------ 3.3
    keys = [("Total (bytes)", "total_bytes", True), ("Correntes (bytes)", "current_bytes", True),
            ("Não correntes (bytes)", "noncurrent_bytes", True), ("Multipart (bytes)", "mpu_bytes", True),
            ("Object Lock (bytes)", "object_lock_bytes", True), ("Total (objetos)", "total_objects", False),
            ("Correntes (objetos)", "current_objects", False),
            ("Não correntes (objetos)", "noncurrent_objects", False),
            ("Delete markers (objetos)", "delete_marker_objects", False)]
    if ref is not None:
        r.sub(f"3.3 Balanço antes x depois ({ref} -> {last}) — bytes em TiB")
        r.add(f"  {'Métrica':<26}{'Antes':>16}{'Depois':>16}{'Variação':>16}{'Var %':>9}")
        balance = []
        for label, k, is_b in keys:
            if not has(ts_p, k):
                continue
            a, b = value_at(ts_p, ref, k), value_at(ts_p, last, k)
            d = b - a
            f = tib if is_b else fmt_count
            r.add(f"  {label:<26}{f(a):>16}{f(b):>16}{(('+' if d >= 0 else '') + f(d)):>16}"
                  f"{pct_of(d, a):>9}")
            balance.append({"metrica": k, "antes": a, "depois": b, "variacao": d})
        pd.DataFrame(balance).to_csv(out_dir / "prefix_balance.csv", index=False)

    # ------------------------------------------------------------------ 3.4
    r.sub("3.4 Progresso diário da deleção no prefixo (TiB)")
    cols = ["total_bytes", "current_bytes", "noncurrent_bytes", "mpu_bytes", "delete_marker_objects"]
    daily = ts_p[cols].astype(float).copy()
    for c in cols:
        daily["d_" + c] = daily[c].diff()
    daily.to_csv(out_dir / "prefix_daily_progress.csv")
    start = (ref - pd.Timedelta(days=3).to_pytimedelta()) if ref else daily.index[0]
    view = daily[daily.index >= start]
    header = (f"  {'Data':<12}{'Total':>10}{'Corr.':>10}{'Não corr.':>11}"
              f"{'ΔTotal':>10}{'ΔCorr.':>10}{'ΔNãoCorr.':>11}{'ΔDelMarkers':>14}")
    r.add(header)

    def line(day, row):
        mark = "  <- referência" if day == ref else ""
        dm = row["d_delete_marker_objects"]
        return (f"  {str(day):<12}{tib(row['total_bytes']):>10}{tib(row['current_bytes']):>10}"
                f"{tib(row['noncurrent_bytes']):>11}{tib(row['d_total_bytes']):>10}"
                f"{tib(row['d_current_bytes']):>10}{tib(row['d_noncurrent_bytes']):>11}"
                f"{('n/d' if pd.isna(dm) else f'{dm:+,.0f}'):>14}{mark}")

    rows_list = list(view.iterrows())
    if len(rows_list) > 30:
        for day, row in rows_list[:15]:
            r.add(line(day, row))
        r.add(f"  ... ({len(rows_list) - 30} dias omitidos; veja prefix_daily_progress.csv) ...")
        rows_list = rows_list[-15:]
    for day, row in rows_list:
        r.add(line(day, row))

    base_day = ref or daily.index[0]
    base_total = value_at(ts_p, base_day, "total_bytes")
    thr = max(0.001 * nz(base_total), 1024**3)
    after = daily[daily.index > base_day]
    drops = after[after["d_total_bytes"] < -thr]
    if drops.empty:
        r.add("\n  Nenhuma redução relevante após o dia de referência.")
        if ref is not None:
            conclusions.append("A regra não reduziu o prefixo após a data informada: "
                               "verifique se a regra está Enabled e se o filtro bate com as chaves.")
    else:
        first_drop, last_drop = drops.index[0], drops.index[-1]
        removed = base_total - value_at(ts_p, last_drop, "total_bytes")
        days = max((last_drop - base_day).days, 1)
        r.add(f"\n  Remoção relevante entre {first_drop} e {last_drop}: {fmt_bytes(removed)} "
              f"em {days} dia(s) (~{fmt_bytes(removed / days)}/dia).")
        if last_drop == last:
            r.add("  A deleção AINDA ESTÁ EM ANDAMENTO no último dia disponível.")
            conclusions.append("A deleção ainda estava ocorrendo no último dia do export: "
                               "parte do residual pode simplesmente não ter sido processada ainda.")
        else:
            r.add(f"  Estável desde {last_drop} ({(last - last_drop).days} dia(s) sem "
                  "remoção relevante): o residual não será removido pela regra atual.")
    nc_growth = after["d_noncurrent_bytes"].clip(lower=0).sum() if has(ts_p, "noncurrent_bytes") else 0
    if nc_growth > thr:
        r.add(f"  Versões não correntes CRESCERAM {fmt_bytes(nc_growth)} durante o período "
              "(objetos expirados que viraram noncurrent em vez de serem apagados).")

    # ------------------------------------------------------------------ 3.5
    r.sub("3.5 Inferência do estado do versionamento durante a execução da regra")
    vs = {"nc_new": float("nan")}
    if has(ts_p, "current_bytes") and has(ts_p, "noncurrent_bytes"):
        sts, scope = ts_p, f"prefixo {prefix}"
    else:
        sts, scope = ts_b, "bucket inteiro (métricas de versão ausentes no prefixo)"
    sref = nearest_on_or_before(sts.index, ref) if ref is not None else None
    if sref is None or sts.empty:
        r.add("  Sem dia de referência anterior à regra: não é possível comparar. "
              "Informe --rule-date ou use exports anteriores à regra.")
    else:
        slast = sts.index[-1]
        g = lambda d, k: value_at(sts, d, k)  # noqa: E731
        dc = g(sref, "current_bytes") - g(slast, "current_bytes")
        dn = g(slast, "noncurrent_bytes") - g(sref, "noncurrent_bytes")
        ddm = g(slast, "delete_marker_objects") - g(sref, "delete_marker_objects")
        dco = g(sref, "current_objects") - g(slast, "current_objects")
        dno = g(slast, "noncurrent_objects") - g(sref, "noncurrent_objects")
        r.add(f"  Escopo: {scope} | {sref} -> {slast}")
        r.add(f"  Versões correntes removidas ........ {fmt_bytes(dc)} | {fmt_count(dco)} objetos")
        r.add(f"  Variação das não correntes ......... {fmt_bytes(dn)} | {fmt_count(dno)} objetos")
        r.add(f"  Delete markers criados ............. {fmt_count(ddm)}")
        if not pd.isna(ddm) and not pd.isna(dco) and dco > 0:
            r.add(f"  Delete markers por objeto expirado . {ddm / dco:.2f}")
        cur_ref = g(sref, "current_bytes")
        vs["nc_new"] = max(nz(dn), 0)

        if pd.isna(dc) or pd.isna(dn):
            state, expl = "INDETERMINADO", "métricas de versão incompletas."
        elif dc <= 0.01 * nz(cur_ref):
            state = "SEM EFEITO"
            expl = ("a regra praticamente não removeu versões correntes. Verifique "
                    "status da regra, filtro (prefixo/tags/tamanho) e Object Lock.")
        else:
            ratio = max(dn, 0) / dc
            dm_up = pd.isna(ddm) or ddm > 0
            if ratio >= 0.9:
                state = "ENABLED (versionamento ATIVO)"
                expl = (f"{ratio:.0%} do que saiu das versões correntes virou não corrente: "
                        "a expiração só criou delete markers e nada foi apagado de fato.")
            elif dm_up and ratio > 0.01:
                state = "SUSPENDED, com objetos herdados do período Enabled"
                expl = (f"apenas {ratio:.1%} ({fmt_bytes(dn)}) do expirado virou não corrente. "
                        "Com versionamento Suspended, a expiração apaga de fato as versões "
                        "'null' (criadas após a suspensão), mas versões correntes com "
                        "VersionId real (criadas quando estava Enabled) apenas viram não "
                        "correntes atrás de um delete marker. Isso é consistente com "
                        "arquivos antigos (ex.: 2022) restantes.")
            elif dm_up:
                state = "SUSPENDED"
                expl = ("os objetos expirados eram versões 'null' e foram apagados; o "
                        "residual não corrente já existia antes da regra.")
            else:
                state = "NÃO VERSIONADO (comportamento observado)"
                expl = ("objetos removidos sem gerar delete markers.")
                if nz(g(sref, "noncurrent_bytes")) > 0:
                    expl += (" Porém há versões não correntes, o que só existe se o "
                             "versionamento já foi habilitado: confira se a métrica de delete "
                             "markers está no export e o atraso do Lens.")
        r.add(f"\n  Estado inferido: {state}")
        r.add(f"  Motivo: {expl}")
        conclusions.insert(0, f"Versionamento inferido durante a regra: {state}.")

    # ------------------------------------------------------------------ 3.6
    r.sub(f"3.6 Decomposição do residual em {last}")
    tot_l = value_at(ts_p, last, "total_bytes")
    cur_l = value_at(ts_p, last, "current_bytes")
    nc_l = value_at(ts_p, last, "noncurrent_bytes")
    mpu_l = value_at(ts_p, last, "mpu_bytes")
    lock_l = value_at(ts_p, last, "object_lock_bytes")
    comps = []
    if ref is not None and has(ts_p, "noncurrent_bytes"):
        nc_ref = value_at(ts_p, ref, "noncurrent_bytes")
        nc_new = max(nz(nc_l) - nz(nc_ref), 0)
        comps.append(("Não correntes já existentes antes da regra", nz(nc_l) - nc_new,
                      "NoncurrentVersionExpiration"))
        comps.append(("Não correntes geradas pela própria regra", nc_new,
                      "NoncurrentVersionExpiration"))
    else:
        comps.append(("Versões não correntes (origem indeterminada)", nz(nc_l),
                      "NoncurrentVersionExpiration"))
    comps.append(("Versões correntes não expiradas", nz(cur_l),
                  "revisar filtro/status da regra, Object Lock ou aguardar"))
    comps.append(("Uploads multipart incompletos", nz(mpu_l), "AbortIncompleteMultipartUpload"))
    explained = sum(c[1] for c in comps)
    unexplained = nz(tot_l) - explained
    r.add(f"  Total do prefixo: {fmt_bytes(tot_l)}")
    for name, v, fix in sorted(comps, key=lambda c: -c[1]):
        r.add(f"  {name:<46}{fmt_bytes(v):>32}  {pct_of(v, tot_l):>7}  -> {fix}")
    r.add(f"  {'Diferença não explicada':<46}{fmt_bytes(unexplained):>32}  {pct_of(unexplained, tot_l):>7}")
    r.add("  (StorageBytes inclui metadados e pode incluir MPU; diferenças pequenas são normais.)")
    if not pd.isna(lock_l) and lock_l > 0:
        r.add(f"  Destes, {fmt_bytes(lock_l)} ({pct_of(lock_l, tot_l)}) têm Object Lock habilitado: "
              "versões sob retenção/legal hold não são apagadas até expirar a retenção.")
        conclusions.append(f"{fmt_bytes(lock_l)} com Object Lock no prefixo: verifique "
                           "retenções e legal holds antes de mudar a regra.")
    if tot_l and abs(unexplained) / tot_l > 0.05:
        conclusions.append(f"{fmt_bytes(unexplained)} não explicados pelas métricas de versão: "
                           "confira se todas as métricas estão no export e se há mais de "
                           "uma configuração do Lens (--config-id).")
    top = max(comps, key=lambda c: c[1])
    if tot_l and top[1] > 0:
        conclusions.append(f"Maior parcela do residual: {top[0]} = {fmt_bytes(top[1])} "
                           f"({pct_of(top[1], tot_l)}). Correção: {top[2]}.")

    # ------------------------------------------------------------------ 3.7
    r.sub("3.7 Residual por storage class (prefixo)")
    sc_l = by_storage_class(exact_rows, lookup, last)
    sc_r = by_storage_class(exact_rows, lookup, ref) if ref is not None else pd.DataFrame()
    if sc_l.empty:
        r.add("  Sem dados por storage class para o prefixo.")
    else:
        tab = pd.DataFrame(index=sorted(set(sc_l.index) | set(sc_r.index)))
        for k in ("total_bytes", "current_bytes", "noncurrent_bytes"):
            tab[k + "_ultimo"] = sc_l.get(k, pd.Series(dtype=float)).reindex(tab.index)
        tab["total_bytes_ref"] = (sc_r.get("total_bytes", pd.Series(dtype=float)).reindex(tab.index)
                                  if not sc_r.empty else float("nan"))
        tab.to_csv(out_dir / "prefix_residual_by_storage_class.csv")
        r.add(f"  {'Classe':<22}{'Antes':>10}{'Depois':>10}{'Removido':>10}"
              f"{'Corr.':>10}{'Não corr.':>11}  (TiB)")
        cold_hits = []
        for cls, row in tab.sort_values("total_bytes_ultimo", ascending=False).iterrows():
            a, b = row["total_bytes_ref"], row["total_bytes_ultimo"]
            rem = (1 - b / a) if a and not pd.isna(a) and not pd.isna(b) else float("nan")
            r.add(f"  {cls:<22}{tib(a):>10}{tib(b):>10}"
                  f"{('n/d' if pd.isna(rem) else f'{rem:.0%}'):>10}"
                  f"{tib(row['current_bytes_ultimo']):>10}{tib(row['noncurrent_bytes_ultimo']):>11}")
            if COLD_CLASSES.get(cls, 0) > 0 and nz(b) > 0:
                cold_hits.append(cls)
        if cold_hits:
            r.add("  Classes com duração mínima de cobrança no residual: "
                  + ", ".join(f"{c} ({COLD_CLASSES[c]}d)" for c in cold_hits)
                  + ". Isso NÃO impede a deleção, mas gera cobrança proporcional se "
                  "apagado antes do mínimo. Não correntes em GLACIER/DEEP_ARCHIVE "
                  "costumam indicar uma regra NoncurrentVersionTransition existente.")

    # ------------------------------------------------------------------ 3.8
    r.sub(f"3.8 Onde está o residual: subprefixos de {prefix}")
    base_depth = prefix.count("/")
    subs = prefix_rows[prefix_rows["norm_prefix"].str.startswith(prefix)
                       & (prefix_rows["norm_prefix"] != prefix)].copy()
    if subs.empty:
        r.add("  Nenhum subprefixo no export. Aumente a profundidade (prefix depth) e/ou "
              "reduza o threshold do prefix aggregation no Lens para detalhar.")
    else:
        p_l = prefix_pivot(subs, lookup, last)
        p_r = prefix_pivot(subs, lookup, ref) if ref is not None else pd.DataFrame()
        det = pd.DataFrame(index=sorted(set(p_l.index) | set(p_r.index)))
        det["depth"] = [p.count("/") for p in det.index]
        det["total_ref"] = (p_r.get("total_bytes", pd.Series(dtype=float)).reindex(det.index)
                            if not p_r.empty else float("nan"))
        for k, c in (("total_bytes", "total_last"), ("current_bytes", "current_last"),
                     ("noncurrent_bytes", "noncurrent_last"), ("mpu_bytes", "mpu_last")):
            det[c] = (p_l.get(k, pd.Series(dtype=float)).reindex(det.index)
                      if not p_l.empty else float("nan"))
        det["removido_pct"] = 1 - det["total_last"] / det["total_ref"]
        det["share_residual"] = det["total_last"] / tot_l if tot_l else float("nan")
        det["causa_dominante"] = [dominant_cause(a, b, c) for a, b, c in
                                  zip(det["current_last"], det["noncurrent_last"], det["mpu_last"])]
        det.sort_values(["depth", "total_last"], ascending=[True, False]).to_csv(
            out_dir / "prefix_subprefixes_detail.csv")

        def show(level_df, n):
            w = max([12] + [len(p) for p in level_df.index]) + 2
            r.add(f"  {'Subprefixo':<{w}}{'Antes':>9}{'Depois':>9}{'Remov.':>8}"
                  f"{'%resid.':>9}  Causa dominante   (TiB)")
            for p, row in level_df.sort_values("total_last", ascending=False).head(n).iterrows():
                rem = row["removido_pct"]
                r.add(f"  {p:<{w}}{tib(row['total_ref']):>9}{tib(row['total_last']):>9}"
                      f"{('n/d' if pd.isna(rem) else f'{rem:.0%}'):>8}"
                      f"{('n/d' if pd.isna(row['share_residual']) else f'{row.share_residual:.1%}'):>9}"
                      f"  {row['causa_dominante']}")

        direct = det[det["depth"] == base_depth + 1]
        if not direct.empty:
            r.add(f"Nível imediato ({len(direct)} subprefixo(s)), ordenado pelo residual:")
            show(direct, 25)
            cov = direct["total_last"].sum() / tot_l if tot_l else float("nan")
            r.add(f"  Residual atribuído aos subprefixos diretos: {pct_of(direct['total_last'].sum(), tot_l)}"
                  " (o restante está em objetos direto no prefixo ou em subprefixos abaixo do threshold).")
            if not pd.isna(cov) and cov > 0:
                top_sub = direct["total_last"].idxmax()
                conclusions.append(
                    f"Residual concentrado em {top_sub} ({pct_of(direct.at[top_sub, 'total_last'], tot_l)} "
                    f"do prefixo; causa dominante: {direct.at[top_sub, 'causa_dominante']}).")
            low = direct[(direct["removido_pct"] < 0.5) & (direct["total_ref"] > thr)]
            if not low.empty:
                r.add("  Subprefixos com remoção < 50% (comportamento diferente do resto): "
                      + ", ".join(low.index[:10]))
                conclusions.append(
                    "Subprefixos com remoção abaixo de 50%: " + ", ".join(low.index[:5])
                    + ". Compare com os que foram limpos: idade dos objetos (período em que o "
                      "versionamento estava Enabled), tags, Object Lock ou storage class.")
        for depth in sorted(d for d in det["depth"].unique() if d > base_depth + 1)[:3]:
            lvl = det[det["depth"] == depth]
            r.add(f"\nNível {depth - base_depth} abaixo de {prefix} — top 10 ({len(lvl)} no total):")
            show(lvl, 10)

    # ------------------------------------------------------------------ 3.9
    r.sub(f"3.9 Conclusão para o prefixo {prefix}")
    for i, c in enumerate(conclusions, 1):
        r.add(f"  {i}. {c}")
    if not conclusions:
        r.add("  Sem conclusões automáticas; revise as tabelas acima.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Investiga por que uma lifecycle rule não removeu todos os "
                    "dados de um bucket, usando exports CSV do S3 Storage Lens."
    )
    ap.add_argument("--input", "-i", required=True, help="Diretório (ou arquivo) com os CSVs do Lens")
    ap.add_argument("--bucket", "-b", required=True, help="Nome do bucket a investigar")
    ap.add_argument("--prefix", "-p", default="exp/", help="Prefixo da lifecycle rule (default: exp/)")
    ap.add_argument("--rule-date", help="Data de aplicação da regra (YYYY-MM-DD) para comparação antes/depois")
    ap.add_argument("--output", "-o", default="./lens_investigation_output", help="Diretório de saída")
    ap.add_argument("--config-id", help="Filtrar uma única configuração do Lens (configuration_id)")
    ap.add_argument("--chunksize", type=int, default=500_000, help="Linhas por chunk na leitura")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = normalize_prefix(args.prefix)
    rule_date = pd.to_datetime(args.rule_date).date() if args.rule_date else None

    files = find_files(in_path)
    if not files:
        print(f"Nenhum CSV encontrado em {in_path}", file=sys.stderr)
        return 1
    print(f"Encontrados {len(files)} arquivo(s). Filtrando bucket '{args.bucket}'...",
          file=sys.stderr)

    df = read_bucket_rows(files, args.bucket, args.chunksize)
    r = Report()

    r.h(f"INVESTIGAÇÃO DE LIFECYCLE - bucket: {args.bucket} | prefixo: {prefix}")
    if df.empty:
        r.add("Nenhuma linha encontrada para este bucket. Verifique o nome do "
              "bucket e se ele está incluído no escopo da configuração do Lens.")
        r.save(out_dir / "report.txt")
        return 2

    if args.config_id:
        df = df[df["configuration_id"] == args.config_id]
        if df.empty:
            r.add(f"Nenhuma linha para configuration_id={args.config_id}.")
            r.save(out_dir / "report.txt")
            return 2

    df.to_csv(out_dir / "bucket_raw_rows.csv.gz", index=False, compression="gzip")

    dates = sorted(d for d in df["report_date"].dropna().unique())
    r.add(f"Linhas do bucket: {len(df):,}")
    r.add(f"Período coberto: {dates[0]} -> {dates[-1]} ({len(dates)} dia(s))")
    r.add(f"Contas: {', '.join(sorted(df['aws_account_number'].unique()))}")
    r.add(f"Regiões: {', '.join(sorted(df['aws_region'].unique()))}")
    r.add(f"Configurações Lens: {', '.join(sorted(df['configuration_id'].unique()))}")
    r.add(f"record_type presentes: {', '.join(sorted(df['record_type'].unique()))}")
    r.add(f"Storage classes: {', '.join(sorted(df['storage_class'].unique()))}")
    if len(df["configuration_id"].unique()) > 1:
        r.add("ATENÇÃO: mais de uma configuração do Lens. Os valores serão SOMADOS "
              "entre configurações (duplicação); use --config-id para escolher uma.")

    metric_names = sorted(df["metric_name"].unique())
    (out_dir / "metric_names.txt").write_text("\n".join(metric_names) + "\n", encoding="utf-8")
    lookup = build_metric_lookup(metric_names)
    missing = [k for k in METRICS if k not in lookup]
    r.add(f"Métricas distintas: {len(metric_names)} (lista em metric_names.txt)")
    if missing:
        r.add(f"Métricas de interesse não encontradas: {', '.join(missing)}")

    # ---------------- Nível BUCKET ----------------
    bucket_df = df[df["record_type"] == "BUCKET"]
    ts_bucket = timeseries(bucket_df, lookup)
    if not ts_bucket.empty:
        ts_bucket.to_csv(out_dir / "timeseries_bucket.csv")

    r.h("1. VISÃO DO BUCKET INTEIRO")
    if ts_bucket.empty:
        r.add("Sem registros record_type=BUCKET.")
    else:
        describe_snapshot(r, "Primeiro dia disponível", ts_bucket, ts_bucket.index[0])
        describe_snapshot(r, "Último dia disponível", ts_bucket, ts_bucket.index[-1])
        sc = by_storage_class(bucket_df, lookup, ts_bucket.index[-1])
        if not sc.empty:
            sc.to_csv(out_dir / "latest_by_storage_class.csv")
            r.add("\nPor storage class (último dia):")
            for cls, row in sc.iterrows():
                r.add(f"  {cls:<22} total {fmt_bytes(row.get('total_bytes', 0)):>30} | "
                      f"noncurrent {fmt_bytes(row.get('noncurrent_bytes', 0))}")

    # ---------------- Nível PREFIX ----------------
    r.h(f"2. VISÃO DO PREFIXO {prefix}")
    prefix_df = df[df["record_type"] == "PREFIX"].copy()
    ts_prefix = pd.DataFrame()
    if prefix_df.empty:
        r.add("Sem registros record_type=PREFIX. Métricas por prefixo exigem "
              "Storage Lens advanced com 'prefix aggregation' habilitado. A "
              "análise seguirá apenas no nível do bucket.")
    else:
        prefix_df["norm_prefix"] = prefix_df["record_value"].map(normalize_prefix)
        exact = prefix_df[prefix_df["norm_prefix"] == prefix]
        ts_prefix = timeseries(exact, lookup)
        if ts_prefix.empty:
            r.add(f"Prefixo {prefix} não aparece no export. O Lens só publica "
                  "prefixos acima do threshold configurado (padrão 1% do bucket) "
                  "e até a profundidade configurada.")
        else:
            ts_prefix.to_csv(out_dir / "timeseries_prefix.csv")
            describe_snapshot(r, "Primeiro dia disponível", ts_prefix, ts_prefix.index[0])
            describe_snapshot(r, "Último dia disponível", ts_prefix, ts_prefix.index[-1])

        # Outros prefixos de 1º nível fora do escopo da regra.
        last = max(prefix_df["report_date"])
        top = prefix_df[(prefix_df["report_date"] == last)
                        & (prefix_df["metric_name"] == lookup.get("total_bytes"))].copy()
        top["depth"] = top["norm_prefix"].str.count("/")
        top = top[top["depth"] == 1].groupby("norm_prefix")["metric_value"].sum()
        top = top.sort_values(ascending=False)
        if not top.empty:
            r.add(f"\nPrefixos de 1º nível no último dia ({last}):")
            for p, v in top.head(20).items():
                tag = "  <- escopo da regra" if p == prefix else ""
                r.add(f"  {p:<45} {fmt_bytes(v)}{tag}")

    # ---------------- Data da regra ----------------
    rule_date_src = "informada via --rule-date"
    if rule_date is None:
        rule_date = infer_rule_date(ts_prefix if not ts_prefix.empty else ts_bucket)
        rule_date_src = "INFERIDA: dia anterior à primeira queda > 1% das versões correntes"

    # ---------------- Análise aprofundada ----------------
    r.h(f"3. ANÁLISE APROFUNDADA DO PREFIXO {prefix}")
    if ts_prefix.empty:
        r.add("Indisponível: o prefixo não tem registros PREFIX no export.")
    else:
        deep_prefix_analysis(r, prefix_df, exact, ts_prefix, ts_bucket, prefix, lookup,
                             rule_date, rule_date_src, out_dir)

    # ---------------- Versionamento ----------------
    r.h("4. LINHA DO TEMPO DE VERSIONAMENTO")
    versioning_timeline(r, ts_prefix if not ts_prefix.empty else ts_bucket)

    # ---------------- Diagnóstico ----------------
    r.h("5. DIAGNÓSTICO DO BUCKET INTEIRO")
    diagnose(r, ts_bucket, "bucket inteiro", rule_date)

    # ---------------- Próximos passos ----------------
    r.h("6. PRÓXIMOS PASSOS SUGERIDOS")
    r.add(f"""\
  * Confirmar estado do versionamento:
      aws s3api get-bucket-versioning --bucket {args.bucket}
  * Ver a regra aplicada:
      aws s3api get-bucket-lifecycle-configuration --bucket {args.bucket}
  * Histórico de mudanças de versionamento (CloudTrail, evento PutBucketVersioning).
  * Amostrar versões antigas para confirmar datas (2022+):
      aws s3api list-object-versions --bucket {args.bucket} --prefix {prefix} \\
        --max-items 1000 --query 'Versions[?IsLatest==`false`].[Key,VersionId,LastModified,Size]'
  * Para inventário completo, use S3 Inventory com 'Include all versions'
    e consulte via Athena para confirmar idade e tipo do residual:
      SELECT year(last_modified_date) AS ano, is_latest, is_delete_marker,
             storage_class, count(*) AS objetos,
             sum(size)/power(1024,4) AS tib
      FROM inventario
      WHERE key LIKE '{prefix}%'
      GROUP BY 1,2,3,4 ORDER BY 1;
  * Regra corrigida sugerida para o prefixo {prefix}:
      {{
        "ID": "expire-{prefix.strip('/')}",
        "Filter": {{"Prefix": "{prefix}"}},
        "Status": "Enabled",
        "Expiration": {{"Days": 1}},
        "NoncurrentVersionExpiration": {{"NoncurrentDays": 1}},
        "AbortIncompleteMultipartUpload": {{"DaysAfterInitiation": 1}}
      }}
    e, depois que as versões não correntes sumirem, uma regra com
    "Expiration": {{"ExpiredObjectDeleteMarker": true}} para limpar delete markers.""")

    r.save(out_dir / "report.txt")
    print(f"\nArquivos gerados em: {out_dir.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
