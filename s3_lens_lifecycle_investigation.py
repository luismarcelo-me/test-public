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
    p = (p or "").strip().lstrip("/")
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
            ts[k] = pd.NA
    return ts[list(METRICS)]


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
        r.add("ATENÇÃO: mais de uma configuração do Lens. Os valores podem estar "
              "duplicados entre configurações; considere filtrar uma só.")

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

        # Subprefixos (NÃO somar: níveis aninhados se sobrepõem).
        last = max(prefix_df["report_date"])
        sub = prefix_df[(prefix_df["report_date"] == last)
                        & prefix_df["norm_prefix"].str.startswith(prefix)
                        & (prefix_df["metric_name"].isin(
                            [lookup.get("total_bytes"), lookup.get("noncurrent_bytes")]))]
        if not sub.empty:
            inv = {v: k for k, v in lookup.items()}
            piv = (sub.groupby(["norm_prefix", "metric_name"])["metric_value"].sum()
                   .unstack("metric_name").rename(columns=inv).fillna(0)
                   .sort_values("total_bytes", ascending=False))
            piv.to_csv(out_dir / "latest_subprefixes.csv")
            r.add(f"\nPrefixos sob {prefix} no último dia ({last}) - top 20 "
                  "(níveis aninhados se sobrepõem, não some):")
            for p, row in piv.head(20).iterrows():
                r.add(f"  {p:<45} total {fmt_bytes(row.get('total_bytes', 0)):>28} | "
                      f"noncurrent {fmt_bytes(row.get('noncurrent_bytes', 0))}")

        # Outros prefixos de 1º nível fora do escopo da regra.
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

    # ---------------- Versionamento ----------------
    r.h("3. LINHA DO TEMPO DE VERSIONAMENTO")
    versioning_timeline(r, ts_prefix if not ts_prefix.empty else ts_bucket)

    # ---------------- Diagnóstico ----------------
    r.h("4. DIAGNÓSTICO")
    if not ts_prefix.empty:
        diagnose(r, ts_prefix, f"prefixo {prefix}", rule_date)
        r.add("")
        r.add("-" * 78)
    diagnose(r, ts_bucket, "bucket inteiro", rule_date)

    # ---------------- Próximos passos ----------------
    r.h("5. PRÓXIMOS PASSOS SUGERIDOS")
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
    e consulte via Athena (IsLatest, IsDeleteMarker, LastModified, ObjectLock*).
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
