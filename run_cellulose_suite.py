#!/usr/bin/env python3
"""Cellulose full suite: correctness + scalability, then organize all useful logs.

Phase 1 (correctness / 正确性验证): run cellulose_validate.py -- a moderate-length
chain computed TWO ways: (a) unified = one full-chain SIESTA job (E_full), and
(b) fragmented = MFCC blocks + dimers + caps, combined into E_MFCC(1)/E_MBE(2).
Verdict: E_MBE(2) should reproduce E_full.

Phase 2 (scalability / 拓展性测试): run weak_scale_cellulose_all.py -- the
1/2/4/8/16-node weak-scaling sweep (each node = N glucose, ntpoly/TRS2).

Phase 3: organize into one folder:
  <out-root>/
    correctness/
      unified/            # 统一计算 (full chain) : siesta.out, input.fdf, energy
      fragmented/         # 分别计算 (MFCC pieces) : block/cap/dimer siesta.out
                          #   + result.json (E_full, E_MFCC, E_MBE(2), err/cut)
                          #   + combine.log (cellulose_validate stdout)
    scalability/          # 拓展性 : n01/..n16/, each with logs + schedule.json
                          #   + weak_scaling_results.json ; + weak_scale_summary.json
    summary.txt           # correctness verdict + scalability table

Usage (intranet, defaults = PE parity):
  python3 run_cellulose_suite.py --out-root /share/.../cellulose_results
  # correctness on 16 glucose; scalability 1/2/4/8/16 x 296 glucose/node, ntpoly,
  # submit_per_node_local.sh. --hosts defaults to the 16 intranet nodes.

Usage (dev smoke test):
  python3 run_cellulose_suite.py --out-root /tmp/cellu_suite \
      --correctness-glucose 8 --glucose-per-node 8 --nodes 1 2 \
      --executor dev-local --correctness-solver diagonali
"""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
TOTAL_RE = re.compile(r"siesta:\s+Total\s*=\s*(-?[\d.]+)")


def run_correctness(args, work):
    print("\n" + "#" * 70 + "\n# Phase 1: CORRECTNESS (cellulose_validate)\n" + "#" * 70, flush=True)
    cmd = [sys.executable, str(REPO / "cellulose_validate.py"),
           "--glucose", str(args.correctness_glucose), "--cuts", "1", "2",
           "--work-root", str(work), "--siesta-bin", args.siesta_bin,
           "--pseudo-dir", str(REPO / "pseudos"),
           "--solution-method", args.correctness_solver, "--timeout", str(args.timeout)]
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print("⚠ cellulose_validate rc=", p.returncode, "\n", p.stderr[-2000:], flush=True)
    (work / "combine.log").write_text(p.stdout + ("\n[stderr]\n" + p.stderr[-3000:]))
    return p.stdout


def run_scalability(args, work):
    print("\n" + "#" * 70 + "\n# Phase 2: SCALABILITY (weak_scale_cellulose_all)\n" + "#" * 70, flush=True)
    cmd = [sys.executable, str(REPO / "weak_scale_cellulose_all.py"),
           "--glucose-per-node", str(args.glucose_per_node),
           "--nodes", *map(str, args.nodes),
           "--big-out", str(work), "--executor", args.executor]
    if args.executor == "dev-local":
        cmd += ["--dev-procs", str(args.dev_procs)]
    if args.remote_out_dir:
        cmd += ["--remote-out-dir", args.remote_out_dir]
    if args.hosts:
        cmd += ["--hosts", *args.hosts]
    subprocess.run(cmd, cwd=str(REPO), check=False)


def parse_energy(siout: Path):
    if not siout.exists():
        return None
    ms = TOTAL_RE.findall(siout.read_text(errors="ignore"))
    return float(ms[-1]) if ms else None


def _organize_correctness(corr_work: Path, out: Path, stdout_txt: str, args):
    # ---- correctness/unified (full-chain LOG only; no heavy SIESTA outputs) ----
    uni = out / "correctness" / "unified"; uni.mkdir(parents=True, exist_ok=True)
    if (corr_work / "full" / "siesta.out").exists():
        shutil.copy2(corr_work / "full" / "siesta.out", uni / "siesta.out")
    e_full = parse_energy(uni / "siesta.out")

    # ---- correctness/fragmented (MFCC pieces) ----
    frag = out / "correctness" / "fragmented"; frag.mkdir(parents=True, exist_ok=True)
    if (corr_work / "combine.log").exists():
        shutil.copy2(corr_work / "combine.log", frag / "combine.log")
    per_job = {}
    for d in sorted(corr_work.iterdir()):
        if d.is_dir() and (d / "siesta.out").exists():
            shutil.copy2(d / "siesta.out", frag / f"{d.name}.siesta.out")
            per_job[d.name] = parse_energy(d / "siesta.out")
    result = {"E_full_ev": e_full, "solver": args.correctness_solver,
              "glucose": args.correctness_glucose, "per_job_ev": per_job, "cuts": []}
    cur = None
    for ln in stdout_txt.splitlines():
        m = re.search(r"=== (\d+) cut\(s\) ===", ln)
        if m:
            cur = {"ncuts": int(m.group(1))}; result["cuts"].append(cur)
        if cur:
            mm = re.search(r"MFCC\(1\):\s*([+-]?[\d.]+)\s+err=([+-]?[\d.]+)", ln)
            if mm:
                cur["E_mfcc_ev"] = float(mm.group(1)); cur["mfcc_err_ev"] = float(mm.group(2))
            mm = re.search(r"MBE\(2\)\s*:\s*([+-]?[\d.]+)\s+err=([+-]?[\d.]+)\s+\(([+-]?[\d.]+)/cut\)", ln)
            if mm:
                cur["E_mbe2_ev"] = float(mm.group(1))
                cur["mbe2_err_ev"] = float(mm.group(2))
                cur["mbe2_err_per_cut_ev"] = float(mm.group(3))
    (out / "correctness" / "fragmented" / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def organize(out: Path, corr_work, scal_work: Path, stdout_txt: str, args, skip_correctness=False):
    print("\n" + "#" * 70 + "\n# Phase 3: ORGANIZE\n" + "#" * 70, flush=True)
    if skip_correctness:
        rj = out / "correctness" / "fragmented" / "result.json"
        result = json.loads(rj.read_text()) if rj.exists() else {"E_full_ev": None, "cuts": []}
        print(f"[skip] correctness already done -> reuse {out / 'correctness'}", flush=True)
    else:
        result = _organize_correctness(corr_work, out, stdout_txt, args)

    # ---- scalability (LOGS + jsons only: no .ion/.BONDS/.XV/_pseudos/etc.) ----
    scal_dst = out / "scalability"; scal_dst.mkdir(parents=True, exist_ok=True)
    if scal_work.exists():
        for item in sorted(scal_work.iterdir()):
            if item.name == "weak_scale_summary.json":
                shutil.copy2(item, scal_dst / item.name)
            elif item.is_dir():                     # nXX/
                nd = scal_dst / item.name; nd.mkdir(exist_ok=True)
                for j in ("schedule.json", "weak_scaling_results.json"):
                    if (item / j).exists():
                        shutil.copy2(item / j, nd / j)
                for jd in sorted(item.iterdir()):   # block_*/dimer_*/cap_*/full
                    if jd.is_dir() and (jd / "siesta.out").exists():
                        shutil.copy2(jd / "siesta.out", nd / f"{jd.name}.siesta.out")

    # ---- summary.txt ----
    write_summary(out, result, scal_dst)


def write_summary(out, corr_result, scal_dst):
    lines = ["CELLULOSE I-beta SUITE — SUMMARY", "=" * 60, "",
             "## Correctness (MFCC+MBE(2) vs unified full chain)", ""]
    ef = corr_result["E_full_ev"]
    lines.append(f"  chain          : {corr_result['glucose']} glucose, solver {corr_result['solver']}")
    lines.append(f"  E_full (unified): {ef} eV" if ef is not None else "  E_full: FAILED")
    for c in corr_result["cuts"]:
        nc = c.get("ncuts")
        lines.append(f"  --- {nc} cut(s) ---")
        lines.append(f"    MFCC(1): {c.get('E_mfcc_ev')}  err={c.get('mfcc_err_ev')} eV")
        lines.append(f"    MBE(2) : {c.get('E_mbe2_ev')}  err={c.get('mbe2_err_ev')} eV "
                     f"({c.get('mbe2_err_per_cut_ev')}/cut)")
        if c.get("mbe2_err_per_cut_ev") is not None:
            lines.append(f"    verdict: {'PASS' if abs(c['mbe2_err_per_cut_ev']) < 0.1 else 'CHECK'} "
                         f"(|err/cut| {'<' if abs(c['mbe2_err_per_cut_ev']) < 0.1 else '>='} 0.1 eV)")
    lines += ["", "## Scalability (weak-scale sweep)", ""]
    summ = scal_dst / "weak_scale_summary.json"
    if summ.exists():
        rows = json.loads(summ.read_text())
        lines.append(f"  {'nodes':>5} {'glu/node':>8} {'atoms':>7} {'method':>8} {'E_total(eV)':>15} {'E/glu':>12}")
        for r in rows:
            e = f"{r['E_total_ev']:.3f}" if r["E_total_ev"] is not None else "FAIL"
            per = f"{r['E_per_glucose_ev']:.5f}" if r["E_per_glucose_ev"] is not None else "-"
            lines.append(f"  {r['nodes']:>5} {r['glucose_per_node']:>8} {str(r['total_atoms']):>7} "
                         f"{str(r['method']):>8} {e:>15} {per:>12}")
    else:
        lines.append("  (no scalability summary found)")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, help="output folder (gets correctness/ scalability/ summary.txt)")
    ap.add_argument("--correctness-glucose", type=int, default=16, help="chain length for correctness test")
    ap.add_argument("--correctness-solver", default="ntpoly", choices=["ntpoly", "diagonali"])
    ap.add_argument("--glucose-per-node", type=int, default=296, help="scalability per-node glucose (296=PE parity)")
    ap.add_argument("--nodes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--siesta-bin",
                    default="/share/honpas/xzz/siesta-20260520/siesta/build-clang/Src/siesta")
    ap.add_argument("--executor", choices=["submit", "dev-local"], default="submit")
    ap.add_argument("--dev-procs", type=int, default=8)
    ap.add_argument("--hosts", nargs="+", default=None)
    ap.add_argument("--remote-out-dir", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    # skip Phase 1 if correctness already organized in a previous run
    corr_done = (out / "correctness" / "fragmented" / "result.json").exists()

    raw = out / "_raw"; raw.mkdir(exist_ok=True)
    corr_work = None
    stdout_txt = ""
    if corr_done:
        print(f"\n[skip] correctness already done -> reuse {out / 'correctness'}\n", flush=True)
    else:
        corr_work = raw / "correctness"; corr_work.mkdir(exist_ok=True)
        stdout_txt = run_correctness(args, corr_work)
    scal_work = raw / "scalability"

    run_scalability(args, scal_work)
    organize(out, corr_work, scal_work, stdout_txt, args, skip_correctness=corr_done)

    # clean raw (keep only the organized tree)
    shutil.rmtree(raw, ignore_errors=True)
    print(f"\n[DONE] organized results -> {out}")
    print(f"  correctness/unified + correctness/fragmented (+ result.json)")
    print(f"  scalability/n01..n{{:02d}} (+ schedule.json, weak_scaling_results.json each)".format(max(args.nodes)))
    print(f"  summary.txt")


if __name__ == "__main__":
    main()
