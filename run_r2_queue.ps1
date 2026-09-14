<#
R2 two-GPU job queue.

One worker = one PowerShell process pinned to one GPU.  Jobs are taken from a plain
text file, one full command line per line, and split across workers by line index
modulo:  worker i of k runs the lines whose kept-index satisfies  idx % k == i.

Job file format
---------------
  # comments and blank lines are ignored
  run_m3.py --task henon --algo skrtrl-r16 ... --outdir results/r2/d4_tune --tag lr1e-03  # out=results/r2/d4_tune/henon_skrtrl-r16_s100_lr1e-03.json

The trailing `# out=<path>` annotation is how the queue knows whether a job is already
done (exists-skip).  It is authoritative -- no filename is ever re-derived from the
flags -- so a job without it is always run.  A line may start with the .py script
(the queue prepends -Python), with `python`, or with an explicit executable.

Usage
-----
  # GPU 0
  powershell -ExecutionPolicy Bypass -File run_r2_queue.ps1 -Worker 0 -NWorkers 2 -Dev 0 -Jobs jobs/d4_tune.txt
  # GPU 1
  powershell -ExecutionPolicy Bypass -File run_r2_queue.ps1 -Worker 1 -NWorkers 2 -Dev 1 -Jobs jobs/d4_tune.txt

Each job gets a timeout (the CPU SVD fallback looks like a hang) and one retry.
stdout+stderr of job <idx> land in <LogDir>/<idx>.log.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$Worker,
    [int]$NWorkers = 2,
    [int]$Dev = 0,
    [Parameter(Mandatory = $true)][string]$Jobs,
    [int]$TimeoutSec = 10800,
    [int]$Retries = 1,
    [string]$Python = "D:\Anaconda\envs\multilingual_lora\python.exe",
    [string]$LogDir = "logs\r2",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if ($Worker -lt 0 -or $Worker -ge $NWorkers) { throw "Worker must be in 0..$($NWorkers-1)" }
if (-not (Test-Path $Jobs)) { throw "job file not found: $Jobs" }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

# ---- read and index the job list (index over KEPT lines only, so it is stable) ----
$raw = Get-Content -Path $Jobs
$kept = New-Object System.Collections.ArrayList
foreach ($line in $raw) {
    $t = $line.Trim()
    if ($t.Length -eq 0) { continue }
    if ($t.StartsWith("#")) { continue }
    [void]$kept.Add($t)
}

$total = $kept.Count
$mine = @()
for ($i = 0; $i -lt $total; $i++) {
    if (($i % $NWorkers) -eq $Worker) { $mine += $i }
}

Write-Output "[queue] worker $Worker/$NWorkers  dev $Dev  jobs file $Jobs"
Write-Output "[queue] $total jobs total, $($mine.Count) assigned to this worker"
Write-Output "[queue] timeout ${TimeoutSec}s, retries $Retries, logs -> $LogDir"

$env:CUDA_VISIBLE_DEVICES = "$Dev"
$env:PYTHONUNBUFFERED = "1"

$nOk = 0; $nSkip = 0; $nFail = 0; $nTimeout = 0
$failed = @()
$t0 = Get-Date

foreach ($idx in $mine) {
    $line = $kept[$idx]

    # --- split off the "# out=<path>" annotation ---
    $outPath = $null
    $cmd = $line
    $m = [regex]::Match($line, '#\s*out\s*=\s*(\S+)\s*$')
    if ($m.Success) {
        $outPath = $m.Groups[1].Value
        $cmd = $line.Substring(0, $m.Index).Trim()
    }

    if ($outPath -and (Test-Path $outPath)) {
        Write-Output "[$idx] SKIP (exists) $outPath"
        $nSkip++
        continue
    }

    # --- resolve executable and argument string ---
    $first = ($cmd -split '\s+')[0]
    if ($first -match '\.py$') {
        $exe = $Python; $argline = $cmd
    }
    elseif ($first -match '^(python|python\.exe|py)$') {
        $exe = $Python; $argline = $cmd.Substring($first.Length).Trim()
    }
    else {
        $exe = $first; $argline = $cmd.Substring($first.Length).Trim()
    }

    $log = Join-Path $LogDir "$idx.log"
    $outLog = Join-Path $LogDir "$idx.out.part"
    $errLog = Join-Path $LogDir "$idx.err.part"

    if ($DryRun) {
        Write-Output "[$idx] DRYRUN $exe $argline"
        continue
    }

    $ok = $false
    $timedOut = $false
    $code = -1
    for ($attempt = 1; $attempt -le (1 + $Retries); $attempt++) {
        $stamp = (Get-Date).ToString("s")
        Add-Content -Path $log -Encoding utf8 -Value "==== [$idx] attempt $attempt dev $Dev $stamp ===="
        Add-Content -Path $log -Encoding utf8 -Value "==== cmd: $exe $argline"
        $timedOut = $false
        $proc = Start-Process -FilePath $exe -ArgumentList $argline -PassThru -NoNewWindow `
            -RedirectStandardOutput $outLog -RedirectStandardError $errLog
        # Touching .Handle caches the process handle in this session; without it
        # Start-Process -PassThru can leave .ExitCode unreadable ($null) after exit.
        try { $null = $proc.Handle } catch {}
        if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
            $timedOut = $true
            try { $proc.Kill() } catch {}
            $proc.WaitForExit()
        }
        $code = $null
        try { $code = $proc.ExitCode } catch {}
        # fold the redirect targets into the single per-job log
        foreach ($f in @($outLog, $errLog)) {
            if (Test-Path $f) {
                Get-Content -Path $f | Add-Content -Path $log -Encoding utf8
                Remove-Item -Path $f -Force
            }
        }
        $producedOut = (-not $outPath) -or (Test-Path $outPath)
        # $code can be $null if the handle was lost; the declared output file is then
        # the authority on success (the runner writes it only on a clean finish).
        $codeOk = ($code -eq 0) -or (($null -eq $code) -and $producedOut)
        if ((-not $timedOut) -and $codeOk -and $producedOut) { $ok = $true; break }
        $why = if ($timedOut) { "TIMEOUT" } elseif (-not $codeOk) { "exit $code" } else { "no output file" }
        Write-Output "[$idx] attempt $attempt failed ($why)"
        Add-Content -Path $log -Encoding utf8 -Value "==== [$idx] attempt $attempt failed ($why)"
    }

    if ($ok) {
        Write-Output "[$idx] OK   $cmd"
        $nOk++
    }
    else {
        if ($timedOut) { $nTimeout++ }
        $nFail++
        $failed += $idx
        Write-Output "[$idx] FAIL $cmd  (see $log)"
    }
}

$dt = (Get-Date) - $t0
Write-Output "--------------------------------------------------------------"
Write-Output "[queue] worker $Worker done in $([int]$dt.TotalMinutes) min"
Write-Output "[queue] assigned $($mine.Count)  ok $nOk  skipped $nSkip  failed $nFail  (of which timeouts $nTimeout)"
if ($failed.Count -gt 0) { Write-Output "[queue] failed job indices: $($failed -join ',')" }
if ($nFail -gt 0) { exit 1 }
