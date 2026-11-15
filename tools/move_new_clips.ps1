# Move the new idle clips (dataset\temp, 2026-10-30) to dataset\phone_raw as p02..p06.
# Run from the repo root: powershell -ExecutionPolicy Bypass -File tools\move_new_clips.ps1
$raw = "dataset\phone_raw"
$proc = "dataset\phone"
$temp = "dataset\temp"

function Move-Clip($src, $dst) {
    if (Test-Path $dst) { Write-Host "skip: $dst already exists"; return }
    if (-not (Test-Path $src)) { Write-Host "skip: $src not found"; return }
    Move-Item $src $dst
    Write-Host "$src -> $dst"
}

# ---- step 2: new idle clips from dataset\temp -> p02..p06
$new = [ordered]@{
    "20261030_180347" = "p02_office_idle_01"
    "20261030_180458" = "p02_office_idle_02"
    "20261030_180625" = "p03_office_idle_01"
    "20261030_180736" = "p03_office_idle_02"
    "20261030_180900" = "p04_office_idle_01"
    "20261030_181010" = "p04_office_idle_02"
    "20261030_181137" = "p05_office_idle_01"
    "20261030_181246" = "p05_office_idle_02"
    "20261030_181414" = "p06_office_idle_01"
    "20261030_181523" = "p06_office_idle_02"
}
foreach ($k in $new.Keys) { Move-Clip (Join-Path $temp $k) (Join-Path $raw $new[$k]) }

foreach ($x in @("test01", "summary.csv")) {
    if (Test-Path (Join-Path $proc $x)) { Remove-Item (Join-Path $proc $x) -Recurse -Force }
}
if ((Test-Path $temp) -and -not (Get-ChildItem $temp)) { Remove-Item $temp; Write-Host "removed empty $temp" }
Write-Host ""
Get-ChildItem $raw -Directory | Select-Object -ExpandProperty Name
Write-Host "done. Next: python -m data_pipeline.batch_preprocess"
