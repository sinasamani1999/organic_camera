# Rename phone clips: all 12 current clips were recorded by one person (p01).
# Numbering follows recording time. Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File tools\rename_clips.ps1
$raw = "dataset\phone_raw"
$proc = "dataset\phone"
$map = [ordered]@{
    "p02_office_idle_01" = "p01_office_idle_04"   # 13:54:07
    "p02_office_idle_02" = "p01_office_idle_05"   # 13:55:26
    "p03_office_idle_01" = "p01_office_idle_06"   # 13:56:45
    "p02_office_walk_01" = "p01_office_walk_02"   # 15:22:46
    "p03_office_walk_01" = "p01_office_walk_03"   # 15:24:16
    "p04_office_walk_01" = "p01_office_walk_04"   # 15:25:50
    "p05_office_walk_01" = "p01_office_walk_05"   # 15:27:23
    "p06_office_walk_01" = "p01_office_walk_06"   # 15:28:44
}
foreach ($old in $map.Keys) {
    $new = $map[$old]
    if (Test-Path (Join-Path $raw $new)) { Write-Host "skip $old : $new already exists"; continue }
    if (Test-Path (Join-Path $raw $old)) {
        Rename-Item (Join-Path $raw $old) $new
        Write-Host "$old -> $new"
    } else { Write-Host "skip $old : not found" }
    # processed output under the old name is stale; it is rebuilt by batch_preprocess
    if (Test-Path (Join-Path $proc $old)) { Remove-Item (Join-Path $proc $old) -Recurse -Force }
}
if (Test-Path (Join-Path $proc "test01")) { Remove-Item (Join-Path $proc "test01") -Recurse -Force }
if (Test-Path (Join-Path $proc "summary.csv")) { Remove-Item (Join-Path $proc "summary.csv") }
Write-Host "done. Next: python -m data_pipeline.batch_preprocess"
