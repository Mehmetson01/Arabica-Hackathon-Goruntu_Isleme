# Görev 1 - Kafe kişi sayımı + cinsiyet (kapı çizgisi configs ile ayarlı)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$line = (Get-Content configs\task1_line.txt -ErrorAction SilentlyContinue | Select-Object -First 1)
if (-not $line) { $line = "0.25,0.60,0.75,0.60" }
$extra = (Get-Content configs\task1_args.txt -ErrorAction SilentlyContinue | Select-Object -First 1)
$cmd = "& .\.venv\Scripts\python.exe -m src.task1_people.run --video kisi_sayimi.mp4 --line `"$line`" $extra $args"
Write-Host $cmd
Invoke-Expression $cmd
