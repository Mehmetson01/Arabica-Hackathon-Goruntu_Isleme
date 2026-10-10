# Görev 2 - Şehir girişi araç sayımı + plaka OCR
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$video = (Get-ChildItem -Filter "4K*.mp4" | Select-Object -First 1).Name
$line = (Get-Content configs\task2_line.txt -ErrorAction SilentlyContinue | Select-Object -First 1)
if (-not $line) { $line = "0.0,0.62,1.0,0.62" }
$extra = (Get-Content configs\task2_args.txt -ErrorAction SilentlyContinue | Select-Object -First 1)
$cmd = "& .\.venv\Scripts\python.exe -m src.task2_vehicles.run --video `"$video`" --line `"$line`" $extra $args"
Write-Host $cmd
Invoke-Expression $cmd
