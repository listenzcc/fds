# file: runme.ps1

# Start time
$date = Get-Date
Write-Host "Simulation started at: $date"

cd sample

../fds_local.bat sample.fds

$t1 = (Get-Date) - $date

python ../fds2txt.py

$t2 = (Get-Date) - $date

python ../txt2gif.py

$t3 = (Get-Date) - $date

cd ..

# End time, with time consuming
$cost = (Get-Date) - $date
Write-Host "Simulation ended at: $(Get-Date)"
Write-Host "Total time consuming: $cost ($t1, $t2, $t3)"
