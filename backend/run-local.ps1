$ErrorActionPreference = "Stop"

$env:JAVA_HOME = "C:\Program Files\Java\jdk-17"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"

$env:DATABASE_URL = "jdbc:postgresql://tabee-onurmelis1234-8692.j.aivencloud.com:11255/defaultdb?ssl=require"
$env:DATABASE_USERNAME = "avnadmin"

if (-not $env:DATABASE_PASSWORD) {
    $env:DATABASE_PASSWORD = Read-Host "Aiven database password"
}

if (-not $env:SPRING_JPA_HIBERNATE_DDL_AUTO) {
    $env:SPRING_JPA_HIBERNATE_DDL_AUTO = "none"
}

$existingPorts = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" -and $_.OwningProcess -gt 0 }

foreach ($existingPort in $existingPorts) {
    Write-Host "Stopping process $($existingPort.OwningProcess) on port 8080..."
    Stop-Process -Id $existingPort.OwningProcess -Force
}

if ($existingPorts) {
    Start-Sleep -Seconds 2
}

mvn package -DskipTests
java -jar target\tabee-backend-0.0.1-SNAPSHOT.jar
