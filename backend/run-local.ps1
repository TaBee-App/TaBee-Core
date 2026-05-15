$ErrorActionPreference = "Stop"

$env:JAVA_HOME = "C:\Program Files\Java\jdk-17"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"

$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
        }
    }
}

if (-not $env:DATABASE_URL) {
    $env:DATABASE_URL = Read-Host "Database JDBC URL"
}

if (-not $env:DATABASE_USERNAME) {
    $env:DATABASE_USERNAME = Read-Host "Database username"
}

if (-not $env:DATABASE_PASSWORD) {
    $env:DATABASE_PASSWORD = Read-Host "Database password"
}

if ($env:DATABASE_URL -like "*your-aiven-host*") {
    throw "DATABASE_URL still contains the placeholder 'your-aiven-host'. Edit backend/.env and paste your real Aiven JDBC URL."
}

if ($env:DATABASE_USERNAME -eq "your-database-username") {
    throw "DATABASE_USERNAME still contains a placeholder. Edit backend/.env and paste your real database username."
}

if ($env:DATABASE_PASSWORD -eq "your-aiven-password") {
    throw "DATABASE_PASSWORD still contains a placeholder. Edit backend/.env and paste your real database password."
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
