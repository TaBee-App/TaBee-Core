$env:JAVA_HOME = "C:\Program Files\Java\jdk-17"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
$env:DATABASE_URL = "jdbc:postgresql://your-aiven-host:11255/defaultdb?ssl=require"
$env:DATABASE_USERNAME = "your-database-username"
$env:DATABASE_PASSWORD = "your-aiven-password"
$env:SPRING_JPA_HIBERNATE_DDL_AUTO = "none"

mvn clean package -DskipTests
java -jar target\tabee-backend-0.0.1-SNAPSHOT.jar
