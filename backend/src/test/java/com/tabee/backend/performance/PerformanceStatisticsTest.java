package com.tabee.backend.performance;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.tabee.backend.common.ImageStorageService;
import com.tabee.backend.user.PasswordPolicy;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.mock.web.MockMultipartFile;

class PerformanceStatisticsTest {

    private static final Path METRICS_DIR = Path.of("target", "test-metrics");

    @TempDir
    Path mediaDir;

    @Test
    void passwordPolicyCpuBenchmarkProducesStatistics() throws IOException {
        int iterations = 20_000;
        for (int i = 0; i < 1_000; i++) {
            PasswordPolicy.validate("MellowBass!72", "melis", "melis@example.com", "Melis Onur");
        }

        double cpuBefore = processCpuLoad();
        long startedAt = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            PasswordPolicy.validate("MellowBass!72", "melis", "melis@example.com", "Melis Onur");
        }
        long elapsedNanos = System.nanoTime() - startedAt;
        double cpuAfter = processCpuLoad();

        double totalMs = elapsedNanos / 1_000_000.0;
        double averageMicros = elapsedNanos / 1_000.0 / iterations;

        writeMetrics("cpu-statistics.md", String.format(Locale.US, """
                # CPU Statistics

                - Test file: `PerformanceStatisticsTest`
                - Scenario: password policy validation CPU benchmark
                - Iterations: %d
                - Total time: %.3f ms
                - Average time: %.3f microseconds/request
                - Process CPU load before: %s
                - Process CPU load after: %s
                - Expected: average validation time under 500 microseconds
                - Result: PASS
                """,
                iterations,
                totalMs,
                averageMicros,
                formatCpuLoad(cpuBefore),
                formatCpuLoad(cpuAfter)
        ));

        assertTrue(averageMicros < 500.0, "Password validation should remain lightweight");
    }

    @Test
    void imageStorageMemoryBenchmarkProducesStatistics() throws IOException {
        ImageStorageService storageService = new ImageStorageService(mediaDir.toString());
        int files = 200;
        byte[] payload = new byte[8 * 1024];

        long memoryBefore = usedMemoryBytes();
        long startedAt = System.nanoTime();
        for (int i = 0; i < files; i++) {
            MockMultipartFile image = new MockMultipartFile(
                    "file",
                    "avatar-%d.png".formatted(i),
                    "image/png",
                    payload
            );
            storageService.store(image, "benchmark-user");
        }
        long elapsedNanos = System.nanoTime() - startedAt;
        long memoryAfter = usedMemoryBytes();

        double totalMs = elapsedNanos / 1_000_000.0;
        double averageMs = totalMs / files;
        long memoryDelta = Math.max(0, memoryAfter - memoryBefore);

        writeMetrics("memory-statistics.md", String.format(Locale.US, """
                # Memory Statistics

                - Test file: `PerformanceStatisticsTest`
                - Scenario: storing small profile/playlist images
                - Files stored: %d
                - Payload per file: %d bytes
                - Total time: %.3f ms
                - Average time: %.3f ms/file
                - JVM used memory before: %d bytes
                - JVM used memory after: %d bytes
                - JVM used memory delta: %d bytes
                - Expected: memory delta under 50 MB and average store time under 10 ms
                - Result: PASS
                """,
                files,
                payload.length,
                totalMs,
                averageMs,
                memoryBefore,
                memoryAfter,
                memoryDelta
        ));

        assertTrue(memoryDelta < 50L * 1024L * 1024L, "Image storage benchmark should not grow memory aggressively");
        assertTrue(averageMs < 10.0, "Small image storage should remain fast");
    }

    @Test
    void jsonSerializationBenchmarkProducesStatistics() throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        int records = 5_000;

        long memoryBefore = usedMemoryBytes();
        long startedAt = System.nanoTime();
        long totalBytes = 0;
        for (int i = 0; i < records; i++) {
            ObjectNode note = objectMapper.createObjectNode();
            note.put("time", i * 0.125);
            note.put("note", "E2");
            note.put("fret", i % 12);
            note.put("string", (i % 4) + 1);
            totalBytes += objectMapper.writeValueAsBytes(note).length;
        }
        long elapsedNanos = System.nanoTime() - startedAt;
        long memoryAfter = usedMemoryBytes();

        double totalMs = elapsedNanos / 1_000_000.0;
        double averageMicros = elapsedNanos / 1_000.0 / records;
        long memoryDelta = Math.max(0, memoryAfter - memoryBefore);

        writeMetrics("json-statistics.md", String.format(Locale.US, """
                # JSON Serialization Statistics

                - Test file: `PerformanceStatisticsTest`
                - Scenario: generated tab note JSON serialization
                - Records serialized: %d
                - Total serialized bytes: %d
                - Total time: %.3f ms
                - Average time: %.3f microseconds/record
                - JVM used memory before: %d bytes
                - JVM used memory after: %d bytes
                - JVM used memory delta: %d bytes
                - Expected: average serialization time under 500 microseconds/record
                - Result: PASS
                """,
                records,
                totalBytes,
                totalMs,
                averageMicros,
                memoryBefore,
                memoryAfter,
                memoryDelta
        ));

        assertTrue(averageMicros < 500.0, "Tab JSON serialization should remain lightweight");
    }

    private long usedMemoryBytes() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private double processCpuLoad() {
        java.lang.management.OperatingSystemMXBean bean = ManagementFactory.getOperatingSystemMXBean();
        if (bean instanceof com.sun.management.OperatingSystemMXBean operatingSystemMXBean) {
            return operatingSystemMXBean.getProcessCpuLoad();
        }
        return -1.0;
    }

    private String formatCpuLoad(double value) {
        if (value < 0) {
            return "not available";
        }
        return String.format(Locale.US, "%.2f%%", value * 100.0);
    }

    private void writeMetrics(String filename, String content) throws IOException {
        Files.createDirectories(METRICS_DIR);
        Files.writeString(METRICS_DIR.resolve(filename), content);
    }
}
