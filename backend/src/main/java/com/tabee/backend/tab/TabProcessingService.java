package com.tabee.backend.tab;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.tab.TabDtos.TabRequest;
import com.tabee.backend.user.User;

@Service
public class TabProcessingService {
    private final TabService tabService;
    private final ObjectMapper objectMapper;
    private final Path coreRoot;
    private final String pythonCommand;
    private final Duration processingTimeout;

    public TabProcessingService(TabService tabService,
                                ObjectMapper objectMapper,
                                @Value("${tabee.core-root:..}") String coreRoot,
                                @Value("${tabee.python-command:python}") String pythonCommand,
                                @Value("${tabee.processing-timeout-minutes:10}") long processingTimeoutMinutes) {
        this.tabService = tabService;
        this.objectMapper = objectMapper;
        this.coreRoot = Path.of(coreRoot).toAbsolutePath().normalize();
        this.pythonCommand = pythonCommand;
        this.processingTimeout = Duration.ofMinutes(processingTimeoutMinutes);
    }

    public Tab generateTabFromUpload(User owner, MultipartFile audioFile) {
        if (audioFile == null || audioFile.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Audio file is required");
        }

        Path outputDir;
        try {
            outputDir = Files.createTempDirectory("tabee-processing-");
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Could not create processing directory", e);
        }

        String originalFilename = safeFilename(audioFile.getOriginalFilename());
        Path audioPath = outputDir.resolve(originalFilename);
        Path jsonOut = outputDir.resolve("tab.json");

        try {
            audioFile.transferTo(audioPath);
        Path processorLog = jsonOut.resolveSibling("processor.log");
        runPythonProcessor(audioPath, jsonOut, processorLog);
            GeneratedTabResult result = readResult(jsonOut);
            JsonNode tabJson = objectMapper.valueToTree(result);

            TabRequest request = new TabRequest(
                    defaultTitle(originalFilename),
                    null,
                    result.tuning(),
                    result.estimatedTempo(),
                    tabJson
            );
            return tabService.create(owner, request);
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Could not store temporary audio file", e);
        } finally {
            deleteQuietly(outputDir);
        }
    }

    private void runPythonProcessor(Path audioPath, Path jsonOut, Path processorLog) {
        ProcessBuilder processBuilder = new ProcessBuilder(
                pythonCommand,
                "cli/audio_to_tab.py",
                audioPath.toString(),
                "--json-out",
                jsonOut.toString(),
                "--ascii-out",
                jsonOut.resolveSibling("tab.txt").toString()
        );
        processBuilder.directory(coreRoot.toFile());
        processBuilder.redirectErrorStream(true);
        processBuilder.redirectOutput(processorLog.toFile());

        try {
            Process process = processBuilder.start();
            boolean finished = process.waitFor(processingTimeout.toMillis(), java.util.concurrent.TimeUnit.MILLISECONDS);
            String output = readProcessorLog(processorLog);

            if (!finished) {
                process.destroyForcibly();
                throw new ResponseStatusException(
                        HttpStatus.GATEWAY_TIMEOUT,
                        "Audio processing timed out after " + processingTimeout.toMinutes() + " minutes. " + output
                );
            }
            if (process.exitValue() != 0) {
                throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Audio processing failed: " + output);
            }
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Could not start Python audio processor", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Audio processing was interrupted", e);
        }
    }

    private String readProcessorLog(Path processorLog) {
        try {
            if (!Files.exists(processorLog)) {
                return "";
            }
            String output = Files.readString(processorLog, StandardCharsets.UTF_8).trim();
            if (output.length() <= 1500) {
                return output;
            }
            return output.substring(output.length() - 1500);
        } catch (IOException e) {
            return "";
        }
    }

    private GeneratedTabResult readResult(Path jsonOut) {
        try {
            return objectMapper.readValue(jsonOut.toFile(), GeneratedTabResult.class);
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Could not read generated tab JSON", e);
        }
    }

    private String defaultTitle(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

    private String safeFilename(String filename) {
        if (filename == null || filename.isBlank()) {
            return "upload.wav";
        }
        return Path.of(filename).getFileName().toString().replaceAll("[^A-Za-z0-9._-]", "_");
    }

    private void deleteQuietly(Path directory) {
        try (var stream = Files.walk(directory)) {
            stream.sorted((left, right) -> right.compareTo(left))
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException ignored) {
                        }
                    });
        } catch (IOException ignored) {
        }
    }
}
