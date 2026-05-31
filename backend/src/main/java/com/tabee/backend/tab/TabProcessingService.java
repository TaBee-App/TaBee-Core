package com.tabee.backend.tab;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Locale;
import java.util.Set;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.tab.TabDtos.TabRequest;
import com.tabee.backend.user.User;

@Service
public class TabProcessingService {
    private static final Set<String> ALLOWED_EXTENSIONS = Set.of(".wav", ".mp4", ".mp3");
    private static final Set<String> ALLOWED_TUNINGS = Set.of("EADG", "BEADG", "CGCF", "EBABDBGB");
    private static final Set<String> ALLOWED_CONTENT_TYPES = Set.of(
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/vnd.wave",
            "audio/mpeg",
            "audio/mp3",
            "video/mp4",
            "audio/mp4",
            "application/octet-stream"
    );

    private final TabService tabService;
    private final TabAudioStorage tabAudioStorage;
    private final ObjectMapper objectMapper;
    private final Path coreRoot;
    private final String pythonCommand;
    private final Duration processingTimeout;

    public TabProcessingService(TabService tabService,
                                TabAudioStorage tabAudioStorage,
                                ObjectMapper objectMapper,
                                @Value("${tabee.core-root:..}") String coreRoot,
                                @Value("${tabee.python-command:python}") String pythonCommand,
                                @Value("${tabee.processing-timeout-minutes:10}") long processingTimeoutMinutes) {
        this.tabService = tabService;
        this.tabAudioStorage = tabAudioStorage;
        this.objectMapper = objectMapper;
        this.coreRoot = Path.of(coreRoot).toAbsolutePath().normalize();
        this.pythonCommand = pythonCommand;
        this.processingTimeout = Duration.ofMinutes(processingTimeoutMinutes);
    }

    public Tab generateTabFromUpload(User owner, MultipartFile audioFile, String tuning) {
        if (audioFile == null || audioFile.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Audio file is required");
        }

        validateSupportedAudioFile(audioFile);
        String normalizedTuning = normalizeTuning(tuning);

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
        runPythonProcessor(audioPath, jsonOut, processorLog, normalizedTuning);
            GeneratedTabResult result = readResult(jsonOut);
            JsonNode tabJson = objectMapper.valueToTree(result);

            TabRequest request = new TabRequest(
                    defaultTitle(originalFilename),
                    null,
                    result.tuning(),
                    result.estimatedTempo(),
                    tabJson
            );
            Tab tab = tabService.create(owner, request);
            String storedAudioFile = tabAudioStorage.store(tab, audioPath, originalFilename);
            ObjectNode jsonWithAudio = tabJson.deepCopy();
            jsonWithAudio.put(TabAudioStorage.AUDIO_FILE_KEY, storedAudioFile);
            return tabService.update(owner, tab.getId(), new TabDtos.TabUpdateRequest(
                    null,
                    null,
                    null,
                    null,
                    jsonWithAudio
            ));
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Could not store temporary audio file", e);
        } finally {
            deleteQuietly(outputDir);
        }
    }

    private void runPythonProcessor(Path audioPath, Path jsonOut, Path processorLog, String tuning) {
        ProcessBuilder processBuilder = new ProcessBuilder(
                pythonCommand,
                "cli/audio_to_tab.py",
                audioPath.toString(),
                "--json-out",
                jsonOut.toString(),
                "--ascii-out",
                jsonOut.resolveSibling("tab.txt").toString(),
                "--tuning",
                tuning
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

    private void validateSupportedAudioFile(MultipartFile audioFile) {
        String filename = audioFile.getOriginalFilename() == null ? "" : audioFile.getOriginalFilename();
        String lowercaseFilename = filename.toLowerCase(Locale.ROOT);
        boolean supportedExtension = ALLOWED_EXTENSIONS.stream().anyMatch(lowercaseFilename::endsWith);

        String contentType = audioFile.getContentType();
        boolean supportedContentType = contentType == null || contentType.isBlank()
                || ALLOWED_CONTENT_TYPES.contains(contentType.toLowerCase(Locale.ROOT));

        if (!supportedExtension || !supportedContentType) {
            throw new ResponseStatusException(
                    HttpStatus.BAD_REQUEST,
                    "Unsupported file format. Please upload a .wav, .mp3, or .mp4 file."
            );
        }
    }

    private String normalizeTuning(String tuning) {
        String normalized = tuning == null || tuning.isBlank()
                ? "EADG"
                : tuning.trim().toUpperCase(Locale.ROOT);

        if (!ALLOWED_TUNINGS.contains(normalized)) {
            throw new ResponseStatusException(
                    HttpStatus.BAD_REQUEST,
                    "Unsupported tuning. Please use EADG, BEADG, CGCF, or EBABDBGB."
            );
        }

        return normalized;
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
