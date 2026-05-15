package com.tabee.backend.tab;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.audio.AudioFile;
import com.tabee.backend.tab.TabDtos.NoteEventRequest;
import com.tabee.backend.tab.TabDtos.TabRequest;
import com.tabee.backend.user.User;

@Service
public class TabProcessingService {
    private final TabService tabService;
    private final ObjectMapper objectMapper;
    private final Path coreRoot;
    private final String pythonCommand;

    public TabProcessingService(TabService tabService,
                                ObjectMapper objectMapper,
                                @Value("${tabee.core-root:..}") String coreRoot,
                                @Value("${tabee.python-command:python}") String pythonCommand) {
        this.tabService = tabService;
        this.objectMapper = objectMapper;
        this.coreRoot = Path.of(coreRoot).toAbsolutePath().normalize();
        this.pythonCommand = pythonCommand;
    }

    public Tab generateTabFromAudio(User owner, AudioFile audioFile, Path uploadDir) {
        Path audioPath = audioFile.resolveStoredPath(uploadDir).toAbsolutePath().normalize();
        if (!Files.exists(audioPath)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Stored audio file not found: " + audioPath);
        }

        Path outputDir;
        try {
            outputDir = Files.createTempDirectory("tabee-processing-");
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Could not create processing directory", e);
        }

        Path jsonOut = outputDir.resolve("tab.json");
        runPythonProcessor(audioPath, jsonOut);
        GeneratedTabResult result = readResult(jsonOut);

        TabRequest request = new TabRequest(
                audioFile.getId(),
                defaultTitle(audioFile),
                null,
                result.tuning(),
                result.estimatedTempo(),
                toNoteRequests(result.noteEvents())
        );
        return tabService.create(owner, request);
    }

    private void runPythonProcessor(Path audioPath, Path jsonOut) {
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

        try {
            Process process = processBuilder.start();
            boolean finished = process.waitFor(Duration.ofMinutes(3).toMillis(), java.util.concurrent.TimeUnit.MILLISECONDS);
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);

            if (!finished) {
                process.destroyForcibly();
                throw new ResponseStatusException(HttpStatus.GATEWAY_TIMEOUT, "Audio processing timed out");
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

    private GeneratedTabResult readResult(Path jsonOut) {
        try {
            return objectMapper.readValue(jsonOut.toFile(), GeneratedTabResult.class);
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Could not read generated tab JSON", e);
        }
    }

    private String defaultTitle(AudioFile audioFile) {
        String filename = audioFile.getOriginalFilename();
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

    private List<NoteEventRequest> toNoteRequests(List<GeneratedTabResult.GeneratedNoteEvent> events) {
        if (events == null) {
            return List.of();
        }

        return events.stream()
                .map(event -> new NoteEventRequest(
                        event.time(),
                        event.frequency(),
                        normalizeConfidence(event.confidence()),
                        event.noteName(),
                        event.midiNumber(),
                        event.fret(),
                        event.stringNumber()
                ))
                .toList();
    }

    private BigDecimal normalizeConfidence(BigDecimal confidence) {
        if (confidence == null) {
            return null;
        }
        if (confidence.compareTo(BigDecimal.ZERO) < 0) {
            return BigDecimal.ZERO;
        }
        if (confidence.compareTo(BigDecimal.ONE) > 0) {
            return BigDecimal.ONE;
        }
        return confidence;
    }
}
