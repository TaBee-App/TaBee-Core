package com.tabee.backend.tab;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import com.fasterxml.jackson.databind.JsonNode;

@Service
public class TabAudioStorage {
    static final String AUDIO_FILE_KEY = "sourceAudioFile";

    private final Path uploadDir;

    public TabAudioStorage(@Value("${tabee.upload-dir:uploads}") String uploadDir) {
        this.uploadDir = Path.of(uploadDir).toAbsolutePath().normalize();
    }

    public String store(Tab tab, Path temporaryAudioPath, String originalFilename) {
        try {
            Files.createDirectories(uploadDir);
            String storedFilename = tab.getOwner().getId() + "-" + tab.getId() + "-" + safeFilename(originalFilename);
            Path target = uploadDir.resolve(storedFilename).normalize();

            if (!target.startsWith(uploadDir)) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid audio filename");
            }

            Files.copy(temporaryAudioPath, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            return storedFilename;
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Could not store uploaded audio", e);
        }
    }

    public Resource load(Tab tab) {
        JsonNode jsonData = tab.getTabData().getJsonData();
        JsonNode storedFile = jsonData == null ? null : jsonData.get(AUDIO_FILE_KEY);
        if (storedFile == null || storedFile.asText().isBlank()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Audio file not found for tab");
        }

        try {
            Path audioPath = uploadDir.resolve(storedFile.asText()).normalize();
            if (!audioPath.startsWith(uploadDir) || !Files.isRegularFile(audioPath)) {
                throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Audio file not found for tab");
            }
            return new UrlResource(audioPath.toUri());
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Could not read uploaded audio", e);
        }
    }

    private String safeFilename(String filename) {
        if (filename == null || filename.isBlank()) {
            return "upload.wav";
        }
        return Path.of(filename).getFileName().toString().replaceAll("[^A-Za-z0-9._-]", "_");
    }
}
