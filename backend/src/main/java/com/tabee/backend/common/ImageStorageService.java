package com.tabee.backend.common;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

@Service
public class ImageStorageService {
    private static final Set<String> ALLOWED_CONTENT_TYPES = Set.of("image/png", "image/jpeg");

    private final Path mediaDir;

    public ImageStorageService(@Value("${tabee.media-dir:media}") String mediaDir) {
        this.mediaDir = Path.of(mediaDir).toAbsolutePath().normalize();
    }

    public String store(MultipartFile file, String prefix) {
        validate(file);
        String extension = extension(file);
        String filename = prefix + "-" + UUID.randomUUID() + extension;
        try {
            Files.createDirectories(mediaDir);
            file.transferTo(mediaDir.resolve(filename));
            return filename;
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Could not store image", e);
        }
    }

    public Path resolve(String filename) {
        Path resolved = mediaDir.resolve(filename).normalize();
        if (!resolved.startsWith(mediaDir) || !Files.exists(resolved)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Image not found");
        }
        return resolved;
    }

    public void deleteQuietly(String filename) {
        if (filename == null || filename.isBlank()) {
            return;
        }
        try {
            Files.deleteIfExists(mediaDir.resolve(filename).normalize());
        } catch (IOException ignored) {
        }
    }

    public String url(String filename) {
        return filename == null || filename.isBlank() ? null : "/api/media/" + filename;
    }

    private void validate(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Image file is required");
        }
        String contentType = file.getContentType() == null ? "" : file.getContentType().toLowerCase(Locale.ROOT);
        String name = file.getOriginalFilename() == null ? "" : file.getOriginalFilename().toLowerCase(Locale.ROOT);
        boolean validType = ALLOWED_CONTENT_TYPES.contains(contentType);
        boolean validExtension = name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg");
        if (!validType || !validExtension) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Only PNG and JPEG images are supported");
        }
    }

    private String extension(MultipartFile file) {
        String name = file.getOriginalFilename() == null ? "" : file.getOriginalFilename().toLowerCase(Locale.ROOT);
        if (name.endsWith(".png")) {
            return ".png";
        }
        return ".jpg";
    }
}
