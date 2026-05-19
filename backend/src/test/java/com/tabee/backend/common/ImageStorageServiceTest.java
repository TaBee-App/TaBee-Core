package com.tabee.backend.common;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.http.HttpStatus;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.server.ResponseStatusException;

class ImageStorageServiceTest {

    @TempDir
    Path mediaDir;

    @Test
    void storesPngAndReturnsMediaUrl() {
        ImageStorageService service = new ImageStorageService(mediaDir.toString());
        MockMultipartFile image = new MockMultipartFile(
                "file",
                "avatar.png",
                "image/png",
                new byte[] {1, 2, 3}
        );

        String filename = service.store(image, "user-1");

        assertTrue(filename.startsWith("user-1-"));
        assertTrue(filename.endsWith(".png"));
        assertTrue(Files.exists(mediaDir.resolve(filename)));
        assertEquals("/api/media/" + filename, service.url(filename));
    }

    @Test
    void rejectsNonImageContentType() {
        ImageStorageService service = new ImageStorageService(mediaDir.toString());
        MockMultipartFile textFile = new MockMultipartFile(
                "file",
                "avatar.png",
                "text/plain",
                new byte[] {1}
        );

        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                service.store(textFile, "user-1"));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Only PNG and JPEG images are supported", exception.getReason());
    }

    @Test
    void rejectsPathTraversalWhenResolving() {
        ImageStorageService service = new ImageStorageService(mediaDir.toString());

        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                service.resolve("../secret.png"));

        assertEquals(HttpStatus.NOT_FOUND, exception.getStatusCode());
        assertEquals("Image not found", exception.getReason());
    }

    @Test
    void storesJpegWithNormalizedJpgExtension() {
        ImageStorageService service = new ImageStorageService(mediaDir.toString());
        MockMultipartFile image = new MockMultipartFile(
                "file",
                "cover.jpeg",
                "image/jpeg",
                new byte[] {4, 5, 6}
        );

        String filename = service.store(image, "playlist-9");

        assertTrue(filename.startsWith("playlist-9-"));
        assertTrue(filename.endsWith(".jpg"));
        assertTrue(Files.exists(mediaDir.resolve(filename)));
    }

    @Test
    void rejectsImageWhenExtensionDoesNotMatchContentType() {
        ImageStorageService service = new ImageStorageService(mediaDir.toString());
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "avatar.gif",
                "image/png",
                new byte[] {1, 2, 3}
        );

        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                service.store(file, "user-1"));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Only PNG and JPEG images are supported", exception.getReason());
    }

    @Test
    void returnsNullUrlForBlankFilename() {
        ImageStorageService service = new ImageStorageService(mediaDir.toString());

        assertNull(service.url(null));
        assertNull(service.url("   "));
    }

    @Test
    void deleteQuietlyIgnoresMissingOrBlankFilenames() {
        ImageStorageService service = new ImageStorageService(mediaDir.toString());

        assertDoesNotThrow(() -> service.deleteQuietly(null));
        assertDoesNotThrow(() -> service.deleteQuietly(""));
        assertDoesNotThrow(() -> service.deleteQuietly("missing.png"));
    }
}
