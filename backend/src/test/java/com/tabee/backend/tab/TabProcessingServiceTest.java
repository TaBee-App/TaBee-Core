package com.tabee.backend.tab;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.server.ResponseStatusException;

class TabProcessingServiceTest {

    @Test
    void rejectsUnsupportedAudioExtensionBeforeProcessing() {
        TabProcessingService service = new TabProcessingService(
                null,
                null,
                new ObjectMapper(),
                ".",
                "python",
                1
        );
        MockMultipartFile textFile = new MockMultipartFile(
                "file",
                "notes.txt",
                "text/plain",
                "not audio".getBytes()
        );

        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                service.generateTabFromUpload(null, textFile));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Unsupported file format. Please upload a .wav, .mp3, or .mp4 file.", exception.getReason());
    }

    @Test
    void rejectsMissingAudioFile() {
        TabProcessingService service = new TabProcessingService(
                null,
                null,
                new ObjectMapper(),
                ".",
                "python",
                1
        );

        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                service.generateTabFromUpload(null, null));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Audio file is required", exception.getReason());
    }

    @Test
    void rejectsSupportedExtensionWithUnsupportedContentType() {
        TabProcessingService service = new TabProcessingService(
                null,
                null,
                new ObjectMapper(),
                ".",
                "python",
                1
        );
        MockMultipartFile fakeMp3 = new MockMultipartFile(
                "file",
                "song.mp3",
                "text/plain",
                "fake audio".getBytes()
        );

        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                service.generateTabFromUpload(null, fakeMp3));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Unsupported file format. Please upload a .wav, .mp3, or .mp4 file.", exception.getReason());
    }

    @Test
    void rejectsUnsupportedExtensionWithAudioContentType() {
        TabProcessingService service = new TabProcessingService(
                null,
                null,
                new ObjectMapper(),
                ".",
                "python",
                1
        );
        MockMultipartFile renamedText = new MockMultipartFile(
                "file",
                "song.txt",
                "audio/mpeg",
                "fake audio".getBytes()
        );

        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                service.generateTabFromUpload(null, renamedText));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Unsupported file format. Please upload a .wav, .mp3, or .mp4 file.", exception.getReason());
    }
}
