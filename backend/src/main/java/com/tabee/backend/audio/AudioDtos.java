package com.tabee.backend.audio;

import java.time.OffsetDateTime;

import io.swagger.v3.oas.annotations.media.Schema;
import org.springframework.web.multipart.MultipartFile;

public final class AudioDtos {
    private AudioDtos() {
    }

    public record AudioFileResponse(
            @Schema(example = "1")
            Long id,
            @Schema(example = "1")
            Long ownerUserId,
            @Schema(example = "test.wav")
            String originalFilename,
            @Schema(example = "2c2ef1ef-5b1d-41b0-ae72-test.wav")
            String storedFilename,
            @Schema(example = "audio/wav")
            String contentType,
            @Schema(example = "123456")
            Long fileSizeBytes,
            @Schema(example = "UPLOADED")
            String processingStatus,
            OffsetDateTime uploadedAt
    ) {
        public static AudioFileResponse from(AudioFile audioFile) {
            return new AudioFileResponse(
                    audioFile.getId(),
                    audioFile.getOwner().getId(),
                    audioFile.getOriginalFilename(),
                    audioFile.getStoredFilename(),
                    audioFile.getContentType(),
                    audioFile.getFileSizeBytes(),
                    audioFile.getProcessingStatus().name(),
                    audioFile.getUploadedAt()
            );
        }
    }

    public record AudioProcessingResponse(
            Long audioFileId,
            String status,
            String message,
            Long tabId
    ) {
    }

    public record AudioUploadAndProcessResponse(
            AudioFileResponse audioFile,
            String message,
            Long tabId
    ) {
    }

    public static class AudioUploadRequest {
        @Schema(type = "string", format = "binary", description = "Audio file to upload")
        private MultipartFile file;

        public MultipartFile getFile() {
            return file;
        }

        public void setFile(MultipartFile file) {
            this.file = file;
        }
    }
}
