package com.tabee.backend.audio;

import io.swagger.v3.oas.annotations.media.Schema;
import org.springframework.web.multipart.MultipartFile;

import com.tabee.backend.tab.TabDtos.TabResponse;

public final class AudioDtos {
    private AudioDtos() {
    }

    public record AudioUploadAndProcessResponse(
            String message,
            Long tabId,
            TabResponse tab
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
