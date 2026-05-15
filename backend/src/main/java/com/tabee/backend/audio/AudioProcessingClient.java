package com.tabee.backend.audio;

import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

@Component
public class AudioProcessingClient {
    private final RestTemplate restTemplate;
    private final String baseUrl;

    public AudioProcessingClient(RestTemplate restTemplate,
                                 @Value("${tabee.audio-processing.base-url:}") String baseUrl) {
        this.restTemplate = restTemplate;
        this.baseUrl = baseUrl == null ? "" : baseUrl.trim();
    }

    public String requestProcessing(AudioFile audioFile) {
        if (baseUrl.isBlank()) {
            return "Audio processing service is not configured yet";
        }

        Map<String, Object> payload = Map.of(
                "audioFileId", audioFile.getId(),
                "storedFilename", audioFile.getStoredFilename(),
                "originalFilename", audioFile.getOriginalFilename()
        );

        return restTemplate.postForObject(baseUrl + "/process", payload, String.class);
    }
}
