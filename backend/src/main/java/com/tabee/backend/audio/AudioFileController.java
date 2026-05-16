package com.tabee.backend.audio;

import java.util.List;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import com.tabee.backend.audio.AudioDtos.AudioFileResponse;
import com.tabee.backend.audio.AudioDtos.AudioUploadRequest;
import com.tabee.backend.audio.AudioDtos.AudioUploadAndProcessResponse;
import com.tabee.backend.security.CurrentUser;
import com.tabee.backend.user.User;

@RestController
@RequestMapping("/api/audio-files")
public class AudioFileController {
    private final AudioFileService audioFileService;
    private final CurrentUser currentUser;

    public AudioFileController(AudioFileService audioFileService, CurrentUser currentUser) {
        this.audioFileService = audioFileService;
        this.currentUser = currentUser;
    }

    @GetMapping
    public List<AudioFileResponse> findMine(@AuthenticationPrincipal User currentUser) {
        return audioFileService.findByOwner(this.currentUser.require(currentUser)).stream()
                .map(AudioFileResponse::from)
                .toList();
    }

    @GetMapping("/{id}")
    public AudioFileResponse findById(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        AudioFileResponse response = AudioFileResponse.from(audioFileService.findById(id));
        if (!response.ownerUserId().equals(this.currentUser.require(currentUser).getId())) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.FORBIDDEN,
                    "Audio file does not belong to current user"
            );
        }
        return response;
    }

    @PostMapping(value = "/upload-and-process", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @ResponseStatus(HttpStatus.CREATED)
    @Operation(summary = "Upload an audio file and immediately generate a tab")
    @RequestBody(content = @Content(
            mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
            schema = @Schema(implementation = AudioUploadRequest.class)
    ))
    public AudioUploadAndProcessResponse uploadAndProcess(
            @AuthenticationPrincipal User currentUser,
            @ModelAttribute AudioUploadRequest request) {
        return audioFileService.uploadAndProcess(this.currentUser.require(currentUser), request.getFile());
    }

    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        audioFileService.delete(this.currentUser.require(currentUser), id);
    }
}
