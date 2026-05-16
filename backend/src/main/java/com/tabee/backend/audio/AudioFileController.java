package com.tabee.backend.audio;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import com.tabee.backend.audio.AudioDtos.AudioUploadRequest;
import com.tabee.backend.audio.AudioDtos.AudioUploadAndProcessResponse;
import com.tabee.backend.security.CurrentUser;
import com.tabee.backend.tab.Tab;
import com.tabee.backend.tab.TabDtos.TabResponse;
import com.tabee.backend.tab.TabProcessingService;
import com.tabee.backend.user.User;

@RestController
@RequestMapping("/api/audio-files")
public class AudioFileController {
    private final TabProcessingService tabProcessingService;
    private final CurrentUser currentUser;

    public AudioFileController(TabProcessingService tabProcessingService, CurrentUser currentUser) {
        this.tabProcessingService = tabProcessingService;
        this.currentUser = currentUser;
    }

    @PostMapping(value = "/upload-and-process", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @ResponseStatus(HttpStatus.CREATED)
    @Operation(summary = "Upload an audio file, process it immediately, and store the generated tab JSON")
    @RequestBody(content = @Content(
            mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
            schema = @Schema(implementation = AudioUploadRequest.class)
    ))
    public AudioUploadAndProcessResponse uploadAndProcess(
            @AuthenticationPrincipal User currentUser,
            @ModelAttribute AudioUploadRequest request) {
        Tab tab = tabProcessingService.generateTabFromUpload(this.currentUser.require(currentUser), request.getFile());
        return new AudioUploadAndProcessResponse(
                "Audio processed successfully. The uploaded file was not stored in the database.",
                tab.getId(),
                TabResponse.from(tab)
        );
    }
}
