package com.tabee.backend.audio;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.audio.AudioDtos.AudioProcessingResponse;
import com.tabee.backend.user.User;
import com.tabee.backend.user.UserService;

@Service
public class AudioFileService {
    private final AudioFileRepository audioFileRepository;
    private final UserService userService;
    private final AudioProcessingClient audioProcessingClient;
    private final Path uploadDir;

    public AudioFileService(AudioFileRepository audioFileRepository,
                            UserService userService,
                            AudioProcessingClient audioProcessingClient,
                            @Value("${tabee.upload-dir:uploads}") String uploadDir) {
        this.audioFileRepository = audioFileRepository;
        this.userService = userService;
        this.audioProcessingClient = audioProcessingClient;
        this.uploadDir = Path.of(uploadDir);
    }

    @Transactional(readOnly = true)
    public List<AudioFile> findByOwner(User owner) {
        return audioFileRepository.findByOwnerIdOrderByUploadedAtDesc(owner.getId());
    }

    @Transactional(readOnly = true)
    public AudioFile findById(Long id) {
        return audioFileRepository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Audio file not found"));
    }

    @Transactional
    public AudioFile upload(User owner, MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Audio file is required");
        }

        String originalFilename = file.getOriginalFilename() == null ? "audio" : file.getOriginalFilename();
        String storedFilename = UUID.randomUUID() + "-" + originalFilename.replaceAll("[^a-zA-Z0-9._-]", "_");

        try {
            Files.createDirectories(uploadDir);
            file.transferTo(uploadDir.resolve(storedFilename));
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Could not store audio file", e);
        }

        AudioFile audioFile = new AudioFile();
        audioFile.setOwner(owner);
        audioFile.setOriginalFilename(originalFilename);
        audioFile.setStoredFilename(storedFilename);
        audioFile.setContentType(file.getContentType());
        audioFile.setFileSizeBytes(file.getSize());
        return audioFileRepository.save(audioFile);
    }

    @Transactional
    public AudioProcessingResponse requestProcessing(Long audioFileId) {
        AudioFile audioFile = findById(audioFileId);
        audioFile.setProcessingStatus(ProcessingStatus.PROCESSING);
        audioFileRepository.save(audioFile);

        try {
            String message = audioProcessingClient.requestProcessing(audioFile);
            return new AudioProcessingResponse(audioFile.getId(), audioFile.getProcessingStatus().name(), message);
        } catch (RuntimeException e) {
            audioFile.setProcessingStatus(ProcessingStatus.FAILED);
            audioFileRepository.save(audioFile);
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "Audio processing service failed", e);
        }
    }

    @Transactional
    public void delete(Long id) {
        audioFileRepository.delete(findById(id));
    }
}
