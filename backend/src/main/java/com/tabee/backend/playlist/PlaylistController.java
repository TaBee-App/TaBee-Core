package com.tabee.backend.playlist;

import java.util.List;

import jakarta.validation.Valid;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.tabee.backend.playlist.PlaylistDtos.AddTabRequest;
import com.tabee.backend.playlist.PlaylistDtos.PlaylistRequest;
import com.tabee.backend.playlist.PlaylistDtos.PlaylistResponse;
import com.tabee.backend.security.CurrentUser;
import com.tabee.backend.user.User;

@RestController
@RequestMapping("/api/playlists")
public class PlaylistController {
    private final PlaylistService playlistService;
    private final CurrentUser currentUser;

    public PlaylistController(PlaylistService playlistService, CurrentUser currentUser) {
        this.playlistService = playlistService;
        this.currentUser = currentUser;
    }

    private PlaylistResponse response(UserPlaylist playlist, Long currentUserId, boolean savedByCurrentUser) {
        return PlaylistResponse.from(
                playlist,
                currentUserId,
                savedByCurrentUser,
                playlistService.countSaves(playlist.getId()),
                playlistService.ownerProfileImageUrl(playlist),
                playlistService.coverImageUrl(playlist)
        );
    }

    @GetMapping
    public List<PlaylistResponse> findMine(@AuthenticationPrincipal User user) {
        User current = currentUser.require(user);
        return playlistService.findMine(current).stream()
                .map(playlist -> response(playlist, current.getId(), false))
                .toList();
    }

    @GetMapping("/archive")
    public List<PlaylistResponse> findArchive(@AuthenticationPrincipal User user) {
        User current = currentUser.require(user);
        return playlistService.findArchive().stream()
                .map(playlist -> response(playlist, current.getId(), playlistService.isSavedBy(current, playlist.getId())))
                .toList();
    }

    @GetMapping("/discovery")
    public List<PlaylistResponse> findDiscoveryArchive(@AuthenticationPrincipal User user) {
        User current = currentUser.require(user);
        return playlistService.findDiscoveryArchive().stream()
                .map(playlist -> response(playlist, current.getId(), playlistService.isSavedBy(current, playlist.getId())))
                .toList();
    }

    @GetMapping("/saved")
    public List<PlaylistResponse> findSaved(@AuthenticationPrincipal User user) {
        User current = currentUser.require(user);
        return playlistService.findSaved(current).stream()
                .map(saved -> response(saved.getPlaylist(), current.getId(), true))
                .toList();
    }

    @GetMapping("/{id}")
    public PlaylistResponse findById(@AuthenticationPrincipal User user, @PathVariable Long id) {
        User current = currentUser.require(user);
        return response(playlistService.findPublicById(id), current.getId(), playlistService.isSavedBy(current, id));
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public PlaylistResponse create(@AuthenticationPrincipal User user, @Valid @RequestBody PlaylistRequest request) {
        User current = currentUser.require(user);
        UserPlaylist playlist = playlistService.create(current, request);
        return response(playlist, current.getId(), false);
    }

    @PutMapping("/{id}")
    public PlaylistResponse update(@AuthenticationPrincipal User user, @PathVariable Long id,
                                   @Valid @RequestBody PlaylistRequest request) {
        User current = currentUser.require(user);
        return response(playlistService.update(current, id, request), current.getId(), false);
    }

    @PostMapping(value = "/{id}/cover-image", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public PlaylistResponse updateCoverImage(@AuthenticationPrincipal User user, @PathVariable Long id,
                                             @ModelAttribute PlaylistCoverUploadRequest request) {
        User current = currentUser.require(user);
        return response(playlistService.updateCoverImage(current, id, request.file()), current.getId(), false);
    }

    @DeleteMapping("/{id}/cover-image")
    public PlaylistResponse removeCoverImage(@AuthenticationPrincipal User user, @PathVariable Long id) {
        User current = currentUser.require(user);
        return response(playlistService.removeCoverImage(current, id), current.getId(), false);
    }

    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@AuthenticationPrincipal User user, @PathVariable Long id) {
        playlistService.delete(currentUser.require(user), id);
    }

    @PostMapping("/{id}/tabs")
    public PlaylistResponse addTab(@AuthenticationPrincipal User user, @PathVariable Long id,
                                   @Valid @RequestBody AddTabRequest request) {
        User current = currentUser.require(user);
        return response(playlistService.addTab(current, id, request.tabId()), current.getId(), false);
    }

    @DeleteMapping("/{playlistId}/tabs/{tabId}")
    public PlaylistResponse removeTab(@AuthenticationPrincipal User user, @PathVariable Long playlistId,
                                      @PathVariable Long tabId) {
        User current = currentUser.require(user);
        return response(playlistService.removeTab(current, playlistId, tabId), current.getId(), false);
    }

    @PostMapping("/{id}/save")
    public PlaylistResponse savePlaylist(@AuthenticationPrincipal User user, @PathVariable Long id) {
        User current = currentUser.require(user);
        return response(playlistService.savePlaylist(current, id), current.getId(), true);
    }

    @DeleteMapping("/{id}/save")
    public PlaylistResponse unsavePlaylist(@AuthenticationPrincipal User user, @PathVariable Long id) {
        User current = currentUser.require(user);
        return response(playlistService.unsavePlaylist(current, id), current.getId(), false);
    }

    public record PlaylistCoverUploadRequest(MultipartFile file) {
    }
}
