package com.tabee.backend.playlist;

import java.util.List;

import jakarta.validation.Valid;

import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

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

    @GetMapping
    public List<PlaylistResponse> findMine(@AuthenticationPrincipal User user) {
        return playlistService.findMine(currentUser.require(user)).stream().map(PlaylistResponse::from).toList();
    }

    @GetMapping("/{id}")
    public PlaylistResponse findById(@AuthenticationPrincipal User user, @PathVariable Long id) {
        return PlaylistResponse.from(playlistService.findById(currentUser.require(user), id));
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public PlaylistResponse create(@AuthenticationPrincipal User user, @Valid @RequestBody PlaylistRequest request) {
        return PlaylistResponse.from(playlistService.create(currentUser.require(user), request));
    }

    @PutMapping("/{id}")
    public PlaylistResponse update(@AuthenticationPrincipal User user, @PathVariable Long id,
                                   @Valid @RequestBody PlaylistRequest request) {
        return PlaylistResponse.from(playlistService.update(currentUser.require(user), id, request));
    }

    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@AuthenticationPrincipal User user, @PathVariable Long id) {
        playlistService.delete(currentUser.require(user), id);
    }

    @PostMapping("/{id}/tabs")
    public PlaylistResponse addTab(@AuthenticationPrincipal User user, @PathVariable Long id,
                                   @Valid @RequestBody AddTabRequest request) {
        return PlaylistResponse.from(playlistService.addTab(currentUser.require(user), id, request.tabId()));
    }

    @DeleteMapping("/{playlistId}/tabs/{tabId}")
    public PlaylistResponse removeTab(@AuthenticationPrincipal User user, @PathVariable Long playlistId,
                                      @PathVariable Long tabId) {
        return PlaylistResponse.from(playlistService.removeTab(currentUser.require(user), playlistId, tabId));
    }
}
