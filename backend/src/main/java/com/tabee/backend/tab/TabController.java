package com.tabee.backend.tab;

import java.util.List;
import java.util.Set;

import jakarta.validation.Valid;

import org.springframework.http.HttpStatus;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.http.MediaType;
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

import com.tabee.backend.tab.TabDtos.TabRequest;
import com.tabee.backend.tab.TabDtos.TabResponse;
import com.tabee.backend.tab.TabDtos.TabUpdateRequest;
import com.tabee.backend.common.ImageStorageService;
import com.tabee.backend.security.CurrentUser;
import com.tabee.backend.user.User;

@RestController
@RequestMapping("/api/tabs")
public class TabController {
    private final TabService tabService;
    private final TabAudioStorage tabAudioStorage;
    private final CurrentUser currentUser;
    private final ImageStorageService imageStorageService;

    public TabController(TabService tabService, TabAudioStorage tabAudioStorage, CurrentUser currentUser,
                         ImageStorageService imageStorageService) {
        this.tabService = tabService;
        this.tabAudioStorage = tabAudioStorage;
        this.currentUser = currentUser;
        this.imageStorageService = imageStorageService;
    }

    private TabResponse response(Tab tab, Long currentUserId, boolean favoritedByCurrentUser, long favoriteCount) {
        return TabResponse.from(
                tab,
                currentUserId,
                favoritedByCurrentUser,
                favoriteCount,
                imageStorageService.url(tab.getOwner().getProfileImageFilename())
        );
    }

    @GetMapping
    public List<TabResponse> findMine(@AuthenticationPrincipal User currentUser) {
        User current = this.currentUser.require(currentUser);
        return tabService.findByOwner(current).stream()
                .map(tab -> response(tab, current.getId(), false, tabService.countFavorites(tab.getId())))
                .toList();
    }

    @GetMapping("/public")
    public List<TabResponse> findPublicTabs(@AuthenticationPrincipal User currentUser) {
        User current = this.currentUser.require(currentUser);
        Set<Long> favoriteTabIds = tabService.findFavoriteTabIds(current);
        return tabService.findPublicTabs().stream()
                .map(tab -> response(tab, current.getId(), favoriteTabIds.contains(tab.getId()), tabService.countFavorites(tab.getId())))
                .toList();
    }

    @GetMapping("/discovery")
    public List<TabResponse> findDiscoveryTabs(@AuthenticationPrincipal User currentUser) {
        User current = this.currentUser.require(currentUser);
        Set<Long> favoriteTabIds = tabService.findFavoriteTabIds(current);
        return tabService.findDiscoveryTabs().stream()
                .map(tab -> response(tab, current.getId(), favoriteTabIds.contains(tab.getId()), tabService.countFavorites(tab.getId())))
                .toList();
    }

    @GetMapping("/public/{id}")
    public TabResponse findPublicById(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        User current = this.currentUser.require(currentUser);
        return response(tabService.findById(id), current.getId(), tabService.isFavoritedBy(current, id), tabService.countFavorites(id));
    }

    @GetMapping("/favorites")
    public List<TabResponse> findFavorites(@AuthenticationPrincipal User currentUser) {
        User current = this.currentUser.require(currentUser);
        return tabService.findFavorites(current).stream()
                .map(favorite -> response(favorite.getTab(), current.getId(), true, tabService.countFavorites(favorite.getTab().getId())))
                .toList();
    }

    @GetMapping("/{id}")
    public TabResponse findById(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        User current = this.currentUser.require(currentUser);
        return response(tabService.findByOwnerAndId(current, id), current.getId(), false, tabService.countFavorites(id));
    }

    @GetMapping("/{id}/audio")
    public ResponseEntity<Resource> findAudio(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        Tab tab = tabService.findByOwnerAndId(this.currentUser.require(currentUser), id);
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .body(tabAudioStorage.load(tab));
    }

    @GetMapping("/public/{id}/audio")
    public ResponseEntity<Resource> findPublicAudio(@PathVariable Long id) {
        Tab tab = tabService.findById(id);
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .body(tabAudioStorage.load(tab));
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public TabResponse create(@AuthenticationPrincipal User currentUser, @Valid @RequestBody TabRequest request) {
        User current = this.currentUser.require(currentUser);
        Tab tab = tabService.create(current, request);
        return response(tab, current.getId(), false, tabService.countFavorites(tab.getId()));
    }

    @PutMapping("/{id}")
    public TabResponse update(@AuthenticationPrincipal User currentUser, @PathVariable Long id,
                              @Valid @RequestBody TabUpdateRequest request) {
        User current = this.currentUser.require(currentUser);
        return response(tabService.update(current, id, request), current.getId(), false, tabService.countFavorites(id));
    }

    @PostMapping("/{id}/favorite")
    public TabResponse favoriteTab(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        User current = this.currentUser.require(currentUser);
        return response(tabService.favoriteTab(current, id), current.getId(), true, tabService.countFavorites(id));
    }

    @DeleteMapping("/{id}/favorite")
    public TabResponse unfavoriteTab(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        User current = this.currentUser.require(currentUser);
        return response(tabService.unfavoriteTab(current, id), current.getId(), false, tabService.countFavorites(id));
    }

    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        tabService.delete(this.currentUser.require(currentUser), id);
    }
}
