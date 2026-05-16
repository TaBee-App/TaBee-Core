package com.tabee.backend.tab;

import java.util.List;

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
import com.tabee.backend.security.CurrentUser;
import com.tabee.backend.user.User;

@RestController
@RequestMapping("/api/tabs")
public class TabController {
    private final TabService tabService;
    private final TabAudioStorage tabAudioStorage;
    private final CurrentUser currentUser;

    public TabController(TabService tabService, TabAudioStorage tabAudioStorage, CurrentUser currentUser) {
        this.tabService = tabService;
        this.tabAudioStorage = tabAudioStorage;
        this.currentUser = currentUser;
    }

    @GetMapping
    public List<TabResponse> findMine(@AuthenticationPrincipal User currentUser) {
        return tabService.findByOwner(this.currentUser.require(currentUser)).stream().map(TabResponse::from).toList();
    }

    @GetMapping("/public")
    public List<TabResponse> findPublicTabs() {
        return tabService.findPublicTabs().stream().map(TabResponse::from).toList();
    }

    @GetMapping("/public/{id}")
    public TabResponse findPublicById(@PathVariable Long id) {
        return TabResponse.from(tabService.findById(id));
    }

    @GetMapping("/{id}")
    public TabResponse findById(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        return TabResponse.from(tabService.findByOwnerAndId(this.currentUser.require(currentUser), id));
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
        return TabResponse.from(tabService.create(this.currentUser.require(currentUser), request));
    }

    @PutMapping("/{id}")
    public TabResponse update(@AuthenticationPrincipal User currentUser, @PathVariable Long id,
                              @Valid @RequestBody TabUpdateRequest request) {
        return TabResponse.from(tabService.update(this.currentUser.require(currentUser), id, request));
    }

    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@AuthenticationPrincipal User currentUser, @PathVariable Long id) {
        tabService.delete(this.currentUser.require(currentUser), id);
    }
}
