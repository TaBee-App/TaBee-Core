package com.tabee.backend.playlist;

import java.util.List;

import jakarta.persistence.EntityManager;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.playlist.PlaylistDtos.PlaylistRequest;
import com.tabee.backend.tab.Tab;
import com.tabee.backend.tab.TabService;
import com.tabee.backend.user.User;

@Service
public class PlaylistService {
    private final UserPlaylistRepository playlistRepository;
    private final PlaylistTabRepository playlistTabRepository;
    private final SavedPlaylistRepository savedPlaylistRepository;
    private final TabService tabService;
    private final EntityManager entityManager;

    public PlaylistService(UserPlaylistRepository playlistRepository,
                           PlaylistTabRepository playlistTabRepository,
                           SavedPlaylistRepository savedPlaylistRepository,
                           TabService tabService,
                           EntityManager entityManager) {
        this.playlistRepository = playlistRepository;
        this.playlistTabRepository = playlistTabRepository;
        this.savedPlaylistRepository = savedPlaylistRepository;
        this.tabService = tabService;
        this.entityManager = entityManager;
    }

    @Transactional(readOnly = true)
    public List<UserPlaylist> findMine(User owner) {
        return playlistRepository.findByOwnerIdOrderByCreatedAtDesc(owner.getId());
    }

    @Transactional(readOnly = true)
    public List<UserPlaylist> findArchive() {
        return playlistRepository.findAllByOrderByCreatedAtDesc();
    }

    @Transactional(readOnly = true)
    public List<SavedPlaylist> findSaved(User user) {
        return savedPlaylistRepository.findByUser_IdOrderBySavedAtDesc(user.getId());
    }

    @Transactional(readOnly = true)
    public UserPlaylist findPublicById(Long id) {
        return playlistRepository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Playlist not found"));
    }

    @Transactional(readOnly = true)
    public boolean isSavedBy(User user, Long playlistId) {
        return savedPlaylistRepository.existsByUser_IdAndPlaylist_Id(user.getId(), playlistId);
    }

    @Transactional(readOnly = true)
    public long countSaves(Long playlistId) {
        return savedPlaylistRepository.countByPlaylist_Id(playlistId);
    }

    @Transactional(readOnly = true)
    public UserPlaylist findById(User owner, Long id) {
        return playlistRepository.findByIdAndOwnerId(id, owner.getId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Playlist not found"));
    }

    @Transactional
    public UserPlaylist create(User owner, PlaylistRequest request) {
        UserPlaylist playlist = new UserPlaylist();
        playlist.setOwner(owner);
        playlist.setName(request.name());
        playlist.setDescription(request.description());
        return playlistRepository.save(playlist);
    }

    @Transactional
    public UserPlaylist update(User owner, Long id, PlaylistRequest request) {
        UserPlaylist playlist = findById(owner, id);
        playlist.setName(request.name());
        playlist.setDescription(request.description());
        return playlistRepository.save(playlist);
    }

    @Transactional
    public void delete(User owner, Long id) {
        playlistRepository.delete(findById(owner, id));
    }

    @Transactional
    public UserPlaylist addTab(User owner, Long playlistId, Long tabId) {
        findById(owner, playlistId);
        Tab tab = tabService.findByOwnerAndId(owner, tabId);

        playlistTabRepository.insertIgnoreConflict(playlistId, tab.getId());
        entityManager.flush();
        entityManager.clear();
        return findById(owner, playlistId);
    }

    @Transactional
    public UserPlaylist removeTab(User owner, Long playlistId, Long tabId) {
        findById(owner, playlistId);
        int deleted = playlistTabRepository.deleteByPlaylistIdAndTabId(playlistId, tabId);
        if (deleted == 0) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Tab is not in this playlist");
        }
        entityManager.flush();
        entityManager.clear();
        return findById(owner, playlistId);
    }

    @Transactional
    public UserPlaylist savePlaylist(User user, Long playlistId) {
        UserPlaylist playlist = findPublicById(playlistId);
        if (playlist.getOwner().getId().equals(user.getId())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "You already created this playlist");
        }
        SavedPlaylist savedPlaylist = new SavedPlaylist();
        savedPlaylist.setId(new SavedPlaylistId(user.getId(), playlistId));
        savedPlaylist.setUser(user);
        savedPlaylist.setPlaylist(playlist);
        savedPlaylistRepository.save(savedPlaylist);
        return findPublicById(playlistId);
    }

    @Transactional
    public UserPlaylist unsavePlaylist(User user, Long playlistId) {
        savedPlaylistRepository.deleteById(new SavedPlaylistId(user.getId(), playlistId));
        return findPublicById(playlistId);
    }

}
