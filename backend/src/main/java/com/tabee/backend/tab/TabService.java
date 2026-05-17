package com.tabee.backend.tab;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.tab.TabDtos.TabRequest;
import com.tabee.backend.tab.TabDtos.TabUpdateRequest;
import com.tabee.backend.user.User;

@Service
public class TabService {
    private final TabRepository tabRepository;
    private final FavoriteTabRepository favoriteTabRepository;

    public TabService(TabRepository tabRepository, FavoriteTabRepository favoriteTabRepository) {
        this.tabRepository = tabRepository;
        this.favoriteTabRepository = favoriteTabRepository;
    }

    @Transactional(readOnly = true)
    public List<Tab> findByOwner(User owner) {
        return tabRepository.findByOwnerIdOrderByCreatedAtDesc(owner.getId());
    }

    @Transactional(readOnly = true)
    public List<Tab> findPublicTabs() {
        return tabRepository.findAllByOrderByCreatedAtDesc();
    }

    @Transactional(readOnly = true)
    public List<FavoriteTab> findFavorites(User user) {
        return favoriteTabRepository.findByUser_IdOrderByFavoritedAtDesc(user.getId());
    }

    @Transactional(readOnly = true)
    public Set<Long> findFavoriteTabIds(User user) {
        return favoriteTabRepository.findTabIdsByUserId(user.getId()).stream().collect(Collectors.toSet());
    }

    @Transactional(readOnly = true)
    public Tab findById(Long id) {
        return tabRepository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Tab not found"));
    }

    @Transactional(readOnly = true)
    public Tab findByOwnerAndId(User owner, Long id) {
        return tabRepository.findByIdAndOwnerId(id, owner.getId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Tab not found"));
    }

    @Transactional
    public Tab create(User owner, TabRequest request) {
        TabData tabData = new TabData();
        tabData.setTuning(request.tuning());
        tabData.setEstimatedTempo(request.estimatedTempo());
        tabData.setJsonData(request.jsonData());

        Tab tab = new Tab();
        tab.setOwner(owner);
        tab.setTabData(tabData);
        tab.setTitle(request.title());
        tab.setArtist(request.artist());
        return tabRepository.save(tab);
    }

    @Transactional
    public Tab update(User owner, Long id, TabUpdateRequest request) {
        Tab tab = findByOwnerAndId(owner, id);

        if (request.title() != null && !request.title().isBlank()) {
            tab.setTitle(request.title());
        }
        if (request.artist() != null) {
            tab.setArtist(request.artist());
        }
        if (request.tuning() != null) {
            tab.getTabData().setTuning(request.tuning());
        }
        if (request.estimatedTempo() != null) {
            tab.getTabData().setEstimatedTempo(request.estimatedTempo());
        }
        if (request.jsonData() != null) {
            tab.getTabData().setJsonData(request.jsonData());
        }

        return tabRepository.save(tab);
    }

    @Transactional(readOnly = true)
    public boolean isFavoritedBy(User user, Long tabId) {
        return favoriteTabRepository.existsByUser_IdAndTab_Id(user.getId(), tabId);
    }

    @Transactional(readOnly = true)
    public long countFavorites(Long tabId) {
        return favoriteTabRepository.countByTab_Id(tabId);
    }

    @Transactional
    public Tab favoriteTab(User user, Long tabId) {
        Tab tab = findById(tabId);
        if (tab.getOwner().getId().equals(user.getId())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "You already created this tab");
        }
        FavoriteTabId id = new FavoriteTabId(user.getId(), tabId);
        if (!favoriteTabRepository.existsById(id)) {
            FavoriteTab favoriteTab = new FavoriteTab();
            favoriteTab.setId(id);
            favoriteTab.setUser(user);
            favoriteTab.setTab(tab);
            favoriteTabRepository.save(favoriteTab);
        }
        return tab;
    }

    @Transactional
    public Tab unfavoriteTab(User user, Long tabId) {
        favoriteTabRepository.deleteById(new FavoriteTabId(user.getId(), tabId));
        return findById(tabId);
    }

    @Transactional
    public void delete(Long id) {
        tabRepository.delete(findById(id));
    }

    @Transactional
    public void delete(User owner, Long id) {
        tabRepository.delete(findByOwnerAndId(owner, id));
    }
}
