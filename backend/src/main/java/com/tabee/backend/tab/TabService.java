package com.tabee.backend.tab;

import java.util.List;

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

    public TabService(TabRepository tabRepository) {
        this.tabRepository = tabRepository;
    }

    @Transactional(readOnly = true)
    public List<Tab> findByOwner(User owner) {
        return tabRepository.findByOwnerIdOrderByCreatedAtDesc(owner.getId());
    }

    @Transactional(readOnly = true)
    public Tab findById(Long id) {
        return tabRepository.findById(id)
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
        Tab tab = findById(id);
        ensureOwner(tab, owner);

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

    @Transactional
    public void delete(Long id) {
        tabRepository.delete(findById(id));
    }

    @Transactional
    public void delete(User owner, Long id) {
        Tab tab = findById(id);
        ensureOwner(tab, owner);
        tabRepository.delete(tab);
    }

    private void ensureOwner(Tab tab, User owner) {
        if (!tab.getOwner().getId().equals(owner.getId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Tab does not belong to current user");
        }
    }
}
