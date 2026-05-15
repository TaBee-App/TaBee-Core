package com.tabee.backend.security;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.stereotype.Service;

import com.tabee.backend.user.User;
import com.tabee.backend.user.UserRepository;

@Service
public class AuthTokenService {
    private final UserRepository userRepository;
    private final Map<String, Long> tokens = new ConcurrentHashMap<>();

    public AuthTokenService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public String issueToken(User user) {
        String token = UUID.randomUUID().toString();
        tokens.put(token, user.getId());
        return token;
    }

    public Optional<User> findUserByToken(String token) {
        Long userId = tokens.get(token);
        if (userId == null) {
            return Optional.empty();
        }
        return userRepository.findById(userId);
    }
}
