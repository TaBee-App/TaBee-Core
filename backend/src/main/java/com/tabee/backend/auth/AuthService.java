package com.tabee.backend.auth;

import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.auth.AuthDtos.AuthResponse;
import com.tabee.backend.auth.AuthDtos.LoginRequest;
import com.tabee.backend.auth.AuthDtos.RegisterRequest;
import com.tabee.backend.security.AuthTokenService;
import com.tabee.backend.user.User;
import com.tabee.backend.user.UserDtos.UserResponse;
import com.tabee.backend.user.UserRepository;
import com.tabee.backend.user.UserService;

@Service
public class AuthService {
    private final UserRepository userRepository;
    private final UserService userService;
    private final PasswordEncoder passwordEncoder;
    private final AuthTokenService authTokenService;

    public AuthService(UserRepository userRepository, UserService userService, PasswordEncoder passwordEncoder,
                       AuthTokenService authTokenService) {
        this.userRepository = userRepository;
        this.userService = userService;
        this.passwordEncoder = passwordEncoder;
        this.authTokenService = authTokenService;
    }

    public AuthResponse register(RegisterRequest request) {
        User user = userService.create(request.toUserRequest());
        return new AuthResponse(UserResponse.from(user), authTokenService.issueToken(user), "Registered successfully");
    }

    public AuthResponse login(LoginRequest request) {
        User user = userRepository.findByUsername(request.usernameOrEmail())
                .or(() -> userRepository.findByEmail(request.usernameOrEmail()))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid credentials"));

        if (!passwordEncoder.matches(request.password(), user.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid credentials");
        }

        return new AuthResponse(UserResponse.from(user), authTokenService.issueToken(user), "Login successful");
    }
}
