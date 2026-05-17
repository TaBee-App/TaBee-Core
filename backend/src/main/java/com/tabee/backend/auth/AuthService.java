package com.tabee.backend.auth;

import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.common.ImageStorageService;
import com.tabee.backend.auth.AuthDtos.AuthResponse;
import com.tabee.backend.auth.AuthDtos.RegisterCodeRequest;
import com.tabee.backend.auth.AuthDtos.RegisterCodeResponse;
import com.tabee.backend.auth.AuthDtos.PasswordResetCodeRequest;
import com.tabee.backend.auth.AuthDtos.PasswordResetCodeResponse;
import com.tabee.backend.auth.AuthDtos.PasswordResetConfirmRequest;
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
    private final EmailVerificationService emailVerificationService;
    private final PasswordResetService passwordResetService;
    private final ImageStorageService imageStorageService;

    public AuthService(UserRepository userRepository, UserService userService, PasswordEncoder passwordEncoder,
                       AuthTokenService authTokenService, EmailVerificationService emailVerificationService,
                       PasswordResetService passwordResetService,
                       ImageStorageService imageStorageService) {
        this.userRepository = userRepository;
        this.userService = userService;
        this.passwordEncoder = passwordEncoder;
        this.authTokenService = authTokenService;
        this.emailVerificationService = emailVerificationService;
        this.passwordResetService = passwordResetService;
        this.imageStorageService = imageStorageService;
    }

    public RegisterCodeResponse requestRegisterCode(RegisterCodeRequest request) {
        userService.validateNewUser(request.toUserRequest());
        return emailVerificationService.sendCode(new RegisterRequest(
                request.username(),
                request.email(),
                request.password(),
                request.fullName(),
                "000000"
        ));
    }

    public AuthResponse register(RegisterRequest request) {
        userService.validateNewUser(request.toUserRequest());
        emailVerificationService.verify(request);
        User user = userService.create(request.toUserRequest());
        return new AuthResponse(UserResponse.from(user, imageStorageService.url(user.getProfileImageFilename())), authTokenService.issueToken(user), "Registered successfully");
    }

    public AuthResponse login(LoginRequest request) {
        User user = userRepository.findByUsername(request.usernameOrEmail())
                .or(() -> userRepository.findByEmail(request.usernameOrEmail()))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "No account was found for this username or email"));

        if (!passwordEncoder.matches(request.password(), user.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Password is incorrect");
        }

        return new AuthResponse(UserResponse.from(user, imageStorageService.url(user.getProfileImageFilename())), authTokenService.issueToken(user), "Login successful");
    }

    public PasswordResetCodeResponse requestPasswordResetCode(PasswordResetCodeRequest request) {
        return passwordResetService.sendResetCode(request);
    }

    public void resetPassword(PasswordResetConfirmRequest request) {
        passwordResetService.resetPassword(request);
    }
}
