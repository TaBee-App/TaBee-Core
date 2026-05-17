package com.tabee.backend.auth;

import jakarta.validation.Valid;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import com.tabee.backend.auth.AuthDtos.AuthResponse;
import com.tabee.backend.auth.AuthDtos.LoginRequest;
import com.tabee.backend.auth.AuthDtos.PasswordResetCodeRequest;
import com.tabee.backend.auth.AuthDtos.PasswordResetCodeResponse;
import com.tabee.backend.auth.AuthDtos.PasswordResetConfirmRequest;
import com.tabee.backend.auth.AuthDtos.RegisterCodeRequest;
import com.tabee.backend.auth.AuthDtos.RegisterCodeResponse;
import com.tabee.backend.auth.AuthDtos.RegisterRequest;

@RestController
@RequestMapping("/api/auth")
public class AuthController {
    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/register")
    @ResponseStatus(HttpStatus.CREATED)
    public AuthResponse register(@Valid @RequestBody RegisterRequest request) {
        return authService.register(request);
    }

    @PostMapping("/register/code")
    public RegisterCodeResponse requestRegisterCode(@Valid @RequestBody RegisterCodeRequest request) {
        return authService.requestRegisterCode(request);
    }

    @PostMapping("/login")
    public AuthResponse login(@Valid @RequestBody LoginRequest request) {
        return authService.login(request);
    }

    @PostMapping("/password-reset/code")
    public PasswordResetCodeResponse requestPasswordResetCode(@Valid @RequestBody PasswordResetCodeRequest request) {
        return authService.requestPasswordResetCode(request);
    }

    @PostMapping("/password-reset/confirm")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void resetPassword(@Valid @RequestBody PasswordResetConfirmRequest request) {
        authService.resetPassword(request);
    }
}
