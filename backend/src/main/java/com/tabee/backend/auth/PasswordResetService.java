package com.tabee.backend.auth;

import java.security.SecureRandom;
import java.time.OffsetDateTime;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.mail.MailException;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.auth.AuthDtos.PasswordResetCodeRequest;
import com.tabee.backend.auth.AuthDtos.PasswordResetCodeResponse;
import com.tabee.backend.auth.AuthDtos.PasswordResetConfirmRequest;
import com.tabee.backend.user.PasswordPolicy;
import com.tabee.backend.user.User;
import com.tabee.backend.user.UserRepository;

@Service
public class PasswordResetService {
    private static final Logger LOGGER = LoggerFactory.getLogger(PasswordResetService.class);
    private static final SecureRandom RANDOM = new SecureRandom();
    private static final String GENERIC_RESPONSE = "If this email belongs to an account, a reset code has been sent.";

    private final Map<String, PendingReset> pendingResets = new ConcurrentHashMap<>();
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final ObjectProvider<JavaMailSender> mailSenderProvider;
    private final boolean devMode;
    private final String fromAddress;
    private final String smtpHost;

    public PasswordResetService(UserRepository userRepository,
                                PasswordEncoder passwordEncoder,
                                ObjectProvider<JavaMailSender> mailSenderProvider,
                                @Value("${tabee.email.dev-mode:true}") boolean devMode,
                                @Value("${tabee.email.from:noreply@tabee.local}") String fromAddress,
                                @Value("${spring.mail.host:}") String smtpHost) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.mailSenderProvider = mailSenderProvider;
        this.devMode = devMode;
        this.fromAddress = fromAddress;
        this.smtpHost = smtpHost == null ? "" : smtpHost.trim();
    }

    public PasswordResetCodeResponse sendResetCode(PasswordResetCodeRequest request) {
        String email = normalizeEmail(request.email());
        User user = userRepository.findByEmail(email).orElse(null);
        if (user == null) {
            return new PasswordResetCodeResponse(GENERIC_RESPONSE, null);
        }

        String code = "%06d".formatted(RANDOM.nextInt(1_000_000));
        pendingResets.put(email, new PendingReset(code, OffsetDateTime.now().plusMinutes(10)));

        if (devMode) {
            LOGGER.info("TaBee password reset code for {} is {}", email, code);
            return new PasswordResetCodeResponse(GENERIC_RESPONSE, code);
        }

        sendEmail(email, code);
        return new PasswordResetCodeResponse(GENERIC_RESPONSE, null);
    }

    public void resetPassword(PasswordResetConfirmRequest request) {
        String email = normalizeEmail(request.email());
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.BAD_REQUEST, "Reset code is invalid or expired"));
        PendingReset pending = pendingResets.get(email);
        if (pending == null || OffsetDateTime.now().isAfter(pending.expiresAt())) {
            pendingResets.remove(email);
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Reset code is invalid or expired");
        }
        if (!pending.code().equals(request.verificationCode())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Reset code is incorrect");
        }

        PasswordPolicy.validate(request.newPassword(), user.getUsername(), user.getEmail(), user.getFullName());
        user.setPasswordHash(passwordEncoder.encode(request.newPassword()));
        userRepository.save(user);
        pendingResets.remove(email);
    }

    private void sendEmail(String email, String code) {
        JavaMailSender mailSender = mailSenderProvider.getIfAvailable();
        if (mailSender == null) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Mail sender is not configured");
        }
        if (smtpHost.isBlank()) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "SMTP_HOST is not configured");
        }

        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom(fromAddress);
        message.setTo(email);
        message.setSubject("TaBee password reset code");
        message.setText("Your TaBee password reset code is: " + code + "\n\nThis code expires in 10 minutes.");
        try {
            mailSender.send(message);
        } catch (MailException exception) {
            LOGGER.warn("Could not send TaBee password reset email to {}", email, exception);
            throw new ResponseStatusException(
                    HttpStatus.BAD_GATEWAY,
                    "Password reset email could not be sent. Check SMTP settings.",
                    exception
            );
        }
    }

    private String normalizeEmail(String email) {
        return email.trim().toLowerCase(Locale.ROOT);
    }

    private record PendingReset(String code, OffsetDateTime expiresAt) {
    }
}
