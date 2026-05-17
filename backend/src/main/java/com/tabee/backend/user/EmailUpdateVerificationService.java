package com.tabee.backend.user;

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
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.user.UserDtos.EmailUpdateCodeResponse;

@Service
public class EmailUpdateVerificationService {
    private static final Logger LOGGER = LoggerFactory.getLogger(EmailUpdateVerificationService.class);
    private static final SecureRandom RANDOM = new SecureRandom();

    private final Map<String, PendingEmailUpdate> pendingUpdates = new ConcurrentHashMap<>();
    private final ObjectProvider<JavaMailSender> mailSenderProvider;
    private final boolean devMode;
    private final String fromAddress;
    private final String smtpHost;

    public EmailUpdateVerificationService(ObjectProvider<JavaMailSender> mailSenderProvider,
                                          @Value("${tabee.email.dev-mode:true}") boolean devMode,
                                          @Value("${tabee.email.from:noreply@tabee.local}") String fromAddress,
                                          @Value("${spring.mail.host:}") String smtpHost) {
        this.mailSenderProvider = mailSenderProvider;
        this.devMode = devMode;
        this.fromAddress = fromAddress;
        this.smtpHost = smtpHost == null ? "" : smtpHost.trim();
    }

    public EmailUpdateCodeResponse sendCode(Long userId, String email) {
        String normalizedEmail = normalizeEmail(email);
        String code = "%06d".formatted(RANDOM.nextInt(1_000_000));
        pendingUpdates.put(key(userId, normalizedEmail), new PendingEmailUpdate(code, OffsetDateTime.now().plusMinutes(10)));

        if (devMode) {
            LOGGER.info("TaBee email update code for user {} and {} is {}", userId, normalizedEmail, code);
            return new EmailUpdateCodeResponse("Verification code generated.", code);
        }

        JavaMailSender mailSender = mailSenderProvider.getIfAvailable();
        if (mailSender == null) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Mail sender is not configured");
        }
        if (smtpHost.isBlank()) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "SMTP_HOST is not configured");
        }

        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom(fromAddress);
        message.setTo(normalizedEmail);
        message.setSubject("TaBee email change verification code");
        message.setText("Your TaBee email change code is: " + code + "\n\nThis code expires in 10 minutes.");
        try {
            mailSender.send(message);
        } catch (MailException exception) {
            LOGGER.warn("Could not send TaBee email update verification to {}", normalizedEmail, exception);
            throw new ResponseStatusException(
                    HttpStatus.BAD_GATEWAY,
                    "Email verification could not be sent. Check SMTP settings.",
                    exception
            );
        }

        return new EmailUpdateCodeResponse("Verification code sent to your new email address.", null);
    }

    public void verify(Long userId, String email, String verificationCode) {
        String normalizedEmail = normalizeEmail(email);
        String key = key(userId, normalizedEmail);
        PendingEmailUpdate pending = pendingUpdates.get(key);
        if (pending == null || OffsetDateTime.now().isAfter(pending.expiresAt())) {
            pendingUpdates.remove(key);
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Verification code is invalid or expired");
        }
        if (!pending.code().equals(verificationCode)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Verification code is incorrect");
        }
        pendingUpdates.remove(key);
    }

    private String key(Long userId, String email) {
        return userId + ":" + email;
    }

    private String normalizeEmail(String email) {
        return email.trim().toLowerCase(Locale.ROOT);
    }

    private record PendingEmailUpdate(String code, OffsetDateTime expiresAt) {
    }
}
