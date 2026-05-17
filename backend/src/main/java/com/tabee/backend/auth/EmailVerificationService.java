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
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.auth.AuthDtos.RegisterCodeResponse;
import com.tabee.backend.auth.AuthDtos.RegisterRequest;

@Service
public class EmailVerificationService {
    private static final Logger LOGGER = LoggerFactory.getLogger(EmailVerificationService.class);
    private static final SecureRandom RANDOM = new SecureRandom();

    private final Map<String, PendingVerification> pendingVerifications = new ConcurrentHashMap<>();
    private final ObjectProvider<JavaMailSender> mailSenderProvider;
    private final boolean devMode;
    private final String fromAddress;
    private final String smtpHost;

    public EmailVerificationService(ObjectProvider<JavaMailSender> mailSenderProvider,
                                    @Value("${tabee.email.dev-mode:true}") boolean devMode,
                                    @Value("${tabee.email.from:noreply@tabee.local}") String fromAddress,
                                    @Value("${spring.mail.host:}") String smtpHost) {
        this.mailSenderProvider = mailSenderProvider;
        this.devMode = devMode;
        this.fromAddress = fromAddress;
        this.smtpHost = smtpHost == null ? "" : smtpHost.trim();
    }

    public RegisterCodeResponse sendCode(RegisterRequest request) {
        String email = normalizeEmail(request.email());
        String code = "%06d".formatted(RANDOM.nextInt(1_000_000));
        pendingVerifications.put(email, new PendingVerification(code, OffsetDateTime.now().plusMinutes(10)));

        if (devMode) {
            LOGGER.info("TaBee registration verification code for {} is {}", email, code);
            return new RegisterCodeResponse(
                    "Verification code generated. Development mode returns the code directly.",
                    code
            );
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
        message.setTo(email);
        message.setSubject("TaBee email verification code");
        message.setText("Your TaBee verification code is: " + code + "\n\nThis code expires in 10 minutes.");
        try {
            mailSender.send(message);
        } catch (MailException exception) {
            LOGGER.warn("Could not send TaBee verification email to {}", email, exception);
            throw new ResponseStatusException(
                    HttpStatus.BAD_GATEWAY,
                    "Verification email could not be sent. Check SMTP host, username, app password, and STARTTLS settings.",
                    exception
            );
        }

        return new RegisterCodeResponse("Verification code sent to your email address.", null);
    }

    public void verify(RegisterRequest request) {
        String email = normalizeEmail(request.email());
        PendingVerification pending = pendingVerifications.get(email);
        if (pending == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Verification code was not requested or expired");
        }
        if (OffsetDateTime.now().isAfter(pending.expiresAt())) {
            pendingVerifications.remove(email);
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Verification code expired");
        }
        if (!pending.code().equals(request.verificationCode())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Verification code is incorrect");
        }
        pendingVerifications.remove(email);
    }

    private String normalizeEmail(String email) {
        return email.trim().toLowerCase(Locale.ROOT);
    }

    private record PendingVerification(String code, OffsetDateTime expiresAt) {
    }
}
