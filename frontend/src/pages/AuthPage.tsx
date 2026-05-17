import { KeyRound, LogIn, UserPlus } from "lucide-react";
import type { FormEvent } from "react";
import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { confirmPasswordReset, login, register, requestPasswordResetCode, requestRegisterCode } from "../api/authApi";
import { isAuthenticated } from "../api/authSession";
import { errorMessage } from "../lib/errors";
import { passwordPolicyError, passwordRules } from "../lib/passwordPolicy";

type AuthMode = "login" | "register";

type LocationState = {
  from?: {
    pathname?: string;
  };
};

export function AuthPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as LocationState | null;
  const redirectTo = state?.from?.pathname || "/";
  const [mode, setMode] = useState<AuthMode>("login");
  const [usernameOrEmail, setUsernameOrEmail] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState("");
  const [verificationOpen, setVerificationOpen] = useState(false);
  const [verificationCode, setVerificationCode] = useState("");
  const [verificationHint, setVerificationHint] = useState("");
  const [resetOpen, setResetOpen] = useState(false);
  const [resetStep, setResetStep] = useState<"email" | "code">("email");
  const [resetEmail, setResetEmail] = useState("");
  const [resetCode, setResetCode] = useState("");
  const [resetPassword, setResetPassword] = useState("");
  const [resetConfirmPassword, setResetConfirmPassword] = useState("");
  const [resetHint, setResetHint] = useState("");
  const [resetLoading, setResetLoading] = useState(false);
  const registerPasswordRules = passwordRules(password, { username, email, fullName });
  const resetPasswordRules = passwordRules(resetPassword, { email: resetEmail });

  if (isAuthenticated()) {
    return <Navigate to={redirectTo} replace />;
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      if (mode === "login") {
        await login({ usernameOrEmail, password });
      } else {
        if (password !== confirmPassword) {
          setError("Passwords do not match.");
          setLoading(false);
          return;
        }
        const policyError = passwordPolicyError(password, { username, email, fullName });
        if (policyError) {
          setError(policyError);
          setLoading(false);
          return;
        }
        const verification = await requestRegisterCode({ username, email, fullName: fullName || undefined, password });
        setVerificationHint(verification.devCode ? `Development code: ${verification.devCode}` : verification.message);
        setVerificationCode("");
        setVerificationOpen(true);
        return;
      }
      navigate(redirectTo, { replace: true });
    } catch (caught) {
      setError(errorMessage(caught, "Authentication failed."));
    } finally {
      setLoading(false);
    }
  }

  async function confirmRegistration() {
    setVerifying(true);
    setError("");
    try {
      await register({
        username,
        email,
        fullName: fullName || undefined,
        password,
        verificationCode
      });
      navigate(redirectTo, { replace: true });
    } catch (caught) {
      setError(errorMessage(caught, "Verification failed."));
    } finally {
      setVerifying(false);
    }
  }

  function openPasswordReset() {
    setResetEmail(usernameOrEmail.includes("@") ? usernameOrEmail : "");
    setResetCode("");
    setResetPassword("");
    setResetConfirmPassword("");
    setResetHint("");
    setResetStep("email");
    setError("");
    setResetOpen(true);
  }

  async function requestResetCode() {
    setResetLoading(true);
    setError("");
    try {
      const response = await requestPasswordResetCode({ email: resetEmail });
      setResetHint(response.devCode ? `Development code: ${response.devCode}` : response.message);
      setResetCode("");
      setResetPassword("");
      setResetConfirmPassword("");
      setResetStep("code");
    } catch (caught) {
      setError(errorMessage(caught, "Could not send reset code."));
    } finally {
      setResetLoading(false);
    }
  }

  async function resetForgottenPassword() {
    if (resetPassword !== resetConfirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    const policyError = passwordPolicyError(resetPassword, { email: resetEmail });
    if (policyError) {
      setError(policyError);
      return;
    }

    setResetLoading(true);
    setError("");
    try {
      await confirmPasswordReset({
        email: resetEmail,
        verificationCode: resetCode,
        newPassword: resetPassword
      });
      setResetOpen(false);
      setMode("login");
      setUsernameOrEmail(resetEmail);
      setPassword("");
      setError("");
    } catch (caught) {
      setError(errorMessage(caught, "Could not reset password."));
    } finally {
      setResetLoading(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-panel">
        <div className="auth-brand">
          <img className="brand-mark auth-brand-logo" src="/tabee-logo.png" alt="" />
          <div>
            <h1 className="brand-title">TaBee</h1>
            <p className="brand-subtitle">Connect to your backend workspace</p>
          </div>
        </div>

        <div className="auth-tabs" role="tablist" aria-label="Authentication mode">
          <button className={mode === "login" ? "active" : ""} type="button" onClick={() => setMode("login")}>
            <LogIn size={17} />
            Login
          </button>
          <button className={mode === "register" ? "active" : ""} type="button" onClick={() => setMode("register")}>
            <UserPlus size={17} />
            Register
          </button>
        </div>

        <form onSubmit={submit}>
          {mode === "login" ? (
            <label className="field">
              <span>Username or email</span>
              <input
                value={usernameOrEmail}
                onChange={(event) => setUsernameOrEmail(event.target.value)}
                autoComplete="username"
                required
              />
            </label>
          ) : (
            <>
              <label className="field">
                <span>Username</span>
                <input
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  autoComplete="username"
                  required
                />
              </label>
              <label className="field">
                <span>Email</span>
                <input
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  type="email"
                  required
                />
              </label>
              <label className="field">
                <span>Full name</span>
                <input value={fullName} onChange={(event) => setFullName(event.target.value)} autoComplete="name" />
              </label>
            </>
          )}

          <label className="field">
            <span>Password</span>
            <input
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              type="password"
              minLength={mode === "register" ? 8 : 6}
              required
            />
          </label>

          {mode === "login" ? (
            <button className="text-link auth-forgot" type="button" onClick={openPasswordReset}>
              Forgot password?
            </button>
          ) : null}

          {mode === "register" ? (
            <ul className="password-rules">
              {registerPasswordRules.map((rule) => (
                <li className={rule.passed ? "passed" : ""} key={rule.id}>
                  {rule.label}
                </li>
              ))}
            </ul>
          ) : null}

          {mode === "register" ? (
            <label className="field">
              <span>Confirm password</span>
              <input
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                autoComplete="new-password"
                type="password"
                minLength={8}
                required
              />
            </label>
          ) : null}

          {error ? <div className="form-error">{error}</div> : null}

          <button className="btn primary full auth-submit" disabled={loading} type="submit">
            {mode === "login" ? <LogIn size={18} /> : <UserPlus size={18} />}
            {loading ? "Connecting..." : mode === "login" ? "Login" : "Create account"}
          </button>
        </form>
      </section>

      {verificationOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div className="modal-panel confirm-panel" role="dialog" aria-modal="true" aria-labelledby="verify-title">
            <div className="section-header">
              <div>
                <h2 id="verify-title">Email verification</h2>
                <p>Enter the 6-digit code sent to {email}.</p>
              </div>
            </div>

            <label className="field">
              <span>Verification code</span>
              <input
                value={verificationCode}
                onChange={(event) => setVerificationCode(event.target.value.replace(/\D/g, "").slice(0, 6))}
                inputMode="numeric"
                pattern="\d{6}"
                placeholder="000000"
                required
              />
            </label>

            {verificationHint ? <div className="form-note">{verificationHint}</div> : null}
            {error ? <div className="form-error">{error}</div> : null}

            <div className="confirm-actions">
              <button className="btn ghost" disabled={verifying} onClick={() => setVerificationOpen(false)}>
                Cancel
              </button>
              <button className="btn primary" disabled={verifying || verificationCode.length !== 6} onClick={confirmRegistration}>
                {verifying ? "Checking..." : "Verify and create account"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {resetOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div className="modal-panel confirm-panel" role="dialog" aria-modal="true" aria-labelledby="reset-title">
            <div className="section-header">
              <div className="confirm-title-row">
                <KeyRound size={20} />
                <div>
                  <h2 id="reset-title">Reset password</h2>
                  <p>{resetStep === "email" ? "Enter your account email to receive a reset code." : "Enter the code and choose a new password."}</p>
                </div>
              </div>
            </div>

            {resetStep === "email" ? (
              <label className="field">
                <span>Email</span>
                <input
                  value={resetEmail}
                  onChange={(event) => setResetEmail(event.target.value)}
                  autoComplete="email"
                  type="email"
                  required
                />
              </label>
            ) : (
              <>
                <label className="field">
                  <span>Reset code</span>
                  <input
                    value={resetCode}
                    onChange={(event) => setResetCode(event.target.value.replace(/\D/g, "").slice(0, 6))}
                    inputMode="numeric"
                    pattern="\d{6}"
                    placeholder="000000"
                    required
                  />
                </label>
                <label className="field">
                  <span>New password</span>
                  <input
                    value={resetPassword}
                    onChange={(event) => setResetPassword(event.target.value)}
                    autoComplete="new-password"
                    type="password"
                    minLength={8}
                    required
                  />
                </label>
                <ul className="password-rules">
                  {resetPasswordRules.map((rule) => (
                    <li className={rule.passed ? "passed" : ""} key={rule.id}>
                      {rule.label}
                    </li>
                  ))}
                </ul>
                <label className="field">
                  <span>Confirm new password</span>
                  <input
                    value={resetConfirmPassword}
                    onChange={(event) => setResetConfirmPassword(event.target.value)}
                    autoComplete="new-password"
                    type="password"
                    minLength={8}
                    required
                  />
                </label>
              </>
            )}

            {resetHint ? <div className="form-note">{resetHint}</div> : null}
            {error ? <div className="form-error">{error}</div> : null}

            <div className="confirm-actions">
              <button className="btn ghost" disabled={resetLoading} onClick={() => setResetOpen(false)}>
                Cancel
              </button>
              {resetStep === "email" ? (
                <button className="btn primary" disabled={resetLoading || !resetEmail} onClick={requestResetCode}>
                  {resetLoading ? "Sending..." : "Send code"}
                </button>
              ) : (
                <button className="btn primary" disabled={resetLoading || resetCode.length !== 6} onClick={resetForgottenPassword}>
                  {resetLoading ? "Saving..." : "Change password"}
                </button>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
