import { LogIn, UserPlus } from "lucide-react";
import type { FormEvent } from "react";
import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { login, register } from "../api/authApi";
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
  const [error, setError] = useState("");
  const registerPasswordRules = passwordRules(password, { username, email, fullName });

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
        await register({ username, email, fullName: fullName || undefined, password });
      }
      navigate(redirectTo, { replace: true });
    } catch (caught) {
      setError(errorMessage(caught, "Authentication failed."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-panel">
        <div className="auth-brand">
          <div className="brand-mark">T</div>
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
    </main>
  );
}
