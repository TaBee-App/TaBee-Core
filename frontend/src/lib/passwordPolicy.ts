export interface PasswordIdentity {
  username?: string;
  email?: string;
  fullName?: string;
}

export interface PasswordRule {
  id: string;
  label: string;
  passed: boolean;
}

const sequences = [
  "0123", "1234", "2345", "3456", "4567", "5678", "6789",
  "9876", "8765", "7654", "6543", "5432", "4321", "3210",
  "abcd", "bcde", "cdef", "qwer", "asdf", "zxcv"
];

export function passwordRules(password: string, identity: PasswordIdentity): PasswordRule[] {
  const normalized = password.toLowerCase();
  return [
    {
      id: "length",
      label: "At least 8 characters",
      passed: password.length >= 8
    },
    {
      id: "lower",
      label: "Includes a lowercase letter",
      passed: /[a-z]/.test(password)
    },
    {
      id: "upper",
      label: "Includes an uppercase letter",
      passed: /[A-Z]/.test(password)
    },
    {
      id: "digit",
      label: "Includes a number",
      passed: /\d/.test(password)
    },
    {
      id: "special",
      label: "Includes a special character",
      passed: /[^A-Za-z0-9]/.test(password)
    },
    {
      id: "repeat",
      label: "No character repeated 3 times in a row",
      passed: !/(.)\1\1/.test(normalized)
    },
    {
      id: "sequence",
      label: "No simple sequences like 1234 or qwer",
      passed: !sequences.some((sequence) => normalized.includes(sequence))
    },
    {
      id: "identity",
      label: "Does not include your username, email, or name",
      passed: !containsIdentity(normalized, identity)
    }
  ];
}

export function passwordPolicyError(password: string, identity: PasswordIdentity) {
  const failed = passwordRules(password, identity).find((rule) => !rule.passed);
  return failed ? failed.label : "";
}

function containsIdentity(password: string, identity: PasswordIdentity) {
  const tokens = [
    identity.username,
    ...(identity.email || "").split(/[@._\-+]+/),
    ...(identity.fullName || "").split(/\s+/)
  ];
  return tokens.some((token) => {
    const normalized = token?.trim().toLowerCase() || "";
    return normalized.length >= 3 && password.includes(normalized);
  });
}
