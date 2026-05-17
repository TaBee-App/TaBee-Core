export type ThemeMode = "dark" | "light";

const THEME_KEY = "tabee.theme";

export function getStoredTheme(): ThemeMode {
  return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
}

export function applyTheme(theme: ThemeMode) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
}

export function initializeTheme() {
  applyTheme(getStoredTheme());
}
