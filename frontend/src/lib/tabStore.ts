import type { GeneratedTab } from "../types/tab";

const STORAGE_KEY = "tabee.generatedTabs";
const HIDDEN_RECENTS_KEY = "tabee.hiddenRecentTabs";

export function loadTabs(): GeneratedTab[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveTabs(tabs: GeneratedTab[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(tabs.slice(0, 24)));
}

export function upsertTab(tab: GeneratedTab) {
  unhideRecentTab(tab.id);
  const next = [tab, ...loadTabs().filter((item) => item.id !== tab.id)];
  saveTabs(next);
  return next;
}

export function getTab(tabId: string) {
  return loadTabs().find((tab) => tab.id === tabId) ?? null;
}

export function removeTab(tabId: string) {
  const next = loadTabs().filter((tab) => tab.id !== tabId);
  saveTabs(next);
  return next;
}

export function clearTabs() {
  localStorage.removeItem(STORAGE_KEY);
}

export function loadHiddenRecentTabIds() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HIDDEN_RECENTS_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.filter((id) => typeof id === "string") : [];
  } catch {
    return [];
  }
}

export function hideRecentTab(tabId: string) {
  const hiddenIds = new Set(loadHiddenRecentTabIds());
  hiddenIds.add(tabId);
  localStorage.setItem(HIDDEN_RECENTS_KEY, JSON.stringify([...hiddenIds]));
  return removeTab(tabId);
}

export function clearRecentTabs() {
  const hiddenIds = new Set(loadHiddenRecentTabIds());
  for (const tab of loadTabs()) {
    hiddenIds.add(tab.id);
  }
  localStorage.setItem(HIDDEN_RECENTS_KEY, JSON.stringify([...hiddenIds]));
  clearTabs();
  return [];
}

function unhideRecentTab(tabId: string) {
  const hiddenIds = new Set(loadHiddenRecentTabIds());
  if (!hiddenIds.delete(tabId)) return;
  localStorage.setItem(HIDDEN_RECENTS_KEY, JSON.stringify([...hiddenIds]));
}
