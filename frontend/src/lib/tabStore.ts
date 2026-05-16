import type { GeneratedTab } from "../types/tab";

const STORAGE_KEY = "tabee.generatedTabs";

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
