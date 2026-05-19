import type { GeneratedTab } from "../types/tab";

const STORAGE_KEY = "tabee.generatedTabs";
const HIDDEN_RECENTS_KEY = "tabee.hiddenRecentTabs";

export function loadTabs(): GeneratedTab[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];

    const tabs = parsed.filter(isGeneratedTab).map(normalizeGeneratedTab);
    if (tabs.some((tab, index) => tab !== parsed[index])) {
      saveTabs(tabs);
    }
    return tabs;
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

function isGeneratedTab(value: unknown): value is GeneratedTab {
  return Boolean(value && typeof value === "object" && "id" in value && "alphaTex" in value);
}

function normalizeGeneratedTab(tab: GeneratedTab): GeneratedTab {
  const alphaTex = normalizeAlphaTexBars(tab.alphaTex);
  return alphaTex === tab.alphaTex ? tab : { ...tab, alphaTex };
}

function normalizeAlphaTexBars(alphaTex: string) {
  return alphaTex
    .split(/\r?\n/)
    .flatMap((line) => splitOversizedEighthNoteBar(line))
    .join("\n");
}

function splitOversizedEighthNoteBar(line: string) {
  const match = line.match(/^(:8\s+)(.+?)\s*\|\s*$/);
  if (!match) return [line];

  const notes = match[2].trim().split(/\s+/);
  if (notes.length <= 8) return [line];

  const bars: string[] = [];
  for (let index = 0; index < notes.length; index += 8) {
    bars.push(`${match[1]}${notes.slice(index, index + 8).join(" ")} |`);
  }
  return bars;
}
