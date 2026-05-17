import { ChevronDown, FilePlus2, Music, Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { listTabs } from "../api/tabeeApi";
import { errorMessage } from "../lib/errors";
import { clearRecentTabs, hideRecentTab, loadHiddenRecentTabIds, loadTabs, saveTabs, upsertTab } from "../lib/tabStore";
import { demoTab } from "../lib/demoTab";
import type { GeneratedTab } from "../types/tab";

export function DashboardPage() {
  const navigate = useNavigate();
  const [tabs, setTabs] = useState(loadTabs);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [recentsOpen, setRecentsOpen] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadBackendTabs() {
      setLoading(true);
      setError("");
      try {
        const backendTabs = await listTabs();
        if (!active) return;
        const merged = mergeTabs(backendTabs, loadTabs());
        saveTabs(merged);
        setTabs(merged);
      } catch (caught) {
        if (!active) return;
        setError(errorMessage(caught, "Could not load backend tabs."));
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadBackendTabs();
    return () => {
      active = false;
    };
  }, []);

  const filteredTabs = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return tabs;
    return tabs.filter((tab) => `${tab.title} ${tab.fileName} ${tab.instrument}`.toLowerCase().includes(normalized));
  }, [query, tabs]);

  function loadDemo() {
    setTabs(upsertTab(demoTab));
    navigate("/tabs/demo");
  }

  function removeFromRecents(tabId: string) {
    setTabs(hideRecentTab(tabId));
  }

  function clearRecents() {
    setTabs(clearRecentTabs());
  }

  return (
    <main className="page-grid">
      <section className="library-hero">
        <div>
          <p className="eyebrow">TaBee Library</p>
          <h2>Generated tabs become playable projects here.</h2>
          <p>
            Keep recent outputs close, reopen them for playback, and compare TaBee's generated tablature as the detection
            pipeline improves.
          </p>
        </div>
        <div className="hero-actions">
          <Link to="/generate" className="btn primary">
            <FilePlus2 size={18} />
            Generate tab
          </Link>
          <button className="btn ghost" onClick={loadDemo}>
            <Music size={18} />
            Open demo
          </button>
        </div>
      </section>

      <section className="library-panel">
        <div className={`recent-tabs-accordion${recentsOpen ? " open" : ""}`}>
          <div className="recent-tabs-trigger-row">
            <button
              className="search-section-trigger"
              type="button"
              aria-expanded={recentsOpen}
              onClick={() => setRecentsOpen((current) => !current)}
            >
              <span className="search-section-icon">
                <Music size={18} />
              </span>
              <span>
                <h2>Recent tabs</h2>
                <p>{loading ? "Loading backend tabs..." : `${tabs.length} available in your library`}</p>
              </span>
              <ChevronDown size={18} />
            </button>
            <button className="btn ghost" onClick={clearRecents} disabled={!tabs.length} title="Hide all recent tabs">
              <Trash2 size={17} />
              Clear recents
            </button>
          </div>

          {recentsOpen ? (
            <>
              <label className="search-box">
                <Search size={17} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search title, file, or instrument" />
              </label>

              {error ? <div className="form-error">{error}</div> : null}

              <div className="tab-list">
                {filteredTabs.map((tab) => (
                  <article className="tab-card" key={tab.id}>
                    <Link to={`/tabs/${tab.id}`}>
                      <span className="tab-card-title">{tab.title}</span>
                      <span className="tab-card-meta">
                        {tab.fileName} / {tab.instrument} / {tab.createdAt}
                      </span>
                    </Link>
                    <button
                      className="icon-btn subtle"
                      title="Remove from recents"
                      onClick={() => removeFromRecents(tab.id)}
                    >
                      <Trash2 size={17} />
                    </button>
                  </article>
                ))}
              </div>

              {!filteredTabs.length ? (
                <div className="empty-panel">
                  <h3>No tabs yet</h3>
                  <p>Generate a tab or open the demo to see the viewer and playback controls.</p>
                </div>
              ) : null}
            </>
          ) : null}
        </div>
      </section>

    </main>
  );
}

function mergeTabs(backendTabs: GeneratedTab[], cachedTabs: GeneratedTab[]) {
  const backendIds = new Set(backendTabs.map((tab) => tab.id));
  const hiddenIds = new Set(loadHiddenRecentTabIds());
  return [
    ...backendTabs.filter((tab) => !hiddenIds.has(tab.id)),
    ...cachedTabs.filter((tab) => !backendIds.has(tab.id) && !hiddenIds.has(tab.id)),
  ];
}
