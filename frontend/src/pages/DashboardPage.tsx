import { FilePlus2, Music, Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { deleteGeneratedTab, listPublicTabs, listTabs } from "../api/tabeeApi";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { errorMessage } from "../lib/errors";
import { loadTabs, removeTab, saveTabs, upsertTab } from "../lib/tabStore";
import { demoTab } from "../lib/demoTab";
import type { GeneratedTab } from "../types/tab";

export function DashboardPage() {
  const navigate = useNavigate();
  const [tabs, setTabs] = useState(loadTabs);
  const [publicTabs, setPublicTabs] = useState<GeneratedTab[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [deleteCandidate, setDeleteCandidate] = useState<GeneratedTab | null>(null);
  const [deletingTabId, setDeletingTabId] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadBackendTabs() {
      setLoading(true);
      setError("");
      try {
        const [backendTabs, nextPublicTabs] = await Promise.all([listTabs(), listPublicTabs()]);
        if (!active) return;
        const merged = mergeTabs(backendTabs, loadTabs());
        saveTabs(merged);
        setTabs(merged);
        setPublicTabs(nextPublicTabs);
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

  const discoverTabs = useMemo(() => {
    const ownedIds = new Set(tabs.map((tab) => tab.id));
    const discoverable = publicTabs.filter((tab) => !ownedIds.has(tab.id));
    const normalized = query.trim().toLowerCase();
    if (!normalized) return discoverable;
    return discoverable.filter((tab) => `${tab.title} ${tab.fileName} ${tab.instrument}`.toLowerCase().includes(normalized));
  }, [publicTabs, query, tabs]);

  function loadDemo() {
    setTabs(upsertTab(demoTab));
    navigate("/tabs/demo");
  }

  async function deleteTab(tab: GeneratedTab) {
    setDeletingTabId(tab.id);
    const previousTabs = tabs;
    const nextTabs = removeTab(tab.id);
    setTabs(nextTabs);

    if (tab.id === demoTab.id) {
      setDeleteCandidate(null);
      setDeletingTabId("");
      return;
    }

    try {
      await deleteGeneratedTab(tab.id);
      setDeleteCandidate(null);
    } catch (caught) {
      saveTabs(previousTabs);
      setTabs(previousTabs);
      setError(errorMessage(caught, "Could not delete tab."));
    } finally {
      setDeletingTabId("");
    }
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
        <div className="section-header">
          <div>
            <h2>Recent tabs</h2>
            <p>{loading ? "Loading backend tabs..." : `${tabs.length} available in your library`}</p>
          </div>
        </div>

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
                title="Delete"
                disabled={deletingTabId === tab.id}
                onClick={() => setDeleteCandidate(tab)}
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
      </section>

      <section className="library-panel">
        <div className="section-header">
          <div>
            <h2>Discover tabs</h2>
            <p>{loading ? "Loading public tabs..." : `${discoverTabs.length} tabs from other users`}</p>
          </div>
        </div>

        <div className="tab-list">
          {discoverTabs.map((tab) => (
            <article className="tab-card" key={tab.id}>
              <Link to={`/tabs/${tab.id}`}>
                <span className="tab-card-title">{tab.title}</span>
                <span className="tab-card-meta">
                  {tab.fileName} / {tab.instrument} / {tab.createdAt}
                </span>
              </Link>
            </article>
          ))}
        </div>

        {!loading && !discoverTabs.length ? (
          <div className="empty-panel compact">
            <h3>No public tabs found</h3>
            <p>As more users generate tabs, they will appear here.</p>
          </div>
        ) : null}
      </section>
      {deleteCandidate ? (
        <ConfirmDialog
          title="Delete tab?"
          message={`Delete "${deleteCandidate.title}"? This removes it from your recent library${deleteCandidate.id === demoTab.id ? "." : " and your account."}`}
          confirmLabel="Delete tab"
          loading={deletingTabId === deleteCandidate.id}
          tone="danger"
          onCancel={() => setDeleteCandidate(null)}
          onConfirm={() => deleteTab(deleteCandidate)}
        />
      ) : null}
    </main>
  );
}

function mergeTabs(backendTabs: GeneratedTab[], cachedTabs: GeneratedTab[]) {
  const backendIds = new Set(backendTabs.map((tab) => tab.id));
  return [...backendTabs, ...cachedTabs.filter((tab) => !backendIds.has(tab.id))];
}
