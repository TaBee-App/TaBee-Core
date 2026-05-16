export function SettingsPage() {
  return (
    <main className="settings-page">
      <section className="settings-panel">
        <p className="eyebrow">Settings</p>
        <h2>Viewer preferences</h2>
        <p>
          This page is intentionally small for the first frontend milestone. The settings that matter next are theme,
          default instrument, default tuning, cursor style, and notation scale.
        </p>

        <div className="setting-row">
          <div>
            <strong>Default backend</strong>
            <span>Development proxy points `/api` to `127.0.0.1:8080`.</span>
          </div>
          <code>vite.config.ts</code>
        </div>

        <div className="setting-row">
          <div>
            <strong>Renderer</strong>
            <span>AlphaTab renders generated AlphaTex as tablature.</span>
          </div>
          <code>@coderline/alphatab</code>
        </div>
      </section>
    </main>
  );
}
