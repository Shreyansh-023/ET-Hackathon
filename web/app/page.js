"use client";

import { useEffect, useMemo, useState } from "react";

const STAGE_ORDER = ["ingest", "plan", "assets", "audio", "render", "export"];
const STAGE_LABELS = {
  ingest: "Ingest",
  plan: "Plan",
  assets: "Assets",
  audio: "Audio",
  render: "Render",
  export: "Export"
};

function formatLanguage(language) {
  if (!language) {
    return "Unknown";
  }
  return language.charAt(0).toUpperCase() + language.slice(1);
}

export default function Home() {
  const [inputText, setInputText] = useState("");
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!jobId) {
      return undefined;
    }

    let active = true;
    const loadStatus = async () => {
      try {
        const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
        if (!response.ok) {
          throw new Error("Status fetch failed");
        }
        const payload = await response.json();
        if (active) {
          setStatus(payload);
          setError("");
        }
      } catch (err) {
        if (active) {
          setError("Unable to load job status.");
        }
      }
    };

    loadStatus();
    const interval = setInterval(loadStatus, 4000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [jobId]);

  const stageItems = useMemo(() => {
    return STAGE_ORDER.map((stage) => {
      const entry = status?.stages?.[stage];
      const stageStatus = entry?.status || (jobId ? "pending" : "pending");
      const normalized = stageStatus === "idle" ? "pending" : stageStatus;
      return {
        stage,
        label: STAGE_LABELS[stage],
        status: normalized
      };
    });
  }, [jobId, status]);

  const currentStage = status?.currentStage || "created";
  const currentStageLabel = STAGE_LABELS[currentStage] || "Queued";
  const detectedLanguage = status?.language ? formatLanguage(status.language) : "Unknown";
  const videoUrl = status?.videoUrl || "";
  const thumbnailUrl = status?.thumbnailUrl || "";
  const failedStage = stageItems.find((item) => item.status === "failed");

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);
    setError("");
    setStatus(null);
    setJobId("");

    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText })
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.error || "Generation failed");
      }

      setJobId(payload.jobId);
      setStatus(payload.status || null);
      setInputText("");
    } catch (err) {
      setError(err?.message || "Unable to start the pipeline.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="page">
      <header className="site-header">
        <div className="inner header-bar">
          <span className="header-spacer" aria-hidden="true" />
          <div className="header-center">
            <img
              className="main-logo"
              src="/Main%20header.jpeg"
              alt="The Economic Times"
            />
          </div>
          <nav className="header-nav" aria-label="Automation workflow">
            <span className="nav-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
                <path d="M12 2l2.6 5.7 6.2 1-4.4 4.3 1 6.3L12 16.8 6.6 19.3l1-6.3L3.2 8.7l6.2-1L12 2z" />
              </svg>
            </span>
            <span className="header-nav-text">Text-to-reel automation workflow</span>
          </nav>
        </div>
      </header>

      <main className="main">
        <section className="hero-intro">
          <span className="hero-chip">
            <span className="hero-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
                <path d="M12 2l2.6 5.7 6.2 1-4.4 4.3 1 6.3L12 16.8 6.6 19.3l1-6.3L3.2 8.7l6.2-1L12 2z" />
              </svg>
            </span>
            Modern News AI Video Editor
          </span>
          <h1>Textual News Article to Modern News Reel in minutes.</h1>
          <p>
            Paste a story and let the pipeline plan scenes, generate assets,
            synthesize audio, and deliver a publish-ready reel.
          </p>
        </section>
        <div className="hero-grid">
          <section className="panel form-panel">
            <div>
              <h2>News Article Input</h2>
              <p>Paste article text. Language is detected automatically.</p>
            </div>
            <form onSubmit={handleSubmit}>
              <label className="label" htmlFor="article-text">
                News Article Text
              </label>
              <textarea
                id="article-text"
                value={inputText}
                onChange={(event) => setInputText(event.target.value)}
                placeholder="Paste the full article text here..."
                required
              />
              <div className="button-row">
                <button type="submit" disabled={!inputText.trim() || isSubmitting}>
                  {isSubmitting ? "Generating..." : "Generate"}
                </button>
                <span className="badge">Detected: {detectedLanguage}</span>
              </div>
            </form>

            <div className="status-card">
              <div className="status-row">
                <span>Current stage</span>
                <strong>{currentStageLabel}</strong>
              </div>
              <div className="stage-list">
                {stageItems.map((item) => (
                  <div
                    key={item.stage}
                    className={`stage-item ${item.status}`}
                  >
                    <span className="stage-dot" />
                    <span>{item.label}</span>
                    <span className="stage-status">{item.status}</span>
                  </div>
                ))}
              </div>
              {failedStage ? (
                <div className="error">
                  {failedStage.label} failed. Check logs for details.
                </div>
              ) : null}
              {error ? <div className="error">{error}</div> : null}
            </div>
          </section>

          <section className="panel reel-panel">
            <div>
              <h2>Live Reel Preview</h2>
              <p>Preview renders here once export completes.</p>
            </div>
            <div className="reel-frame">
              {videoUrl ? (
                <video
                  src={videoUrl}
                  poster={thumbnailUrl || undefined}
                  controls
                  playsInline
                />
              ) : (
                <div className="reel-placeholder">
                  <strong>No render yet</strong>
                  <div>Submit an article to start the pipeline.</div>
                </div>
              )}
            </div>
            <div className="reel-meta">
              <span>{jobId ? `Job ${jobId}` : "Ready"}</span>
              <span>{jobId ? `Stage: ${currentStageLabel}` : "Waiting"}</span>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
