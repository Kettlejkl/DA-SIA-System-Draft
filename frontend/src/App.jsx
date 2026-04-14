/**
 * App.jsx — Root component with light/dark mode toggle
 */

import React, { useState, useEffect } from "react";
import { UploadPage } from "./pages/UploadPage";
import "./styles/global.css";

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggle = () => setTheme(t => t === "dark" ? "light" : "dark");

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-persian">نوشته‌خوان</span>
            <span className="logo-sep" />
            <span className="logo-latin">Persian Script Recognizer</span>
          </div>

          <div className="header-right">
            <span className="nav-chip">YOLOv8 + CRNN</span>
            <span className="nav-chip">TIP Manila · CIT 404A</span>
            <button
              className="theme-toggle"
              onClick={toggle}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? "☀️" : "🌙"}
            </button>
          </div>
        </div>
      </header>

      <main className="app-main">
        <UploadPage />
      </main>

      <footer className="app-footer">
        <span className="footer-left">
          <span className="footer-dot" />
          Flask · React · YOLOv8 · CRNN
        </span>
        <span className="footer-right">
          Carbonel · Estacio · Rosales — 2026
        </span>
      </footer>
    </div>
  );
}
