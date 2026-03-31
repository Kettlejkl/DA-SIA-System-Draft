/**
 * StatusBadge.jsx — pipeline state indicator
 * `inline` prop: renders without the outer .status-wrap div (for use in panel headers)
 */

import React from "react";

const STATUS_MAP = {
  idle:        { label: "Ready",        cls: "badge--idle" },
  recognizing: { label: "Recognising",  cls: "badge--busy" },
  translating: { label: "Translating",  cls: "badge--busy" },
  done:        { label: "Complete",     cls: "badge--done" },
  error:       { label: "Error",        cls: "badge--error" },
};

export function StatusBadge({ status, error, inline = false }) {
  const { label, cls } = STATUS_MAP[status] || STATUS_MAP.idle;

  const badge = (
    <span className={["badge", cls].join(" ")}>{label}</span>
  );

  if (inline) return badge;

  return (
    <div className="status-wrap">
      {badge}
      {status === "error" && error && (
        <p className="error-msg">{error}</p>
      )}
    </div>
  );
}
