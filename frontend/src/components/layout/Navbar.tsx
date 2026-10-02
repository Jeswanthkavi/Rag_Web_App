import React from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import { useAuth } from "../../context/AuthContext";

export const Navbar: React.FC = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const router = useRouter();

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  return (
    <header
      style={{
        height: "64px",
        borderBottom: "1px solid var(--border-subtle)",
        background: "rgba(15, 20, 31, 0.8)",
        backdropFilter: "blur(12px)",
        position: "sticky",
        top: 0,
        zIndex: 50,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 1.5rem",
      }}
    >
      {/* Brand Logo */}
      <Link
        href="/"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
          fontWeight: 700,
          fontSize: "1.15rem",
          letterSpacing: "-0.02em",
        }}
      >
        <div
          style={{
            width: "32px",
            height: "32px",
            borderRadius: "8px",
            background: "var(--accent-gradient)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#ffffff",
            boxShadow: "0 0 12px rgba(99, 102, 241, 0.4)",
          }}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </div>
        <span>
          Advanced <span style={{ color: "var(--accent-secondary)" }}>RAG</span>
        </span>
      </Link>

      {/* Navigation Links */}
      {isAuthenticated && (
        <nav style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Link
            href="/"
            style={{
              padding: "0.5rem 1rem",
              borderRadius: "var(--radius-md)",
              fontSize: "0.9rem",
              fontWeight: 500,
              background:
                router.pathname === "/" ? "var(--bg-card)" : "transparent",
              color:
                router.pathname === "/"
                  ? "var(--text-primary)"
                  : "var(--text-secondary)",
              border:
                router.pathname === "/"
                  ? "1px solid var(--border-subtle)"
                  : "1px solid transparent",
              transition: "all 0.2s ease",
            }}
          >
            💬 Chat Assistant
          </Link>
          <Link
            href="/documents"
            style={{
              padding: "0.5rem 1rem",
              borderRadius: "var(--radius-md)",
              fontSize: "0.9rem",
              fontWeight: 500,
              background:
                router.pathname === "/documents"
                  ? "var(--bg-card)"
                  : "transparent",
              color:
                router.pathname === "/documents"
                  ? "var(--text-primary)"
                  : "var(--text-secondary)",
              border:
                router.pathname === "/documents"
                  ? "1px solid var(--border-subtle)"
                  : "1px solid transparent",
              transition: "all 0.2s ease",
            }}
          >
            📄 Knowledge Base
          </Link>
        </nav>
      )}

      {/* User Actions */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        {isAuthenticated && user ? (
          <>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                fontSize: "0.85rem",
                color: "var(--text-secondary)",
                background: "rgba(255, 255, 255, 0.04)",
                padding: "0.35rem 0.75rem",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <div
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  background: "var(--status-success)",
                  boxShadow: "0 0 8px var(--status-success)",
                }}
              />
              <span>{user.email}</span>
            </div>
            <button
              onClick={handleLogout}
              className="btn-secondary"
              style={{ padding: "0.4rem 0.9rem", fontSize: "0.85rem" }}
            >
              Logout
            </button>
          </>
        ) : (
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Link
              href="/login"
              className="btn-secondary"
              style={{ padding: "0.4rem 0.9rem", fontSize: "0.85rem" }}
            >
              Log In
            </Link>
            <Link
              href="/register"
              className="btn-primary"
              style={{ padding: "0.4rem 0.9rem", fontSize: "0.85rem" }}
            >
              Register
            </Link>
          </div>
        )}
      </div>
    </header>
  );
};
