import React from "react";
import { Citation, MessageCitation } from "../../types";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[] | MessageCitation[] | null;
  onSelectCitation?: (citation: Citation) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  content,
  citations,
  onSelectCitation,
}) => {
  const isUser = role === "user";

  const renderFormattedContent = (text: string) => {
    // Replace [Source X] with clickable source badges
    const parts = text.split(/(\[Source \d+\])/g);
    return parts.map((part, idx) => {
      const match = part.match(/\[Source (\d+)\]/);
      if (match) {
        const sourceIdx = parseInt(match[1], 10);
        const matchedCitation = citations?.find(
          (c) =>
            (c as Citation).source_index === sourceIdx ||
            (c as MessageCitation).source_index === sourceIdx
        );

        return (
          <button
            key={idx}
            onClick={() => {
              if (matchedCitation && onSelectCitation) {
                onSelectCitation(matchedCitation as Citation);
              }
            }}
            style={{
              display: "inline-flex",
              alignItems: "center",
              background: "rgba(99, 102, 241, 0.2)",
              color: "#a5b4fc",
              border: "1px solid rgba(99, 102, 241, 0.4)",
              borderRadius: "4px",
              padding: "0.1rem 0.4rem",
              fontSize: "0.75rem",
              fontWeight: 600,
              margin: "0 0.2rem",
              cursor: matchedCitation ? "pointer" : "default",
              transition: "all 0.15s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(99, 102, 241, 0.35)";
              e.currentTarget.style.transform = "translateY(-1px)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(99, 102, 241, 0.2)";
              e.currentTarget.style.transform = "none";
            }}
          >
            {part}
          </button>
        );
      }
      return <span key={idx}>{part}</span>;
    });
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        marginBottom: "1.5rem",
        width: "100%",
      }}
    >
      <div
        style={{
          display: "flex",
          gap: "0.75rem",
          maxWidth: "85%",
          alignItems: "flex-start",
          flexDirection: isUser ? "row-reverse" : "row",
        }}
      >
        {/* Avatar */}
        <div
          style={{
            width: "34px",
            height: "34px",
            borderRadius: "50%",
            background: isUser ? "var(--bg-card)" : "var(--accent-gradient)",
            border: isUser ? "1px solid var(--border-subtle)" : "none",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "0.9rem",
            flexShrink: 0,
            boxShadow: isUser ? "none" : "0 0 10px rgba(99, 102, 241, 0.3)",
          }}
        >
          {isUser ? "👤" : "🤖"}
        </div>

        {/* Bubble */}
        <div
          style={{
            background: isUser
              ? "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"
              : "var(--bg-card)",
            border: isUser
              ? "1px solid rgba(255, 255, 255, 0.1)"
              : "1px solid var(--border-subtle)",
            borderRadius: isUser
              ? "16px 4px 16px 16px"
              : "4px 16px 16px 16px",
            padding: "1rem 1.25rem",
            color: "var(--text-primary)",
            fontSize: "0.95rem",
            lineHeight: 1.6,
            boxShadow: "var(--shadow-sm)",
          }}
        >
          <div style={{ whiteSpace: "pre-wrap" }}>
            {renderFormattedContent(content)}
          </div>

          {/* Citations list tray */}
          {!isUser && citations && citations.length > 0 && (
            <div
              style={{
                marginTop: "1rem",
                paddingTop: "0.75rem",
                borderTop: "1px solid rgba(255, 255, 255, 0.08)",
              }}
            >
              <div
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  color: "var(--text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: "0.5rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.4rem",
                }}
              >
                <span>📚</span> Verified Grounded Sources:
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                {citations.map((c, i) => {
                  const citationObj = c as Citation;
                  const srcIdx = citationObj.source_index || i + 1;
                  return (
                    <button
                      key={i}
                      onClick={() => onSelectCitation && onSelectCitation(citationObj)}
                      className="btn-secondary"
                      style={{
                        padding: "0.25rem 0.6rem",
                        fontSize: "0.75rem",
                        borderRadius: "var(--radius-sm)",
                        background: "rgba(0, 0, 0, 0.3)",
                      }}
                    >
                      <span style={{ color: "var(--text-accent)", fontWeight: 700 }}>
                        [Source {srcIdx}]
                      </span>
                      <span>{citationObj.filename}</span>
                      {citationObj.page_number && (
                        <span style={{ color: "var(--text-muted)" }}>
                          (p.{citationObj.page_number})
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
