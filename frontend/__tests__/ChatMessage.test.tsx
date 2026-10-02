import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { ChatMessage } from "../src/components/chat/ChatMessage";
import { Citation } from "../src/types";

describe("ChatMessage Component", () => {
  it("renders user message without citations", () => {
    render(<ChatMessage role="user" content="What is the refund policy?" />);

    expect(screen.getByText("What is the refund policy?")).toBeInTheDocument();
  });

  it("renders assistant message with formatted citations and triggers callback", () => {
    const mockCitation: Citation = {
      source_id: "doc_1",
      source_index: 1,
      document_id: "d1",
      chunk_id: "c1",
      filename: "refund_policy.pdf",
      page_number: 3,
      chunk_index: 0,
      snippet: "Refunds are processed within 5 days.",
      verified: true,
    };

    const handleSelectCitation = jest.fn();

    render(
      <ChatMessage
        role="assistant"
        content="Refunds are processed within 5 business days [Source 1]."
        citations={[mockCitation]}
        onSelectCitation={handleSelectCitation}
      />
    );

    expect(
      screen.getByText(/Refunds are processed within 5 business days/i)
    ).toBeInTheDocument();
    
    // Check citation button
    const sourceButtons = screen.getAllByText(/\[Source 1\]/i);
    expect(sourceButtons.length).toBeGreaterThan(0);

    fireEvent.click(sourceButtons[0]);
    expect(handleSelectCitation).toHaveBeenCalledWith(mockCitation);
  });
});
