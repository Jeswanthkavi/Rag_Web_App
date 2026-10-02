import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { ChatSidebar } from "../src/components/chat/ChatSidebar";
import { Conversation } from "../src/types";

describe("ChatSidebar Component", () => {
  const mockConversations: Conversation[] = [
    {
      id: "conv-1",
      user_id: "u1",
      title: "Warranty Inquiries",
      created_at: "2026-01-01",
      updated_at: "2026-01-01",
    },
    {
      id: "conv-2",
      user_id: "u1",
      title: "Kubernetes Specs",
      created_at: "2026-01-02",
      updated_at: "2026-01-02",
    },
  ];

  it("renders conversation items and triggers callbacks", () => {
    const handleSelect = jest.fn();
    const handleNew = jest.fn();
    const handleDelete = jest.fn();

    render(
      <ChatSidebar
        conversations={mockConversations}
        activeConversationId="conv-1"
        isLoading={false}
        onSelectConversation={handleSelect}
        onNewConversation={handleNew}
        onDeleteConversation={handleDelete}
      />
    );

    expect(screen.getByText("Warranty Inquiries")).toBeInTheDocument();
    expect(screen.getByText("Kubernetes Specs")).toBeInTheDocument();

    // Click new session button
    fireEvent.click(screen.getByText(/new chat session/i));
    expect(handleNew).toHaveBeenCalled();

    // Click second conversation
    fireEvent.click(screen.getByText("Kubernetes Specs"));
    expect(handleSelect).toHaveBeenCalledWith("conv-2");
  });
});
