import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { DocumentUpload } from "../src/components/documents/DocumentUpload";
import { api } from "../src/lib/api";

describe("DocumentUpload Component", () => {
  it("rejects unsupported file formats with user error", async () => {
    const handleSuccess = jest.fn();
    render(<DocumentUpload onUploadSuccess={handleSuccess} />);

    const invalidFile = new File(["binary content"], "script.exe", {
      type: "application/x-msdownload",
    });

    const fileInput = screen.getByTestId("file-upload-input");
    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    await waitFor(() => {
      expect(screen.getByText(/unsupported file type '\.exe'/i)).toBeInTheDocument();
    });
    expect(handleSuccess).not.toHaveBeenCalled();
  });

  it("uploads valid PDF and calls onUploadSuccess callback", async () => {
    const handleSuccess = jest.fn();
    const mockDoc = {
      id: "doc-123",
      user_id: "user-1",
      filename: "test.pdf",
      file_type: ".pdf",
      file_size: 1024,
      status: "PENDING" as const,
      total_chunks: 0,
      created_at: "2026-01-01",
      updated_at: "2026-01-01",
    };

    jest.spyOn(api, "uploadDocument").mockResolvedValueOnce(mockDoc);

    render(<DocumentUpload onUploadSuccess={handleSuccess} />);

    const validFile = new File(["dummy pdf content"], "test.pdf", {
      type: "application/pdf",
    });

    const fileInput = screen.getByTestId("file-upload-input");
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    await waitFor(() => {
      expect(handleSuccess).toHaveBeenCalledWith(mockDoc);
      expect(screen.getByText(/uploaded successfully/i)).toBeInTheDocument();
    });
  });
});
